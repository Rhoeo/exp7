"""Deterministic local-context datasets for masked and neighbor denoising."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rffi_core.data.datasets import build_label_map
from rffi_core.generators.token_candidates import TokenCandidateGraph


def _integer_seed(seed: int, epoch: int, sample_id: str) -> int:
    payload = "%d|%d|%s" % (int(seed), int(epoch), sample_id)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _stable_select(rows, count, seed, namespace):
    if count is None or count >= len(rows):
        return list(rows)
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            ("%d|%s|%s" % (seed, namespace, row["sample_id"])).encode("utf-8")
        ).hexdigest(),
    )[:count]


def load_rows_for_role(window_index_path, split_config_path, role, max_samples, seed):
    with Path(window_index_path).open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    if role == "generator_train":
        rows = [row for row in all_rows if row["split"] == "generator_train"]
    else:
        split_config = json.loads(Path(split_config_path).read_text(encoding="utf-8"))
        allowed = set(split_config["roles"][role])
        rows = [row for row in all_rows if row["sample_id"] in allowed]
        if len(rows) != len(allowed):
            raise ValueError("role sample IDs do not match the window index")
    return _stable_select(rows, max_samples, seed, role), all_rows


class MaskedTokenWindowDataset(Dataset):
    def __init__(
        self,
        token_cache_path,
        window_index_path,
        split_config_path,
        role,
        candidate_graph_path,
        context_length=256,
        max_samples=None,
        seed=20260902,
        neighbor_corruption_probability=0.5,
        transition_supported=True,
        candidate_top_k=16,
    ):
        self.tokens = np.load(str(token_cache_path), mmap_mode="r", allow_pickle=False)
        self.rows, all_rows = load_rows_for_role(
            window_index_path, split_config_path, role, max_samples, int(seed)
        )
        self.label_map = build_label_map(row["device_id"] for row in all_rows)
        self.graph = TokenCandidateGraph.load(candidate_graph_path)
        self.context_length = int(context_length)
        self.sequence_length = int(self.tokens.shape[1])
        if self.context_length <= 0 or self.context_length > self.sequence_length:
            raise ValueError("invalid context length")
        self.seed = int(seed)
        self.epoch = 0
        self.neighbor_corruption_probability = float(neighbor_corruption_probability)
        self.transition_supported = bool(transition_supported)
        self.candidate_top_k = int(candidate_top_k)
        self.role = role

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def _target_positions(self, rng):
        pattern = int(rng.randint(0, 4))
        if pattern == 0:
            count, contiguous = 1, True
        elif pattern == 1:
            count, contiguous = 2, False
        elif pattern == 2:
            count, contiguous = 2, True
        else:
            count, contiguous = 4, True
        if contiguous:
            start = int(rng.randint(0, self.context_length - count + 1))
            return np.arange(start, start + count, dtype=np.int64)
        return np.sort(rng.choice(self.context_length, size=count, replace=False))

    def __getitem__(self, index):
        row = self.rows[int(index)]
        rng = np.random.RandomState(_integer_seed(self.seed, self.epoch, row["sample_id"]))
        max_start = self.sequence_length - self.context_length
        start = int(rng.randint(0, max_start + 1))
        original = np.asarray(
            self.tokens[int(row["cache_index"]), start : start + self.context_length],
            dtype=np.int64,
        ).copy()
        inputs = original.copy()
        targets = np.full(self.context_length, -100, dtype=np.int64)
        indicator = np.zeros(self.context_length, dtype=np.int64)
        positions = self._target_positions(rng)
        corruption_modes = []
        for position in positions:
            token = int(original[position])
            targets[position] = token
            indicator[position] = 1
            candidates = self.graph.candidates(
                token,
                top_k=self.candidate_top_k,
                transition_supported=self.transition_supported,
            )
            use_neighbor = (
                len(candidates) > 0
                and rng.rand() < self.neighbor_corruption_probability
            )
            if use_neighbor:
                inputs[position] = int(candidates[int(rng.randint(0, len(candidates)))])
                corruption_modes.append("neighbor")
            else:
                inputs[position] = self.graph.codebook_size
                corruption_modes.append("mask")
        global_positions = (
            np.arange(start, start + self.context_length, dtype=np.float32)
            / float(max(1, self.sequence_length - 1))
        )
        return {
            "tokens": torch.from_numpy(inputs),
            "targets": torch.from_numpy(targets),
            "target_indicator": torch.from_numpy(indicator),
            "global_positions": torch.from_numpy(global_positions),
            "device_label": int(self.label_map[row["device_id"]]),
            "sample_id": row["sample_id"],
            "cache_index": int(row["cache_index"]),
            "context_start": start,
            "corruption_modes": ",".join(corruption_modes),
        }
