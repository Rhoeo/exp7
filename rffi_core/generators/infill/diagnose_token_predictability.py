"""Estimate exact-token predictability from generator-train transition counts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(r"E:\data_cache\rffi_v1")


def stable_integer(seed, value):
    return int(
        hashlib.sha256(("%d|%s" % (seed, value)).encode("utf-8")).hexdigest()[:16],
        16,
    ) % (2**32)


def topk_accuracy(scores, targets, k):
    top = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == targets[:, None], axis=1)))


def generator_train_indices(rows):
    return np.asarray(
        [int(row["cache_index"]) for row in rows if row["split"] == "generator_train"],
        dtype=np.int64,
    )


def token_statistics(tokens, indices, codebook_size, chunk_rows=256):
    usage = np.zeros(codebook_size, dtype=np.int64)
    transitions = np.zeros((codebook_size, codebook_size), dtype=np.int64)
    for start in range(0, len(indices), chunk_rows):
        batch = np.asarray(tokens[indices[start : start + chunk_rows]], dtype=np.int64)
        usage += np.bincount(batch.reshape(-1), minlength=codebook_size)
        pair_ids = batch[:, :-1] * codebook_size + batch[:, 1:]
        transitions += np.bincount(
            pair_ids.reshape(-1), minlength=codebook_size * codebook_size
        ).reshape(codebook_size, codebook_size)
    return usage, transitions


def main():
    seed = 20260904
    index_path = CACHE_ROOT / "wifib/window_index.csv"
    token_path = CACHE_ROOT / "tokens/wifib_vq_p1_k1024/tokens.npy"
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    split_config = json.loads(
        (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
    )
    dev_ids = set(split_config["roles"]["generator_search_dev"])
    dev_rows = [row for row in rows if row["sample_id"] in dev_ids]
    dev_rows = sorted(
        dev_rows,
        key=lambda row: stable_integer(seed, row["sample_id"]),
    )[:512]
    tokens = np.load(str(token_path), mmap_mode="r", allow_pickle=False)
    usage, transitions = token_statistics(tokens, generator_train_indices(rows), 1024)
    rng = np.random.RandomState(seed)
    triples = []
    for row in dev_rows:
        sequence = np.asarray(tokens[int(row["cache_index"])], dtype=np.int64)
        positions = rng.choice(np.arange(1, len(sequence) - 1), size=16, replace=False)
        triples.extend(
            (int(sequence[position - 1]), int(sequence[position]), int(sequence[position + 1]))
            for position in positions
        )
    triples = np.asarray(triples, dtype=np.int64)
    left = triples[:, 0]
    target = triples[:, 1]
    right = triples[:, 2]
    log_forward = np.log(transitions + 1.0) - np.log(
        transitions.sum(axis=1, keepdims=True) + 1024.0
    )
    left_scores = log_forward[left]
    right_scores = log_forward[:, right].T
    bidirectional_scores = left_scores + right_scores
    unigram_scores = np.broadcast_to(np.log(usage + 1.0)[None, :], left_scores.shape)
    metrics = {}
    for name, scores in (
        ("unigram", unigram_scores),
        ("left_bigram", left_scores),
        ("right_bigram", right_scores),
        ("bidirectional_bigram", bidirectional_scores),
    ):
        metrics[name] = {
            "top1": topk_accuracy(scores, target, 1),
            "top5": topk_accuracy(scores, target, 5),
            "top16": topk_accuracy(scores, target, 16),
        }
    report = {
        "schema_version": "wifib-token-predictability-diagnostic-v1",
        "source_statistics": "generator_train only",
        "evaluation_role": "generator_search_dev",
        "evaluation_sequences": len(dev_rows),
        "evaluation_positions": len(triples),
        "victim_queries": 0,
        "evaluator_queries": 0,
        "policy_gate_used": False,
        "final_test_used": False,
        "metrics": metrics,
    }
    output = ROOT / "reports/next_stage/token_predictability_diagnostic.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
