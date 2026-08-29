#!/usr/bin/env python
"""Measure whether RF-GPT likelihood actually uses the device condition."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.token_datasets import WiFiBTokenDataset
from rffi_core.generators.rfgpt.models import build_rfgpt
from rffi_core.generators.rfgpt.train_rfgpt import (
    TransitionCropDataset,
    deterministic_subset,
)
from rffi_core.generators.vqvae.train_reconstruction import choose_device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument(
        "--token-cache",
        default="E:/data_cache/rffi_v1/tokens/wifib_vq_p1_k1024/tokens.npy",
    )
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260832)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_rfgpt(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    config = load_json(Path(args.data_config).resolve())
    index_path = Path(config["cache_root"]).resolve() / "wifib" / "window_index.csv"
    base = WiFiBTokenDataset(
        index_path, Path(args.token_cache).resolve(), "reward_validation"
    )
    if base.label_map != checkpoint["label_map"]:
        raise ValueError("validation label map differs from checkpoint")
    base = deterministic_subset(base, args.max_samples, args.seed)
    dataset = TransitionCropDataset(
        base, checkpoint["model_config"]["context_length"], args.seed + 1
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    correct_loss = 0.0
    shuffled_loss = 0.0
    probability_l1 = 0.0
    token_count = 0
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["input_tokens"].to(device=device, dtype=torch.long)
            targets = batch["target_tokens"].to(device=device, dtype=torch.long)
            labels = batch["label"].to(device=device, dtype=torch.long)
            offsets = batch["position_offset"].to(device=device, dtype=torch.long)
            shuffled = torch.roll(labels, shifts=1)
            if torch.equal(shuffled, labels):
                shuffled = (labels + 1) % checkpoint["model_config"]["num_devices"]
            correct_logits = model(inputs, labels, offsets)
            shuffled_logits = model(inputs, shuffled, offsets)
            correct_loss += float(
                F.cross_entropy(
                    correct_logits.reshape(-1, correct_logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                ).cpu()
            )
            shuffled_loss += float(
                F.cross_entropy(
                    shuffled_logits.reshape(-1, shuffled_logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                ).cpu()
            )
            probability_l1 += float(
                (
                    torch.softmax(correct_logits, dim=-1)
                    - torch.softmax(shuffled_logits, dim=-1)
                )
                .abs()
                .sum(dim=-1)
                .sum()
                .cpu()
            )
            token_count += targets.numel()
    correct_nll = correct_loss / max(1, token_count)
    shuffled_nll = shuffled_loss / max(1, token_count)
    delta = shuffled_nll - correct_nll
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "examples": len(dataset),
        "tokens": token_count,
        "correct_condition_nll": correct_nll,
        "correct_condition_perplexity": math.exp(correct_nll),
        "shuffled_condition_nll": shuffled_nll,
        "shuffled_condition_perplexity": math.exp(shuffled_nll),
        "shuffled_minus_correct_nll": delta,
        "mean_output_probability_l1": probability_l1 / max(1, token_count),
        "condition_usage": (
            "strong" if delta >= 0.05 else "weak" if delta >= 0.01 else "negligible"
        ),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "correct NLL %.5f | shuffled NLL %.5f | delta %.5f | condition %s"
        % (correct_nll, shuffled_nll, delta, report["condition_usage"]),
        flush=True,
    )
    print("report: %s" % output_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

