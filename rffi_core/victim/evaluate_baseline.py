#!/usr/bin/env python
"""Evaluate a frozen Stage G0 classifier on protected data splits."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.victim.models import build_model
from rffi_core.victim.train_baseline import choose_device, make_dataset


def classification_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    actual = confusion.sum(axis=1).astype(np.float64)
    precision = true_positive / np.maximum(predicted, 1.0)
    recall = true_positive / np.maximum(actual, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(true_positive.sum() / max(1.0, actual.sum())),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
    }


def evaluate(model, loader, device, class_count):
    criterion = nn.CrossEntropyLoss(reduction="sum")
    confusion = np.zeros((class_count, class_count), dtype=np.int64)
    total_loss = 0.0
    total = 0
    started = time.time()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["iq"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
            logits = model(inputs)
            total_loss += float(criterion(logits, targets).detach().cpu())
            predictions = logits.argmax(dim=1)
            encoded = targets * class_count + predictions
            counts = torch.bincount(encoded, minlength=class_count * class_count)
            confusion += counts.reshape(class_count, class_count).cpu().numpy()
            total += int(targets.shape[0])
    metrics = classification_metrics(confusion)
    metrics.update(
        {
            "loss": total_loss / max(1, total),
            "examples": total,
            "seconds": time.time() - started,
            "confusion_matrix": confusion.tolist(),
        }
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("reward_validation", "defense_train", "final_test"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    training_arguments = checkpoint["arguments"]
    dataset_name = training_arguments["dataset"]
    equalized = int(training_arguments.get("equalized", 0))
    model_name = training_arguments["model"]
    label_map = checkpoint["label_map"]
    model = build_model(model_name, num_classes=len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    config = load_json(Path(args.data_config).resolve())

    inverse_labels = [None] * len(label_map)
    for device_id, label in label_map.items():
        inverse_labels[int(label)] = device_id
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "dataset": dataset_name,
        "model": model_name,
        "equalized": equalized,
        "device": str(device),
        "labels": inverse_labels,
        "splits": {},
    }
    for split in args.splits:
        dataset = make_dataset(config, dataset_name, split=split, equalized=equalized)
        if dataset.label_map != label_map:
            raise ValueError("evaluation label map differs from the checkpoint")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate(model, loader, device, len(label_map))
        metrics["per_class"] = {
            inverse_labels[index]: {
                "precision": metrics["precision"][index],
                "recall": metrics["recall"][index],
                "f1": metrics["f1"][index],
            }
            for index in range(len(inverse_labels))
        }
        report["splits"][split] = metrics
        print(
            "%s: loss %.4f acc %.4f macro_f1 %.4f (%d examples)"
            % (
                split,
                metrics["loss"],
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["examples"],
            ),
            flush=True,
        )
    report["status"] = "complete"
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("Evaluation report: %s" % output_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
