#!/usr/bin/env python
"""Train Stage G0 Victim A or independent Evaluator B."""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.datasets import (
    ManyTxMemmapDataset,
    WiFiBCachedWindowDataset,
    WiFiBWindowDataset,
)
from rffi_core.victim.models import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument("--dataset", choices=("wifib", "manytx"), default="wifib")
    parser.add_argument("--model", choices=("victim_a", "evaluator_b"), required=True)
    parser.add_argument("--equalized", type=int, choices=(0, 1), default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def deterministic_subset(dataset, maximum, seed):
    if maximum is None or maximum >= len(dataset):
        return dataset
    indices = np.random.RandomState(seed).choice(len(dataset), size=maximum, replace=False)
    return Subset(dataset, sorted(indices.tolist()))


def make_dataset(config, dataset_name, split, equalized):
    cache_root = Path(config["cache_root"]).resolve()
    manifest_dir = cache_root / "manifests"
    if dataset_name == "wifib":
        window = config["wifib"]["primary_window"]
        wifib_dir = cache_root / "wifib"
        cache_paths = (
            wifib_dir / "window_index.csv",
            wifib_dir / "iq_window_2048_float32.npy",
            wifib_dir / "power_window_2048_float32.npy",
        )
        if all(path.is_file() for path in cache_paths):
            common = {
                "window_index_path": cache_paths[0],
                "iq_cache_path": cache_paths[1],
                "power_cache_path": cache_paths[2],
                "window_offset": 0,
                "window_length": window["complex_length"],
                "normalize": True,
            }
            dataset = WiFiBCachedWindowDataset(split=split, **common)
        else:
            dataset = WiFiBWindowDataset(
                manifest_dir / "wifib_frames.csv",
                split=split,
                window_start=window["start"],
                window_length=window["complex_length"],
                normalize=True,
            )
    else:
        manytx_dir = cache_root / "manytx"
        common = {
            "group_index_path": manytx_dir / "group_index.csv",
            "iq_cache_path": manytx_dir / "iq_float32.npy",
            "power_cache_path": manytx_dir / "power_mean_square_float32.npy",
            "equalized": equalized,
            "normalize": True,
        }
        dataset = ManyTxMemmapDataset(split=split, **common)
    return dataset


def make_datasets(config, dataset_name, equalized):
    train_dataset = make_dataset(
        config, dataset_name, split="generator_train", equalized=equalized
    )
    val_dataset = make_dataset(
        config, dataset_name, split="reward_validation", equalized=equalized
    )
    if train_dataset.label_map != val_dataset.label_map:
        raise ValueError("train and validation label maps differ")
    return train_dataset, val_dataset


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    started = time.time()
    for batch in loader:
        inputs = batch["iq"].to(device=device, dtype=torch.float32, non_blocking=True)
        targets = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(targets.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().detach().cpu())
        total_examples += batch_size
    return {
        "loss": total_loss / max(1, total_examples),
        "accuracy": total_correct / max(1, total_examples),
        "examples": total_examples,
        "seconds": time.time() - started,
    }


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer, epoch, label_map, arguments, metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "label_map": label_map,
            "arguments": vars(arguments),
            "metrics": metrics,
        },
        str(path),
    )


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    config = load_json(Path(args.data_config).resolve())
    train_dataset, val_dataset = make_datasets(config, args.dataset, args.equalized)
    label_map = train_dataset.label_map
    train_dataset = deterministic_subset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = deterministic_subset(val_dataset, args.max_val_samples, args.seed + 1)
    device = choose_device(args.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_model(args.model, num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(vars(args))
    run_config.update(
        {
            "resolved_device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "class_count": len(label_map),
            "train_dataset_class": type(train_dataset.dataset).__name__
            if isinstance(train_dataset, Subset)
            else type(train_dataset).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    print(
        "Training %s on %s: %d train / %d val, %d classes, device=%s"
        % (
            args.model,
            args.dataset,
            len(train_dataset),
            len(val_dataset),
            len(label_map),
            device,
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        epoch_metrics = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(epoch_metrics)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            label_map,
            args,
            epoch_metrics,
        )
        if val_metrics["accuracy"] > best_accuracy + args.min_delta:
            best_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                label_map,
                args,
                epoch_metrics,
            )
        else:
            epochs_without_improvement += 1
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(
            "epoch %d/%d train loss %.4f acc %.4f | val loss %.4f acc %.4f"
            % (
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
            ),
            flush=True,
        )
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            early_stopped = True
            print(
                "Early stopping after %d epochs without sufficient improvement"
                % epochs_without_improvement,
                flush=True,
            )
            break
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_validation_accuracy": best_accuracy,
                "best_epoch": best_epoch,
                "completed_epochs": len(history),
                "early_stopped": early_stopped,
                "status": "complete",
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    print("Best validation accuracy: %.4f" % best_accuracy, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
