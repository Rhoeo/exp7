#!/usr/bin/env python
"""Train a device-conditioned RF-GPT on frozen token transitions."""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.token_datasets import WiFiBTokenDataset
from rffi_core.generators.rfgpt.models import build_rfgpt
from rffi_core.generators.vqvae.train_reconstruction import choose_device


class TransitionCropDataset(Dataset):
    """One deterministic random next-token crop per waveform and epoch."""

    def __init__(self, dataset, context_length, seed):
        self.dataset = dataset
        self.context_length = int(context_length)
        self.seed = int(seed)
        self.epoch = 0
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        tokens = item["tokens"]
        maximum_start = len(tokens) - self.context_length - 1
        if maximum_start < 0:
            raise ValueError("token sequence is too short for the requested context")
        mixed_seed = (self.seed + self.epoch * 1000003 + int(index) * 9176) % (2**32)
        start = int(np.random.RandomState(mixed_seed).randint(maximum_start + 1))
        window = tokens[start : start + self.context_length + 1]
        return {
            "input_tokens": np.asarray(window[:-1], dtype=np.int64),
            "target_tokens": np.asarray(window[1:], dtype=np.int64),
            "position_offset": start,
            "label": item["label"],
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument(
        "--token-cache",
        default="E:/data_cache/rffi_v1/tokens/wifib_vq_p1_k1024/tokens.npy",
    )
    parser.add_argument("--codebook-size", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true")
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
    indices = np.random.RandomState(seed).choice(len(dataset), maximum, replace=False)
    return Subset(dataset, sorted(indices.tolist()))


def run_epoch(model, loader, device, optimizer=None, scaler=None, accumulation=1):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    started = time.time()
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        inputs = batch["input_tokens"].to(device=device, dtype=torch.long, non_blocking=True)
        targets = batch["target_tokens"].to(device=device, dtype=torch.long, non_blocking=True)
        labels = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
        offsets = batch["position_offset"].to(device=device, dtype=torch.long, non_blocking=True)
        use_amp = scaler is not None and scaler.is_enabled()
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(inputs, labels, offsets)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if training:
                scaled_loss = loss / accumulation
                if scaler is None:
                    scaled_loss.backward()
                else:
                    scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        count = int(targets.numel())
        total_loss += float(loss.detach().cpu()) * count
        total_correct += int((logits.argmax(dim=-1) == targets).sum().detach().cpu())
        total_tokens += count
    mean_loss = total_loss / max(1, total_tokens)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "token_accuracy": total_correct / max(1, total_tokens),
        "tokens": total_tokens,
        "seconds": time.time() - started,
    }


def initial_token_counts(dataset, num_devices, codebook_size):
    counts = np.zeros((num_devices, codebook_size), dtype=np.int64)
    for index in range(len(dataset)):
        item = dataset[index]
        counts[int(item["label"]), int(item["tokens"][0])] += 1
    return counts


def save_checkpoint(path, model, optimizer, epoch, label_map, args, model_config, metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "label_map": label_map,
            "arguments": vars(args),
            "model_config": model_config,
            "metrics": metrics,
            "training_used_classifier_feedback": False,
        },
        str(path),
    )


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    config = load_json(Path(args.data_config).resolve())
    cache_root = Path(config["cache_root"]).resolve()
    index_path = cache_root / "wifib" / "window_index.csv"
    token_path = Path(args.token_cache).resolve()
    train_base = WiFiBTokenDataset(index_path, token_path, "generator_train")
    val_base = WiFiBTokenDataset(index_path, token_path, "reward_validation")
    if train_base.label_map != val_base.label_map:
        raise ValueError("training and validation label maps differ")
    label_map = train_base.label_map
    train_base = deterministic_subset(train_base, args.max_train_samples, args.seed)
    val_base = deterministic_subset(val_base, args.max_val_samples, args.seed + 1)
    train_dataset = TransitionCropDataset(train_base, args.context_length, args.seed)
    val_dataset = TransitionCropDataset(val_base, args.context_length, args.seed + 2)
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
    sequence_length = int(np.load(str(token_path), mmap_mode="r").shape[1])
    model_config = {
        "codebook_size": args.codebook_size,
        "num_devices": len(label_map),
        "max_sequence_length": sequence_length,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
    }
    model = build_rfgpt(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    scaler = torch.amp.GradScaler(
        device.type, enabled=args.amp and device.type == "cuda"
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        str(output_dir / "initial_token_counts.npy"),
        initial_token_counts(train_base, len(label_map), args.codebook_size),
    )
    run_config = dict(vars(args))
    run_config.update(
        {
            "resolved_device": str(device),
            "model_config": model_config,
            "train_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "classifier_feedback": False,
            "token_cache": str(token_path),
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    print(
        "Training RF-GPT: %d train / %d val, context=%d, device=%s, params=%d"
        % (
            len(train_dataset),
            len(val_dataset),
            args.context_length,
            device,
            run_config["parameter_count"],
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(0)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            scaler=scaler,
            accumulation=args.grad_accumulation,
        )
        val_metrics = run_epoch(model, val_loader, device)
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
            model_config,
            epoch_metrics,
        )
        if val_metrics["loss"] < best_loss - args.min_delta:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                label_map,
                args,
                model_config,
                epoch_metrics,
            )
        else:
            stale += 1
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(
            "epoch %d/%d train loss %.4f ppl %.2f acc %.3f | val loss %.4f ppl %.2f acc %.3f"
            % (
                epoch,
                args.epochs,
                train_metrics["loss"],
                train_metrics["perplexity"],
                train_metrics["token_accuracy"],
                val_metrics["loss"],
                val_metrics["perplexity"],
                val_metrics["token_accuracy"],
            ),
            flush=True,
        )
        if args.patience > 0 and stale >= args.patience:
            print("Early stopping after %d stale epochs" % stale, flush=True)
            break
    summary = {
        "status": "complete",
        "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(min(20.0, best_loss)),
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("Best validation loss: %.5f" % best_loss, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

