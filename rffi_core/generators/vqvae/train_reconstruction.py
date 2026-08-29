#!/usr/bin/env python
"""Train Stage G1 AE/VQ-VAE using waveform losses only."""

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.datasets import ManyTxMemmapDataset, WiFiBCachedWindowDataset
from rffi_core.generators.vqvae.losses import reconstruction_loss
from rffi_core.generators.vqvae.models import build_reconstruction_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument("--dataset", choices=("wifib", "manytx"), default="wifib")
    parser.add_argument("--equalized", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model", choices=("ae", "vqvae"), required=True)
    parser.add_argument(
        "--architecture", choices=("conv", "polyphase"), default="polyphase"
    )
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--base-width", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--commitment-beta", type=float, default=0.25)
    parser.add_argument(
        "--decoder-type",
        choices=("subpixel", "resize_conv", "transpose"),
        default="subpixel",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--waveform-weight", type=float, default=1.0)
    parser.add_argument("--correlation-weight", type=float, default=0.1)
    parser.add_argument("--spectral-weight", type=float, default=0.1)
    parser.add_argument("--power-weight", type=float, default=0.05)
    parser.add_argument("--vq-weight", type=float, default=1.0)
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


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def deterministic_subset(dataset, maximum, seed):
    if maximum is None or maximum >= len(dataset):
        return dataset
    indices = np.random.RandomState(seed).choice(len(dataset), size=maximum, replace=False)
    return Subset(dataset, sorted(indices.tolist()))


def make_dataset(config, dataset_name, split, equalized):
    cache_root = Path(config["cache_root"]).resolve()
    if dataset_name == "wifib":
        window = config["wifib"]["primary_window"]
        cache_dir = cache_root / "wifib"
        return WiFiBCachedWindowDataset(
            window_index_path=cache_dir / "window_index.csv",
            iq_cache_path=cache_dir / "iq_window_2048_float32.npy",
            power_cache_path=cache_dir / "power_window_2048_float32.npy",
            split=split,
            window_offset=0,
            window_length=window["complex_length"],
            normalize=True,
        )
    cache_dir = cache_root / "manytx"
    return ManyTxMemmapDataset(
        group_index_path=cache_dir / "group_index.csv",
        iq_cache_path=cache_dir / "iq_float32.npy",
        power_cache_path=cache_dir / "power_mean_square_float32.npy",
        split=split,
        equalized=equalized,
        normalize=True,
    )


def run_epoch(model, loader, device, loss_weights, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    examples = 0
    code_histogram = None
    started = time.time()
    for batch in loader:
        inputs = batch["iq"].to(device=device, dtype=torch.float32, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(inputs)
            losses = reconstruction_loss(
                outputs["reconstruction"], inputs, outputs["vq_loss"], loss_weights
            )
            if training:
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(inputs.shape[0])
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
        examples += batch_size
        if outputs["code_indices"] is not None:
            size = model.quantizer.codebook_size
            batch_histogram = torch.bincount(
                outputs["code_indices"].detach().flatten(), minlength=size
            ).cpu()
            code_histogram = (
                batch_histogram
                if code_histogram is None
                else code_histogram + batch_histogram
            )
    metrics = {name: value / max(1, examples) for name, value in totals.items()}
    metrics.update({"examples": examples, "seconds": time.time() - started})
    if code_histogram is not None:
        probabilities = code_histogram.float() / code_histogram.sum().clamp_min(1)
        nonzero = probabilities > 0
        metrics["code_perplexity"] = float(
            torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
        )
        metrics["active_code_ratio"] = float(nonzero.float().mean())
    return metrics


def save_checkpoint(path, model, optimizer, epoch, label_map, args, model_config, loss_config, metrics):
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "label_map": label_map,
            "arguments": vars(args),
            "model_config": model_config,
            "loss_config": loss_config,
            "metrics": metrics,
        },
        str(path),
    )


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    config = load_json(Path(args.data_config).resolve())
    train_dataset = make_dataset(config, args.dataset, "generator_train", args.equalized)
    val_dataset = make_dataset(config, args.dataset, "reward_validation", args.equalized)
    if train_dataset.label_map != val_dataset.label_map:
        raise ValueError("training and validation label maps differ")
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
    model_config = {
        "mode": args.model,
        "architecture": args.architecture,
        "patch_size": args.patch_size,
        "base_width": args.base_width,
        "latent_dim": args.latent_dim,
        "codebook_size": args.codebook_size,
        "commitment_beta": args.commitment_beta,
        "decoder_type": args.decoder_type,
    }
    loss_config = {
        "waveform": args.waveform_weight,
        "correlation": args.correlation_weight,
        "spectral": args.spectral_weight,
        "power": args.power_weight,
        "vq": args.vq_weight,
    }
    model = build_reconstruction_model(model_config).to(device)
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
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_config": model_config,
            "loss_config": loss_config,
            "classifier_feedback": False,
        }
    )
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    history = []
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    print(
        "Training %s on %s: %d train / %d val, device=%s, params=%d"
        % (
            args.model,
            args.dataset,
            len(train_dataset),
            len(val_dataset),
            device,
            run_config["parameter_count"],
        ),
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, loss_config, optimizer)
        val_metrics = run_epoch(model, val_loader, device, loss_config)
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
            loss_config,
            epoch_metrics,
        )
        if val_metrics["total"] < best_loss - args.min_delta:
            best_loss = val_metrics["total"]
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                label_map,
                args,
                model_config,
                loss_config,
                epoch_metrics,
            )
        else:
            stale_epochs += 1
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        code_text = ""
        if "code_perplexity" in val_metrics:
            code_text = " | ppl %.1f active %.3f" % (
                val_metrics["code_perplexity"],
                val_metrics["active_code_ratio"],
            )
        print(
            "epoch %d/%d train %.5f | val %.5f nmse %.5f%s"
            % (
                epoch,
                args.epochs,
                train_metrics["total"],
                val_metrics["total"],
                val_metrics["waveform_nmse"],
                code_text,
            ),
            flush=True,
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print("Early stopping after %d stale epochs" % stale_epochs, flush=True)
            break
    summary = {
        "status": "complete",
        "best_validation_loss": best_loss,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("Best validation loss: %.6f" % best_loss, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
