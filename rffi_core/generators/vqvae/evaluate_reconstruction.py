#!/usr/bin/env python
"""Evaluate RF reconstruction and frozen-classifier identity preservation."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.generators.vqvae.losses import (
    complex_fft_magnitude,
    mean_complex_power,
    normalize_complex_power,
)
from rffi_core.generators.vqvae.models import build_reconstruction_model
from rffi_core.generators.vqvae.train_reconstruction import (
    choose_device,
    deterministic_subset,
    make_dataset,
)
from rffi_core.victim.models import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frozen-registry", default="runs/stage_g0/frozen/wifib_v1/frozen_models.json")
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument("--split", default="reward_validation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-evaluator-drop-pp", type=float, default=5.0)
    parser.add_argument("--max-victim-drop-pp", type=float, default=10.0)
    parser.add_argument("--min-reconstruction-snr-db", type=float, default=10.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def load_frozen_classifier(path, expected_label_map, device):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint["label_map"] != expected_label_map:
        raise ValueError("classifier label map differs from generator label map")
    architecture = checkpoint["arguments"]["model"]
    model = build_model(architecture, num_classes=len(expected_label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, architecture, int(checkpoint["epoch"])


def evaluate(generator, classifiers, loader, device, codebook_size):
    totals = {
        "examples": 0,
        "nmse_sum": 0.0,
        "correlation_sum": 0.0,
        "spectral_log_l1_sum": 0.0,
        "output_power_sum": 0.0,
        "output_power_square_sum": 0.0,
        "finite_values": 0,
        "values": 0,
        "peak_absolute": 0.0,
    }
    classifier_totals = {
        name: {
            "clean_correct": 0,
            "reconstructed_correct": 0,
            "retained_from_clean_correct": 0,
            "prediction_agreement": 0,
            "matched_noise_correct": 0,
            "matched_noise_retained_from_clean_correct": 0,
            "matched_noise_prediction_agreement": 0,
        }
        for name in classifiers
    }
    code_histogram = torch.zeros(codebook_size, dtype=torch.long)
    started = time.time()
    generator.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = batch["iq"].to(device=device, dtype=torch.float32, non_blocking=True)
            targets = batch["label"].to(device=device, dtype=torch.long, non_blocking=True)
            outputs = generator(inputs)
            reconstruction = outputs["reconstruction"]
            normalized_reconstruction = normalize_complex_power(reconstruction)
            batch_size = int(inputs.shape[0])

            error_power = mean_complex_power(reconstruction - inputs)
            input_power = mean_complex_power(inputs).clamp_min(1e-8)
            nmse = error_power / input_power
            flat_reconstruction = reconstruction.flatten(start_dim=1)
            flat_inputs = inputs.flatten(start_dim=1)
            correlation = (flat_reconstruction * flat_inputs).sum(dim=1) / (
                flat_reconstruction.square().sum(dim=1).clamp_min(1e-8).sqrt()
                * flat_inputs.square().sum(dim=1).clamp_min(1e-8).sqrt()
            )
            spectral_l1 = (
                torch.log1p(complex_fft_magnitude(reconstruction))
                - torch.log1p(complex_fft_magnitude(inputs))
            ).abs().mean(dim=1)
            output_power = mean_complex_power(reconstruction)
            noise = torch.randn_like(inputs)
            noise = normalize_complex_power(noise)
            matched_noise = normalize_complex_power(
                inputs + noise * error_power.sqrt().view(-1, 1, 1)
            )
            totals["examples"] += batch_size
            totals["nmse_sum"] += float(nmse.sum().cpu())
            totals["correlation_sum"] += float(correlation.sum().cpu())
            totals["spectral_log_l1_sum"] += float(spectral_l1.sum().cpu())
            totals["output_power_sum"] += float(output_power.sum().cpu())
            totals["output_power_square_sum"] += float(output_power.square().sum().cpu())
            totals["finite_values"] += int(torch.isfinite(reconstruction).sum().cpu())
            totals["values"] += reconstruction.numel()
            totals["peak_absolute"] = max(
                totals["peak_absolute"], float(reconstruction.abs().amax().cpu())
            )
            if outputs["code_indices"] is not None:
                code_histogram += torch.bincount(
                    outputs["code_indices"].flatten().cpu(), minlength=codebook_size
                )

            for name, model in classifiers.items():
                clean_predictions = model(inputs).argmax(dim=1)
                reconstructed_predictions = model(normalized_reconstruction).argmax(dim=1)
                matched_noise_predictions = model(matched_noise).argmax(dim=1)
                clean_correct = clean_predictions == targets
                reconstructed_correct = reconstructed_predictions == targets
                matched_noise_correct = matched_noise_predictions == targets
                values = classifier_totals[name]
                values["clean_correct"] += int(clean_correct.sum().cpu())
                values["reconstructed_correct"] += int(reconstructed_correct.sum().cpu())
                values["retained_from_clean_correct"] += int(
                    (clean_correct & reconstructed_correct).sum().cpu()
                )
                values["prediction_agreement"] += int(
                    (clean_predictions == reconstructed_predictions).sum().cpu()
                )
                values["matched_noise_correct"] += int(matched_noise_correct.sum().cpu())
                values["matched_noise_retained_from_clean_correct"] += int(
                    (clean_correct & matched_noise_correct).sum().cpu()
                )
                values["matched_noise_prediction_agreement"] += int(
                    (clean_predictions == matched_noise_predictions).sum().cpu()
                )

    examples = max(1, totals["examples"])
    mean_nmse = totals["nmse_sum"] / examples
    mean_power = totals["output_power_sum"] / examples
    power_variance = max(
        0.0, totals["output_power_square_sum"] / examples - mean_power * mean_power
    )
    reconstruction_metrics = {
        "examples": totals["examples"],
        "mean_nmse": mean_nmse,
        "reconstruction_snr_db": -10.0 * math.log10(max(mean_nmse, 1e-12)),
        "mean_waveform_correlation": totals["correlation_sum"] / examples,
        "mean_spectral_log_l1": totals["spectral_log_l1_sum"] / examples,
        "output_complex_power_mean": mean_power,
        "output_complex_power_std": math.sqrt(power_variance),
        "finite_fraction": totals["finite_values"] / max(1, totals["values"]),
        "peak_absolute": totals["peak_absolute"],
        "seconds": time.time() - started,
    }
    classification = {}
    for name, values in classifier_totals.items():
        clean_correct = values["clean_correct"]
        clean_accuracy = clean_correct / examples
        reconstructed_accuracy = values["reconstructed_correct"] / examples
        classification[name] = {
            "clean_accuracy": clean_accuracy,
            "reconstructed_accuracy": reconstructed_accuracy,
            "accuracy_drop_percentage_points": 100.0
            * (clean_accuracy - reconstructed_accuracy),
            "clean_correct_examples": clean_correct,
            "clean_correct_identity_retention": values["retained_from_clean_correct"]
            / max(1, clean_correct),
            "prediction_agreement": values["prediction_agreement"] / examples,
            "matched_noise_accuracy": values["matched_noise_correct"] / examples,
            "matched_noise_clean_correct_identity_retention": values[
                "matched_noise_retained_from_clean_correct"
            ]
            / max(1, clean_correct),
            "matched_noise_prediction_agreement": values[
                "matched_noise_prediction_agreement"
            ]
            / examples,
        }
    codebook = None
    if int(code_histogram.sum()) > 0:
        probabilities = code_histogram.float() / code_histogram.sum()
        nonzero = probabilities > 0
        codebook = {
            "codebook_size": codebook_size,
            "active_codes": int(nonzero.sum()),
            "active_code_ratio": float(nonzero.float().mean()),
            "perplexity": float(
                torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
            ),
            "maximum_code_fraction": float(probabilities.max()),
        }
    return reconstruction_metrics, classification, codebook


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    generator_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(generator_path), map_location="cpu", weights_only=False)
    generator = build_reconstruction_model(checkpoint["model_config"])
    generator.load_state_dict(checkpoint["model_state"])
    generator.to(device)
    label_map = checkpoint["label_map"]
    training_args = checkpoint["arguments"]
    config = load_json(Path(args.data_config).resolve())
    dataset = make_dataset(
        config,
        training_args["dataset"],
        args.split,
        int(training_args.get("equalized", 0)),
    )
    if dataset.label_map != label_map:
        raise ValueError("evaluation label map differs from generator label map")
    dataset = deterministic_subset(dataset, args.max_samples, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    registry_path = Path(args.frozen_registry).resolve()
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    classifiers = {}
    classifier_metadata = {}
    for name in ("victim_a", "evaluator_b"):
        model_entry = registry["models"][name]
        model, architecture, epoch = load_frozen_classifier(
            Path(model_entry["checkpoint"]), label_map, device
        )
        classifiers[name] = model
        classifier_metadata[name] = {
            "checkpoint": model_entry["checkpoint"],
            "architecture": architecture,
            "checkpoint_epoch": epoch,
            "frozen": True,
        }

    reconstruction, classification, codebook = evaluate(
        generator,
        classifiers,
        loader,
        device,
        int(checkpoint["model_config"]["codebook_size"]),
    )
    gate_checks = {
        "finite_output": reconstruction["finite_fraction"] == 1.0,
        "reconstruction_snr": reconstruction["reconstruction_snr_db"]
        >= args.min_reconstruction_snr_db,
        "victim_accuracy_drop": classification["victim_a"]
        ["accuracy_drop_percentage_points"]
        <= args.max_victim_drop_pp,
        "evaluator_accuracy_drop": classification["evaluator_b"]
        ["accuracy_drop_percentage_points"]
        <= args.max_evaluator_drop_pp,
    }
    report = {
        "status": "complete",
        "checkpoint": str(generator_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model_config": checkpoint["model_config"],
        "training_used_classifier_feedback": False,
        "dataset": training_args["dataset"],
        "split": args.split,
        "device": str(device),
        "reconstruction": reconstruction,
        "classification": classification,
        "classifier_metadata": classifier_metadata,
        "codebook": codebook,
        "identity_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "thresholds": {
                "max_evaluator_drop_percentage_points": args.max_evaluator_drop_pp,
                "max_victim_drop_percentage_points": args.max_victim_drop_pp,
                "min_reconstruction_snr_db": args.min_reconstruction_snr_db,
            },
        },
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "reconstruction: SNR %.2f dB, corr %.4f, spectral L1 %.5f"
        % (
            reconstruction["reconstruction_snr_db"],
            reconstruction["mean_waveform_correlation"],
            reconstruction["mean_spectral_log_l1"],
        ),
        flush=True,
    )
    for name, metrics in classification.items():
        print(
            "%s: clean %.4f reconstructed %.4f drop %.2f pp retention %.4f | matched-noise %.4f"
            % (
                name,
                metrics["clean_accuracy"],
                metrics["reconstructed_accuracy"],
                metrics["accuracy_drop_percentage_points"],
                metrics["clean_correct_identity_retention"],
                metrics["matched_noise_accuracy"],
            ),
            flush=True,
        )
    if codebook is not None:
        print(
            "codebook: %d/%d active, perplexity %.2f, max fraction %.3f"
            % (
                codebook["active_codes"],
                codebook["codebook_size"],
                codebook["perplexity"],
                codebook["maximum_code_fraction"],
            ),
            flush=True,
        )
    print("identity gate: %s" % ("PASS" if all(gate_checks.values()) else "FAIL"), flush=True)
    print("report: %s" % output_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
