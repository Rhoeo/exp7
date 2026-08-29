#!/usr/bin/env python
"""Probe whether sparse RF-GPT token edits create valid hard RFFI samples."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.token_datasets import WiFiBTokenDataset
from rffi_core.generators.rfgpt.evaluate_generation import load_classifier, sample_logits
from rffi_core.generators.rfgpt.models import build_rfgpt
from rffi_core.generators.vqvae.losses import mean_complex_power, normalize_complex_power
from rffi_core.generators.vqvae.models import build_reconstruction_model
from rffi_core.generators.vqvae.train_reconstruction import choose_device
from rffi_core.attacks.rf_constraints import bandlimit_complex


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt")
    parser.add_argument("--frozen-registry", default="runs/stage_g0/frozen/wifib_v1/frozen_models.json")
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument(
        "--token-cache",
        default="E:/data_cache/rffi_v1/tokens/wifib_vq_p1_k1024/tokens.npy",
    )
    parser.add_argument("--split", default="reward_validation")
    parser.add_argument("--samples-per-device", type=int, default=4)
    parser.add_argument("--edit-rates", type=float, nargs="+", default=(0.005, 0.01, 0.02))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--proposal", choices=("rfgpt", "uniform"), default="rfgpt")
    parser.add_argument("--sample-rate-hz", type=float, default=35e6)
    parser.add_argument("--occupied-bandwidth-hz", type=float, default=22e6)
    parser.add_argument("--bandlimit-perturbation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=34)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--min-evaluator-retention", type=float, default=0.95)
    parser.add_argument("--min-dual-valid-hard-rate", type=float, default=0.05)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def balanced_subset(dataset, samples_per_device, seed):
    by_label = {}
    for index, row in enumerate(dataset.rows):
        label = dataset.label_map[row["device_id"]]
        by_label.setdefault(label, []).append(index)
    rng = np.random.RandomState(seed)
    selected = []
    for label in range(len(dataset.label_map)):
        candidates = by_label[label]
        if len(candidates) < samples_per_device:
            raise ValueError("not enough samples for device label %d" % label)
        selected.extend(
            rng.choice(candidates, samples_per_device, replace=False).tolist()
        )
    return Subset(dataset, selected)


def edit_tokens(
    model, reference_tokens, labels, positions, temperature, top_p, proposal="rfgpt"
):
    edited = reference_tokens.clone()
    for position in positions:
        current = edited[:, int(position)]
        if proposal == "uniform":
            replacement = torch.randint(
                0,
                model.codebook_size - 1,
                current.shape,
                device=current.device,
            )
            replacement = replacement + (replacement >= current).long()
        else:
            start = max(0, int(position) - model.context_length)
            context = edited[:, start:int(position)]
            logits = model(context, labels, position_offset=start)[:, -1].clone()
            logits.scatter_(1, current.view(-1, 1), float("-inf"))
            replacement = sample_logits(logits, temperature, top_p)
        edited[:, int(position)] = replacement
    return edited


def classify(models, iq, batch_size, device):
    predictions = {name: [] for name in models}
    with torch.inference_mode():
        for start in range(0, iq.shape[0], batch_size):
            inputs = iq[start : start + batch_size].to(device)
            for name, model in models.items():
                predictions[name].append(model(inputs).argmax(dim=1).cpu())
    return {name: torch.cat(parts) for name, parts in predictions.items()}


def rf_delta_metrics(
    edited_iq, reference_iq, sample_rate_hz=35e6, occupied_bandwidth_hz=22e6
):
    delta = edited_iq - reference_iq
    error_power = mean_complex_power(delta)
    reference_power = mean_complex_power(reference_iq).clamp_min(1e-8)
    nmse = error_power / reference_power
    flat_edited = edited_iq.flatten(start_dim=1)
    flat_reference = reference_iq.flatten(start_dim=1)
    correlation = (flat_edited * flat_reference).sum(dim=1) / (
        flat_edited.square().sum(dim=1).clamp_min(1e-8).sqrt()
        * flat_reference.square().sum(dim=1).clamp_min(1e-8).sqrt()
    )
    delta_power = mean_complex_power(delta)
    delta_peak_power = (delta[:, 0].square() + delta[:, 1].square()).amax(dim=1)
    output_power = mean_complex_power(edited_iq)
    output_peak_power = (
        edited_iq[:, 0].square() + edited_iq[:, 1].square()
    ).amax(dim=1)
    spectrum = torch.fft.fft(
        torch.complex(delta[:, 0].float(), delta[:, 1].float()), dim=-1, norm="ortho"
    )
    spectrum_power = torch.fft.fftshift(
        spectrum.real.square() + spectrum.imag.square(), dim=-1
    )
    edge_bins = int(
        spectrum_power.shape[-1]
        * max(0.0, 1.0 - occupied_bandwidth_hz / sample_rate_hz)
        / 2.0
    )
    if edge_bins > 0:
        out_of_band = spectrum_power[:, :edge_bins].sum(dim=1) + spectrum_power[
            :, -edge_bins:
        ].sum(dim=1)
    else:
        out_of_band = spectrum_power.new_zeros(spectrum_power.shape[0])
    mean_nmse = float(nmse.mean())
    return {
        "mean_nmse": mean_nmse,
        "mean_delta_snr_db": -10.0 * math.log10(max(mean_nmse, 1e-12)),
        "mean_waveform_correlation": float(correlation.mean()),
        "mean_delta_peak_abs": float(delta.abs().amax(dim=(1, 2)).mean()),
        "mean_delta_papr": float(
            (delta_peak_power / delta_power.clamp_min(1e-12)).mean()
        ),
        "mean_output_papr": float(
            (output_peak_power / output_power.clamp_min(1e-12)).mean()
        ),
        "delta_out_of_band_energy_fraction": float(
            (out_of_band / spectrum_power.sum(dim=1).clamp_min(1e-12)).mean()
        ),
        "output_complex_power_mean": float(output_power.mean()),
        "finite_fraction": float(torch.isfinite(edited_iq).float().mean()),
    }


def main(argv=None):
    args = parse_args(argv)
    if args.split == "final_test":
        raise ValueError("final_test cannot be used for hardness probing")
    for rate in args.edit_rates:
        if not 0 < rate < 1:
            raise ValueError("edit rates must be in (0, 1)")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]
    policy = build_rfgpt(checkpoint["model_config"])
    policy.load_state_dict(checkpoint["model_state"])
    policy.to(device).eval()
    tokenizer_checkpoint = torch.load(
        str(Path(args.tokenizer).resolve()), map_location="cpu", weights_only=False
    )
    tokenizer = build_reconstruction_model(tokenizer_checkpoint["model_config"])
    tokenizer.load_state_dict(tokenizer_checkpoint["model_state"])
    tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    config = load_json(Path(args.data_config).resolve())
    index_path = Path(config["cache_root"]).resolve() / "wifib" / "window_index.csv"
    base = WiFiBTokenDataset(index_path, Path(args.token_cache).resolve(), args.split)
    if base.label_map != label_map:
        raise ValueError("probe label map differs from RF-GPT")
    dataset = balanced_subset(base, args.samples_per_device, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    reference_parts = []
    label_parts = []
    cache_indices = []
    for batch in loader:
        reference_parts.append(batch["tokens"].long())
        label_parts.append(batch["label"].long())
        cache_indices.extend(int(value) for value in batch["cache_index"])
    reference_tokens = torch.cat(reference_parts)
    labels = torch.cat(label_parts)
    with torch.inference_mode():
        reference_iq_parts = []
        for start in range(0, len(dataset), args.batch_size):
            tokens = reference_tokens[start : start + args.batch_size].to(device)
            reference_iq_parts.append(tokenizer.decode_code_indices(tokens).cpu())
        reference_iq = normalize_complex_power(torch.cat(reference_iq_parts))
    with Path(args.frozen_registry).resolve().open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    classifiers = {
        name: load_classifier(
            Path(registry["models"][name]["checkpoint"]), label_map, device
        )
        for name in ("victim_a", "evaluator_b")
    }
    reference_predictions = classify(
        classifiers, reference_iq, args.batch_size, device
    )
    reference_correct = {
        name: predictions == labels for name, predictions in reference_predictions.items()
    }
    joint_clean = reference_correct["victim_a"] & reference_correct["evaluator_b"]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.time()
    for rate_index, rate in enumerate(args.edit_rates):
        edit_count = max(1, int(round(reference_tokens.shape[1] * rate)))
        rng = np.random.RandomState(args.seed + rate_index * 1009)
        positions = np.sort(
            rng.choice(
                np.arange(1, reference_tokens.shape[1]),
                size=edit_count,
                replace=False,
            )
        )
        edited_parts = []
        with torch.inference_mode():
            for start in range(0, len(dataset), args.batch_size):
                tokens = reference_tokens[start : start + args.batch_size].to(device)
                batch_labels = labels[start : start + args.batch_size].to(device)
                edited_parts.append(
                    edit_tokens(
                        policy,
                        tokens,
                        batch_labels,
                        positions,
                        args.temperature,
                        args.top_p,
                        args.proposal,
                    ).cpu()
                )
        edited_tokens = torch.cat(edited_parts)
        actual_edit_fraction = float((edited_tokens != reference_tokens).float().mean())
        with torch.inference_mode():
            iq_parts = []
            for start in range(0, len(dataset), args.batch_size):
                tokens = edited_tokens[start : start + args.batch_size].to(device)
                iq_parts.append(tokenizer.decode_code_indices(tokens).cpu())
            edited_iq = normalize_complex_power(torch.cat(iq_parts))
        if args.bandlimit_perturbation:
            edited_iq = normalize_complex_power(
                reference_iq
                + bandlimit_complex(
                    edited_iq - reference_iq,
                    args.sample_rate_hz,
                    args.occupied_bandwidth_hz,
                )
            )
        edited_predictions = classify(classifiers, edited_iq, args.batch_size, device)
        victim_fooled = reference_correct["victim_a"] & (
            edited_predictions["victim_a"] != labels
        )
        evaluator_retained = reference_correct["evaluator_b"] & (
            edited_predictions["evaluator_b"] == labels
        )
        dual_valid_hard = joint_clean & victim_fooled & evaluator_retained
        key = "%.4f" % rate
        rate_dir = output_dir / ("edit_rate_" + key.replace(".", "p"))
        rate_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(rate_dir / "edited_tokens.npy"), edited_tokens.numpy().astype(np.uint16))
        np.save(str(rate_dir / "edited_iq_float32.npy"), edited_iq.numpy().astype(np.float32))
        results[key] = {
            "requested_edit_rate": rate,
            "edit_count": edit_count,
            "actual_edit_fraction": actual_edit_fraction,
            "shared_edit_positions": positions.tolist(),
            "rf_delta": rf_delta_metrics(
                edited_iq,
                reference_iq,
                sample_rate_hz=args.sample_rate_hz,
                occupied_bandwidth_hz=args.occupied_bandwidth_hz,
            ),
            "victim_a": {
                "reference_clean_accuracy": float(reference_correct["victim_a"].float().mean()),
                "edited_accuracy": float((edited_predictions["victim_a"] == labels).float().mean()),
                "fool_rate_on_reference_correct": float(victim_fooled.sum())
                / max(1, int(reference_correct["victim_a"].sum())),
            },
            "evaluator_b": {
                "reference_clean_accuracy": float(reference_correct["evaluator_b"].float().mean()),
                "edited_accuracy": float((edited_predictions["evaluator_b"] == labels).float().mean()),
                "identity_retention_on_reference_correct": float(evaluator_retained.sum())
                / max(1, int(reference_correct["evaluator_b"].sum())),
            },
            "joint_clean_examples": int(joint_clean.sum()),
            "dual_valid_hard_examples": int(dual_valid_hard.sum()),
            "dual_valid_hard_rate": float(dual_valid_hard.sum())
            / max(1, int(joint_clean.sum())),
        }
        print(
            "rate %.3f edits %d | victim fool %.3f evaluator retention %.3f dual %.3f"
            % (
                rate,
                edit_count,
                results[key]["victim_a"]["fool_rate_on_reference_correct"],
                results[key]["evaluator_b"]["identity_retention_on_reference_correct"],
                results[key]["dual_valid_hard_rate"],
            ),
            flush=True,
        )
    passing_rates = [
        key
        for key, result in results.items()
        if result["evaluator_b"]["identity_retention_on_reference_correct"]
        >= args.min_evaluator_retention
        and result["dual_valid_hard_rate"] >= args.min_dual_valid_hard_rate
    ]
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "training_used_classifier_feedback": False,
        "probe_used_classifier_feedback": False,
        "split": args.split,
        "examples": len(dataset),
        "samples_per_device": args.samples_per_device,
        "cache_indices": cache_indices,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "proposal": args.proposal,
        "sample_rate_hz": args.sample_rate_hz,
        "occupied_bandwidth_hz": args.occupied_bandwidth_hz,
        "bandlimit_perturbation": args.bandlimit_perturbation,
        "results": results,
        "hardness_gate": {
            "passed": bool(passing_rates),
            "passing_edit_rates": passing_rates,
            "thresholds": {
                "min_evaluator_identity_retention": args.min_evaluator_retention,
                "min_dual_valid_hard_rate": args.min_dual_valid_hard_rate,
            },
        },
        "seconds": time.time() - started,
    }
    with (output_dir / "hardness_probe.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "hardness gate: %s" % ("PASS" if passing_rates else "FAIL"), flush=True
    )
    print("report: %s" % (output_dir / "hardness_probe.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
