#!/usr/bin/env python
"""Victim-visible PGD baseline under complex-RMS perturbation budgets."""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.datasets import WiFiBCachedWindowDataset
from rffi_core.attacks.rf_constraints import bandlimit_complex
from rffi_core.generators.vqvae.losses import mean_complex_power, normalize_complex_power
from rffi_core.generators.rfgpt.evaluate_generation import load_classifier
from rffi_core.generators.rfgpt.probe_token_edits import balanced_subset
from rffi_core.victim.models import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument("--frozen-registry", default="runs/stage_g0/frozen/wifib_v1/frozen_models.json")
    parser.add_argument("--split", default="reward_validation")
    parser.add_argument("--samples-per-device", type=int, default=4)
    parser.add_argument("--snr-budgets-db", type=float, nargs="+", default=(26.0, 22.0))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step-fraction", type=float, default=0.2)
    parser.add_argument("--sample-rate-hz", type=float, default=35e6)
    parser.add_argument("--occupied-bandwidth-hz", type=float, default=22e6)
    parser.add_argument("--bandlimit-perturbation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def project_complex_rms(delta, budget):
    rms = mean_complex_power(delta).clamp_min(1e-12).sqrt()
    scale = (float(budget) / rms).clamp_max(1.0).view(-1, 1, 1)
    return delta * scale


def pgd_untargeted(
    victim,
    inputs,
    labels,
    budget,
    steps,
    step_fraction,
    bandlimit=False,
    sample_rate_hz=35e6,
    occupied_bandwidth_hz=22e6,
):
    """Maximize true-label CE, projecting in complex RMS after every step."""
    adv = inputs.detach().clone()
    step_rms = float(budget) * float(step_fraction)
    for _ in range(int(steps)):
        adv.requires_grad_(True)
        logits = victim(adv)
        loss = F.cross_entropy(logits, labels)
        gradient = torch.autograd.grad(loss, adv, only_inputs=True)[0]
        gradient_rms = mean_complex_power(gradient).clamp_min(1e-12).sqrt()
        direction = gradient / gradient_rms.view(-1, 1, 1)
        candidate = adv.detach() + step_rms * direction
        delta = candidate - inputs
        if bandlimit:
            delta = bandlimit_complex(delta, sample_rate_hz, occupied_bandwidth_hz)
        delta = project_complex_rms(delta, budget)
        # Keep the classifier input on the same unit-power convention as clean IQ.
        adv = normalize_complex_power(inputs + delta).detach()
    return adv


def load_source_dataset(config, split):
    cache_dir = Path(config["cache_root"]).resolve() / "wifib"
    window = config["wifib"]["primary_window"]
    return WiFiBCachedWindowDataset(
        window_index_path=cache_dir / "window_index.csv",
        iq_cache_path=cache_dir / "iq_window_2048_float32.npy",
        power_cache_path=cache_dir / "power_window_2048_float32.npy",
        split=split,
        window_offset=0,
        window_length=window["complex_length"],
        normalize=True,
    )


def rf_metrics(adv, clean, sample_rate_hz=35e6, occupied_bandwidth_hz=22e6):
    delta = adv - clean
    nmse = mean_complex_power(delta) / mean_complex_power(clean).clamp_min(1e-8)
    flat_adv = adv.flatten(start_dim=1)
    flat_clean = clean.flatten(start_dim=1)
    correlation = (flat_adv * flat_clean).sum(dim=1) / (
        flat_adv.square().sum(dim=1).clamp_min(1e-8).sqrt()
        * flat_clean.square().sum(dim=1).clamp_min(1e-8).sqrt()
    )
    delta_power = mean_complex_power(delta)
    delta_peak = delta[:, 0].square() + delta[:, 1].square()
    output_power = mean_complex_power(adv)
    output_peak_power = (adv[:, 0].square() + adv[:, 1].square()).amax(dim=1)
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
    out_of_band = spectrum_power[:, :edge_bins].sum(dim=1) + spectrum_power[
        :, -edge_bins:
    ].sum(dim=1)
    return {
        "mean_nmse": float(nmse.mean()),
        "mean_snr_db": -10.0 * math.log10(max(float(nmse.mean()), 1e-12)),
        "mean_waveform_correlation": float(correlation.mean()),
        "mean_delta_peak_abs": float(delta.abs().amax(dim=(1, 2)).mean()),
        "mean_delta_papr": float(
            (delta_peak.amax(dim=1) / delta_power.clamp_min(1e-12)).mean()
        ),
        "mean_output_papr": float(
            (output_peak_power / output_power.clamp_min(1e-12)).mean()
        ),
        "delta_out_of_band_energy_fraction": float(
            (out_of_band / spectrum_power.sum(dim=1).clamp_min(1e-12)).mean()
        ),
        "output_power_mean": float(output_power.mean()),
        "output_power_std": float(output_power.std()),
        "finite_fraction": float(torch.isfinite(adv).float().mean()),
    }


def classify(model, inputs, batch_size, device):
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            predictions.append(model(inputs[start : start + batch_size].to(device)).argmax(dim=1).cpu())
    return torch.cat(predictions)


def main(argv=None):
    args = parse_args(argv)
    if args.split == "final_test":
        raise ValueError("final_test is not allowed for baseline selection")
    if args.steps <= 0 or not 0 < args.step_fraction <= 1:
        raise ValueError("steps must be positive and step_fraction must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    config = load_json(Path(args.data_config).resolve())
    dataset = balanced_subset(
        load_source_dataset(config, args.split), args.samples_per_device, args.seed
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    iq_parts = []
    labels_parts = []
    for batch in loader:
        iq_parts.append(batch["iq"].float())
        labels_parts.append(batch["label"].long())
    clean = torch.cat(iq_parts)
    labels = torch.cat(labels_parts)
    with Path(args.frozen_registry).resolve().open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    victim_checkpoint = torch.load(
        str(Path(registry["models"]["victim_a"]["checkpoint"])),
        map_location="cpu",
        weights_only=False,
    )
    label_map = victim_checkpoint["label_map"]
    victim = build_model("victim_a", len(label_map))
    victim.load_state_dict(victim_checkpoint["model_state"])
    victim.to(device).eval()
    for parameter in victim.parameters():
        parameter.requires_grad_(False)
    evaluator = load_classifier(
        Path(registry["models"]["evaluator_b"]["checkpoint"]), label_map, device
    )
    clean_victim = classify(victim, clean, args.batch_size, device)
    clean_evaluator = classify(evaluator, clean, args.batch_size, device)
    victim_clean_correct = clean_victim == labels
    evaluator_clean_correct = clean_evaluator == labels
    joint_clean = victim_clean_correct & evaluator_clean_correct
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.time()
    for budget_db in args.snr_budgets_db:
        budget = math.sqrt(10.0 ** (-float(budget_db) / 10.0))
        batches = []
        for start in range(0, len(clean), args.batch_size):
            batches.append(
                pgd_untargeted(
                    victim,
                    clean[start : start + args.batch_size].to(device),
                    labels[start : start + args.batch_size].to(device),
                    budget,
                    args.steps,
                    args.step_fraction,
                    args.bandlimit_perturbation,
                    args.sample_rate_hz,
                    args.occupied_bandwidth_hz,
                ).cpu()
            )
        adversarial = torch.cat(batches)
        adv_victim = classify(victim, adversarial, args.batch_size, device)
        adv_evaluator = classify(evaluator, adversarial, args.batch_size, device)
        victim_fooled = victim_clean_correct & (adv_victim != labels)
        evaluator_retained = evaluator_clean_correct & (adv_evaluator == labels)
        dual_valid_hard = joint_clean & victim_fooled & evaluator_retained
        key = "%.1f" % budget_db
        rate_dir = output_dir / ("snr_" + key.replace(".", "p") + "db")
        rate_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(rate_dir / "adversarial_iq_float32.npy"), adversarial.numpy().astype(np.float32))
        result = {
            "budget_snr_db": budget_db,
            "budget_complex_rms": budget,
            "steps": args.steps,
            "step_fraction": args.step_fraction,
            "rf_delta": rf_metrics(
                adversarial,
                clean,
                sample_rate_hz=args.sample_rate_hz,
                occupied_bandwidth_hz=args.occupied_bandwidth_hz,
            ),
            "victim_a": {
                "clean_accuracy": float(victim_clean_correct.float().mean()),
                "adversarial_accuracy": float((adv_victim == labels).float().mean()),
                "fool_rate_on_clean_correct": float(victim_fooled.sum())
                / max(1, int(victim_clean_correct.sum())),
            },
            "evaluator_b": {
                "clean_accuracy": float(evaluator_clean_correct.float().mean()),
                "adversarial_accuracy": float((adv_evaluator == labels).float().mean()),
                "identity_retention_on_clean_correct": float(evaluator_retained.sum())
                / max(1, int(evaluator_clean_correct.sum())),
            },
            "joint_clean_examples": int(joint_clean.sum()),
            "dual_valid_hard_examples": int(dual_valid_hard.sum()),
            "dual_valid_hard_rate": float(dual_valid_hard.sum())
            / max(1, int(joint_clean.sum())),
        }
        results[key] = result
        print(
            "SNR %.1f dB | victim fool %.3f evaluator retention %.3f dual %.3f"
            % (
                budget_db,
                result["victim_a"]["fool_rate_on_clean_correct"],
                result["evaluator_b"]["identity_retention_on_clean_correct"],
                result["dual_valid_hard_rate"],
            ),
            flush=True,
        )
    report = {
        "status": "complete",
        "split": args.split,
        "examples": len(dataset),
        "samples_per_device": args.samples_per_device,
        "seed": args.seed,
        "victim_visible_to_attack": True,
        "evaluator_visible_to_attack": False,
        "bandlimit_perturbation": args.bandlimit_perturbation,
        "results": results,
        "seconds": time.time() - started,
    }
    with (output_dir / "pgd_baseline.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("report: %s" % (output_dir / "pgd_baseline.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
