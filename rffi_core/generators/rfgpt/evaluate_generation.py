#!/usr/bin/env python
"""Sample conditional RF-GPT waveforms and apply identity/memorization gates."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rffi_core.data.build_data_manifest import load_json
from rffi_core.data.token_datasets import WiFiBTokenDataset
from rffi_core.generators.rfgpt.models import build_rfgpt
from rffi_core.generators.vqvae.losses import mean_complex_power, normalize_complex_power
from rffi_core.generators.vqvae.models import build_reconstruction_model
from rffi_core.generators.vqvae.train_reconstruction import choose_device
from rffi_core.victim.models import build_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt")
    parser.add_argument("--initial-token-counts", required=True)
    parser.add_argument("--frozen-registry", default="runs/stage_g0/frozen/wifib_v1/frozen_models.json")
    parser.add_argument("--data-config", default="configs/data/rffi_data_v1.json")
    parser.add_argument(
        "--token-cache",
        default="E:/data_cache/rffi_v1/tokens/wifib_vq_p1_k1024/tokens.npy",
    )
    parser.add_argument("--samples-per-device", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--nearest-train-candidates", type=int, default=256)
    parser.add_argument("--prefix-length", type=int, default=0)
    parser.add_argument("--prefix-split", default="reward_validation")
    parser.add_argument("--batch-size", type=int, default=17)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--min-evaluator-identity", type=float, default=0.8)
    parser.add_argument("--min-victim-identity", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def sample_logits(logits, temperature, top_p):
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / float(temperature)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        probabilities = torch.softmax(sorted_logits, dim=-1)
        selected = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        return sorted_indices.gather(1, selected.unsqueeze(1)).squeeze(1)
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).squeeze(1)


def generate_batch(
    model,
    labels,
    initial_counts,
    sequence_length,
    temperature,
    top_p,
    prefix_tokens=None,
):
    device = labels.device
    tokens = torch.empty(
        labels.shape[0], sequence_length, dtype=torch.long, device=device
    )
    if prefix_tokens is not None:
        prefix_tokens = prefix_tokens.to(device=device, dtype=torch.long)
        if prefix_tokens.shape[0] != labels.shape[0]:
            raise ValueError("prefix batch differs from label batch")
        prefix_length = int(prefix_tokens.shape[1])
        if prefix_length <= 0 or prefix_length >= sequence_length:
            raise ValueError("prefix length must be between 1 and sequence_length - 1")
        tokens[:, :prefix_length] = prefix_tokens
        first_generated_position = prefix_length
    else:
        rows = initial_counts[labels.cpu().numpy()]
        rows = torch.from_numpy(rows.astype(np.float32)).to(device)
        rows = rows + 1e-6
        first = torch.multinomial(rows / rows.sum(dim=1, keepdim=True), 1).squeeze(1)
        tokens[:, 0] = first
        first_generated_position = 1
    for position in range(first_generated_position, sequence_length):
        start = max(0, position - model.context_length)
        context = tokens[:, start:position]
        logits = model(context, labels, position_offset=start)[:, -1]
        tokens[:, position] = sample_logits(logits, temperature, top_p)
        if position % 512 == 0 or position + 1 == sequence_length:
            print("sampled %d/%d tokens" % (position + 1, sequence_length), flush=True)
    return tokens


def select_reference_prefixes(dataset, samples_per_device, prefix_length):
    selected = {}
    for index in range(len(dataset)):
        item = dataset[index]
        label = int(item["label"])
        selected.setdefault(label, [])
        if len(selected[label]) < samples_per_device:
            selected[label].append(item)
    missing = [
        label
        for label in range(len(dataset.label_map))
        if len(selected.get(label, [])) < samples_per_device
    ]
    if missing:
        raise ValueError("prefix split lacks samples for labels %s" % missing)
    prefixes = []
    sample_ids = []
    for label in range(len(dataset.label_map)):
        for item in selected[label]:
            prefixes.append(item["tokens"][:prefix_length])
            sample_ids.append(item["sample_id"])
    return np.stack(prefixes), sample_ids


def load_classifier(path, label_map, device):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if checkpoint["label_map"] != label_map:
        raise ValueError("classifier label map differs from RF-GPT label map")
    architecture = checkpoint["arguments"]["model"]
    model = build_model(architecture, len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def sequence_hash(tokens):
    return hashlib.sha256(np.asarray(tokens, dtype=np.uint16).tobytes()).digest()


def memorization_metrics(generated, generated_labels, train_dataset, candidates, seed):
    exact_hashes = set()
    by_label = {}
    for index in range(len(train_dataset)):
        item = train_dataset[index]
        exact_hashes.add(sequence_hash(item["tokens"]))
        by_label.setdefault(int(item["label"]), []).append(index)
    rng = np.random.RandomState(seed)
    maximum_agreements = []
    for tokens, label in zip(generated, generated_labels):
        indices = by_label[int(label)]
        if len(indices) > candidates:
            indices = rng.choice(indices, candidates, replace=False).tolist()
        references = np.stack([train_dataset[index]["tokens"] for index in indices])
        maximum_agreements.append(float((references == tokens).mean(axis=1).max()))
    exact_duplicates = sum(sequence_hash(tokens) in exact_hashes for tokens in generated)
    return {
        "exact_training_duplicates": int(exact_duplicates),
        "maximum_same_device_token_agreement_mean": float(np.mean(maximum_agreements)),
        "maximum_same_device_token_agreement_max": float(np.max(maximum_agreements)),
        "nearest_candidates_per_device": int(candidates),
    }


def main(argv=None):
    args = parse_args(argv)
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]
    inverse_labels = [None] * len(label_map)
    for name, label in label_map.items():
        inverse_labels[int(label)] = name
    model = build_rfgpt(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    tokenizer_checkpoint = torch.load(
        str(Path(args.tokenizer).resolve()), map_location="cpu", weights_only=False
    )
    tokenizer = build_reconstruction_model(tokenizer_checkpoint["model_config"])
    tokenizer.load_state_dict(tokenizer_checkpoint["model_state"])
    tokenizer.to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    initial_counts = np.load(str(Path(args.initial_token_counts).resolve()))
    if initial_counts.shape != (len(label_map), model.codebook_size):
        raise ValueError("initial token count shape differs from RF-GPT configuration")
    labels = np.repeat(np.arange(len(label_map)), args.samples_per_device)
    config = load_json(Path(args.data_config).resolve())
    index_path = Path(config["cache_root"]).resolve() / "wifib" / "window_index.csv"
    prefix_tokens = None
    prefix_sample_ids = None
    if args.prefix_length > 0:
        if args.prefix_split == "final_test":
            raise ValueError("final_test cannot be used as a generation prefix source")
        prefix_dataset = WiFiBTokenDataset(
            index_path, Path(args.token_cache).resolve(), args.prefix_split
        )
        if prefix_dataset.label_map != label_map:
            raise ValueError("prefix dataset label map differs from RF-GPT")
        prefix_tokens, prefix_sample_ids = select_reference_prefixes(
            prefix_dataset, args.samples_per_device, args.prefix_length
        )
    generated_batches = []
    started = time.time()
    with torch.inference_mode():
        for start in range(0, len(labels), args.batch_size):
            selected = torch.from_numpy(labels[start : start + args.batch_size]).to(
                device=device, dtype=torch.long
            )
            generated_batches.append(
                generate_batch(
                    model,
                    selected,
                    initial_counts,
                    model.max_sequence_length,
                    args.temperature,
                    args.top_p,
                    None
                    if prefix_tokens is None
                    else torch.from_numpy(
                        prefix_tokens[start : start + args.batch_size]
                    ),
                ).cpu()
            )
        generated_tokens = torch.cat(generated_batches, dim=0)
        iq_batches = []
        for start in range(0, len(labels), args.batch_size):
            selected = generated_tokens[start : start + args.batch_size].to(device)
            iq_batches.append(tokenizer.decode_code_indices(selected).cpu())
        generated_iq = torch.cat(iq_batches, dim=0)

    normalized_iq = normalize_complex_power(generated_iq)
    registry_path = Path(args.frozen_registry).resolve()
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    classification = {}
    intended = torch.from_numpy(labels).long()
    with torch.inference_mode():
        for name in ("victim_a", "evaluator_b"):
            classifier = load_classifier(
                Path(registry["models"][name]["checkpoint"]), label_map, device
            )
            predictions = []
            for start in range(0, len(labels), args.batch_size):
                inputs = normalized_iq[start : start + args.batch_size].to(device)
                predictions.append(classifier(inputs).argmax(dim=1).cpu())
            predictions = torch.cat(predictions)
            correct = predictions == intended
            per_device = {}
            for label, device_name in enumerate(inverse_labels):
                selected = intended == label
                per_device[device_name] = float(correct[selected].float().mean())
            classification[name] = {
                "conditional_identity_accuracy": float(correct.float().mean()),
                "correct": int(correct.sum()),
                "examples": len(labels),
                "per_device_accuracy": per_device,
                "predictions": [inverse_labels[int(value)] for value in predictions],
            }

    token_array = generated_tokens.numpy().astype(np.uint16)
    iq_array = generated_iq.numpy().astype(np.float32)
    powers = mean_complex_power(generated_iq).numpy()
    complex_peak = generated_iq[:, 0].square() + generated_iq[:, 1].square()
    papr = (complex_peak.amax(dim=1) / mean_complex_power(generated_iq).clamp_min(1e-8)).numpy()
    unique_sequences = len({sequence_hash(row) for row in token_array})
    token_histogram = np.bincount(token_array.reshape(-1), minlength=model.codebook_size)
    probabilities = token_histogram / token_histogram.sum()
    active = probabilities > 0
    token_perplexity = float(
        np.exp(-np.sum(probabilities[active] * np.log(probabilities[active])))
    )
    train_dataset = WiFiBTokenDataset(
        index_path,
        Path(args.token_cache).resolve(),
        "generator_train",
    )
    memorization = memorization_metrics(
        token_array,
        labels,
        train_dataset,
        args.nearest_train_candidates,
        args.seed,
    )
    gate_checks = {
        "finite_iq": bool(np.isfinite(iq_array).all()),
        "unique_sequences": unique_sequences == len(labels),
        "no_exact_training_duplicate": memorization["exact_training_duplicates"] == 0,
        "victim_conditional_identity": classification["victim_a"]
        ["conditional_identity_accuracy"]
        >= args.min_victim_identity,
        "evaluator_conditional_identity": classification["evaluator_b"]
        ["conditional_identity_accuracy"]
        >= args.min_evaluator_identity,
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "generated_tokens.npy"), token_array)
    np.save(str(output_dir / "generated_iq_float32.npy"), iq_array)
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "training_used_classifier_feedback": False,
        "sampling": {
            "samples_per_device": args.samples_per_device,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "seconds": time.time() - started,
            "prefix_length": args.prefix_length,
            "prefix_split": args.prefix_split if args.prefix_length > 0 else None,
            "prefix_sample_ids": prefix_sample_ids,
        },
        "labels": inverse_labels,
        "classification": classification,
        "rf_validity": {
            "finite_fraction": float(np.isfinite(iq_array).mean()),
            "complex_power_mean": float(powers.mean()),
            "complex_power_std": float(powers.std()),
            "papr_mean": float(papr.mean()),
            "papr_max": float(papr.max()),
        },
        "diversity": {
            "unique_sequences": unique_sequences,
            "unique_fraction": unique_sequences / len(labels),
            "active_codes": int(active.sum()),
            "token_perplexity": token_perplexity,
        },
        "memorization": memorization,
        "conditional_generation_gate": {
            "passed": all(gate_checks.values()),
            "checks": gate_checks,
            "thresholds": {
                "min_victim_identity": args.min_victim_identity,
                "min_evaluator_identity": args.min_evaluator_identity,
            },
        },
    }
    with (output_dir / "generation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        "identity: victim %.3f evaluator %.3f | unique %d/%d | duplicates %d"
        % (
            classification["victim_a"]["conditional_identity_accuracy"],
            classification["evaluator_b"]["conditional_identity_accuracy"],
            unique_sequences,
            len(labels),
            memorization["exact_training_duplicates"],
        ),
        flush=True,
    )
    print(
        "generation gate: %s" % ("PASS" if all(gate_checks.values()) else "FAIL"),
        flush=True,
    )
    print("report: %s" % (output_dir / "generation_report.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
