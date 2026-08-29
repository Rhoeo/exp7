"""Run the P4 no-Victim waveform-validity and preliminary identity Gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from rffi_core.attacks.common_projection_pipeline import (
    actual_decoded_local_precheck,
    common_projection_pipeline,
)
from rffi_core.data.datasets import build_label_map, natural_key
from rffi_core.generators.infill.models import build_infill_model
from rffi_core.generators.infill.context_prior import BidirectionalTransitionPrior
from rffi_core.generators.token_candidates import TokenCandidateGraph
from rffi_core.generators.vqvae.models import build_reconstruction_model
from rffi_core.metrics.rf_validity_extended import (
    aggregate_metric_tensors,
    basic_validity_mask,
    rf_metric_tensors,
)
from rffi_core.victim.models import build_model


ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(r"E:\data_cache\rffi_v1")


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_integer(seed: int, namespace: str, sample_id: str) -> int:
    payload = "%d|%s|%s" % (int(seed), namespace, sample_id)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def load_index(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_balanced_development_sources(rows, role_ids, samples_per_device, seed):
    by_device = defaultdict(list)
    for row in rows:
        if row["sample_id"] in role_ids:
            by_device[row["device_id"]].append(row)
    selected = []
    for device_id in sorted(by_device, key=natural_key):
        ordered = sorted(
            by_device[device_id],
            key=lambda row: stable_integer(seed, "p4_source", row["sample_id"]),
        )
        if len(ordered) < samples_per_device:
            raise ValueError("insufficient development sources for device %s" % device_id)
        selected.extend(ordered[:samples_per_device])
    return selected


def load_codec(path: Path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    model = build_reconstruction_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_evaluator(path: Path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]
    model = build_model("evaluator_b", len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_map


def decode_batches(codec, token_array, batch_size=64):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(token_array), batch_size):
            batch = torch.as_tensor(token_array[start : start + batch_size]).long()
            outputs.append(codec.decode_code_indices(batch).cpu())
    return torch.cat(outputs, dim=0)


def classify_batches(model, iq, batch_size=32):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(iq), batch_size):
            outputs.append(model(iq[start : start + batch_size]).argmax(dim=1).cpu())
    return torch.cat(outputs)


def load_normalized_iq(rows, iq_path: Path, power_path: Path):
    iq_cache = np.load(str(iq_path), mmap_mode="r", allow_pickle=False)
    power_cache = np.load(str(power_path), mmap_mode="r", allow_pickle=False)
    outputs = []
    for row in rows:
        index = int(row["cache_index"])
        iq = np.asarray(iq_cache[index], dtype=np.float32).copy()
        iq /= np.float32(math.sqrt(max(float(power_cache[index]), 1e-12)))
        outputs.append(iq)
    return torch.from_numpy(np.stack(outputs, axis=0))


def nested_positions(rows, sequence_length, maximum_count, seed):
    result = []
    for row in rows:
        rng = np.random.RandomState(stable_integer(seed, "p4_positions", row["sample_id"]))
        result.append(rng.permutation(sequence_length)[:maximum_count].astype(np.int64))
    return np.stack(result, axis=0)


def propose_candidates(
    model,
    graph,
    reference_tokens,
    decoded_reference,
    decoded_codebook,
    positions,
    device_labels,
    context_length,
    delta_ll,
    candidate_top_k,
    transition_supported,
    precheck_config,
    transition_prior,
    transition_prior_weight,
    batch_size=32,
):
    records = []
    sequence_length = int(reference_tokens.shape[1])
    for sample_index in range(reference_tokens.shape[0]):
        for order, position in enumerate(positions[sample_index]):
            position = int(position)
            start = min(max(position - context_length // 2, 0), sequence_length - context_length)
            local = position - start
            context = reference_tokens[sample_index, start : start + context_length].copy()
            original = int(context[local])
            context[local] = graph.codebook_size
            records.append(
                {
                    "sample_index": sample_index,
                    "order": order,
                    "position": position,
                    "context_start": start,
                    "local_position": local,
                    "original": original,
                    "tokens": context,
                }
            )
    output = [[None for _ in row] for row in positions]
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        token_tensor = torch.from_numpy(
            np.stack([record["tokens"] for record in batch_records], axis=0)
        ).long()
        local_positions = torch.tensor(
            [record["local_position"] for record in batch_records], dtype=torch.long
        )
        starts = torch.tensor(
            [record["context_start"] for record in batch_records], dtype=torch.float32
        )
        global_positions = (
            starts[:, None]
            + torch.arange(context_length, dtype=torch.float32)[None, :]
        ) / float(sequence_length - 1)
        indicator = torch.zeros_like(token_tensor)
        indicator[torch.arange(len(batch_records)), local_positions] = 1
        batch_devices = torch.tensor(
            [device_labels[record["sample_index"]] for record in batch_records],
            dtype=torch.long,
        )
        with torch.inference_mode():
            logits = model(token_tensor, batch_devices, global_positions, indicator)
            selected_logits = logits[
                torch.arange(len(batch_records)), local_positions
            ]
            selected_logits = selected_logits + float(transition_prior_weight) * transition_prior.scores(
                token_tensor, indicator
            )
            log_prob = torch.log_softmax(selected_logits, dim=1)
        for batch_index, record in enumerate(batch_records):
            original = record["original"]
            prior = graph.candidates(
                original,
                top_k=candidate_top_k,
                transition_supported=transition_supported,
            )
            original_log_prob = float(log_prob[batch_index, original])
            probability_eligible = [
                int(candidate)
                for candidate in prior
                if float(log_prob[batch_index, int(candidate)])
                >= original_log_prob - float(delta_ll)
            ]
            ranked = sorted(
                probability_eligible,
                key=lambda candidate: float(log_prob[batch_index, candidate]),
                reverse=True,
            )
            rf_eligible = []
            precheck_metrics = {}
            for candidate in ranked:
                passed, metrics = actual_decoded_local_precheck(
                    decoded_reference[record["sample_index"]],
                    record["position"],
                    decoded_codebook[candidate],
                    **precheck_config,
                )
                precheck_metrics[str(candidate)] = metrics
                if passed:
                    rf_eligible.append(candidate)
            output[record["sample_index"]][record["order"]] = {
                "position": record["position"],
                "original": original,
                "prior_count": len(prior),
                "probability_count": len(probability_eligible),
                "eligible": rf_eligible,
                "best": rf_eligible[0] if rf_eligible else None,
                "precheck_metrics": precheck_metrics,
            }
    return output


def edit_tokens(reference, proposals, count, method, seed, sample_ids, codebook_size):
    edited = reference.copy()
    realized = []
    replacements = Counter()
    for sample_index, sample_proposals in enumerate(proposals):
        rng = np.random.RandomState(
            stable_integer(seed + count, method, sample_ids[sample_index])
        )
        sample_realized = 0
        for proposal in sample_proposals[:count]:
            position = proposal["position"]
            original = proposal["original"]
            if method == "infill":
                candidate = proposal["best"]
            elif method == "random_plausible":
                eligible = proposal["eligible"]
                candidate = int(eligible[int(rng.randint(0, len(eligible)))]) if eligible else None
            elif method == "full_codebook_random_diagnostic":
                candidate = int(rng.randint(0, codebook_size - 1))
                if candidate >= original:
                    candidate += 1
            else:
                raise ValueError("unknown method")
            if candidate is not None and candidate != original:
                edited[sample_index, position] = candidate
                sample_realized += 1
                replacements[candidate] += 1
        realized.append(sample_realized)
    return edited, np.asarray(realized, dtype=np.int64), replacements


def identity_metrics(predictions, baseline_predictions, labels, devices):
    baseline_correct = baseline_predictions == labels
    retained = baseline_correct & (predictions == labels)
    pooled = float(retained.sum()) / max(1, int(baseline_correct.sum()))
    per_device = {}
    for device_id in sorted(set(devices), key=natural_key):
        mask = torch.tensor([value == device_id for value in devices], dtype=torch.bool)
        eligible = baseline_correct & mask
        per_device[device_id] = float((retained & mask).sum()) / max(1, int(eligible.sum()))
    values = list(per_device.values())
    return {
        "pooled_retention_vs_codec_correct": pooled,
        "macro_retention_vs_codec_correct": float(np.mean(values)),
        "minimum_device_retention": float(np.min(values)),
        "per_device": per_device,
        "codec_correct_sources": int(baseline_correct.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/generator/wifib_infill_gate_v1.json"
    )
    parser.add_argument(
        "--infill-checkpoint", default="runs/next_stage/p3_infill_hybrid_v2/best.pt"
    )
    parser.add_argument(
        "--codec-checkpoint", default="runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt"
    )
    parser.add_argument(
        "--evaluator-checkpoint", default="runs/stage_g0/frozen/wifib_v1/evaluator_b.pt"
    )
    parser.add_argument(
        "--graph", default="artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz"
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_config = json.loads(
        (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
    )
    window_index = CACHE_ROOT / "wifib/window_index.csv"
    rows = load_index(window_index)
    role_ids = set(split_config["roles"][config["source_role"]])
    selected_rows = select_balanced_development_sources(
        rows, role_ids, config["samples_per_device"], config["seed"]
    )
    sample_ids = [row["sample_id"] for row in selected_rows]
    devices = [row["device_id"] for row in selected_rows]
    label_map = build_label_map(row["device_id"] for row in rows)
    labels = torch.tensor([label_map[value] for value in devices], dtype=torch.long)
    device_labels = [label_map[value] for value in devices]

    token_cache = np.load(
        str(CACHE_ROOT / "tokens/wifib_vq_p1_k1024/tokens.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    reference_tokens = np.stack(
        [np.asarray(token_cache[int(row["cache_index"])], dtype=np.int64) for row in selected_rows]
    )
    raw_clean = load_normalized_iq(
        selected_rows,
        CACHE_ROOT / "wifib/iq_window_2048_float32.npy",
        CACHE_ROOT / "wifib/power_window_2048_float32.npy",
    )
    codec = load_codec(resolve(args.codec_checkpoint))
    decoded_reference = decode_batches(codec, reference_tokens)
    with torch.inference_mode():
        decoded_codebook = codec.decode_code_indices(
            torch.arange(1024, dtype=torch.long)[:, None]
        ).cpu()[:, :, 0]

    infill_checkpoint = torch.load(
        str(resolve(args.infill_checkpoint)), map_location="cpu", weights_only=False
    )
    infill = build_infill_model(infill_checkpoint["model_config"])
    infill.load_state_dict(infill_checkpoint["model_state"])
    infill.eval()
    graph = TokenCandidateGraph.load(resolve(args.graph))
    transition_prior = BidirectionalTransitionPrior(
        graph.transition_counts,
        graph.arrays["usage_counts"],
        alpha=infill_checkpoint["config"]["training"]["transition_prior_alpha"],
    )
    positions = nested_positions(
        selected_rows,
        reference_tokens.shape[1],
        max(config["edit_counts"]),
        config["seed"],
    )
    proposals = propose_candidates(
        infill,
        graph,
        reference_tokens,
        decoded_reference,
        decoded_codebook,
        positions,
        device_labels,
        infill_checkpoint["model_config"]["context_length"],
        config["delta_ll"],
        config["candidate_top_k"],
        config["transition_supported"],
        config["local_precheck"],
        transition_prior,
        infill_checkpoint["config"]["training"]["transition_prior_weight"],
    )
    proposal_rows = [proposal for sample in proposals for proposal in sample]
    proposal_metrics = {
        "positions": len(proposal_rows),
        "mean_prior_candidates": float(np.mean([row["prior_count"] for row in proposal_rows])),
        "mean_probability_candidates": float(
            np.mean([row["probability_count"] for row in proposal_rows])
        ),
        "mean_rf_eligible_candidates": float(
            np.mean([len(row["eligible"]) for row in proposal_rows])
        ),
        "probability_candidate_coverage": float(
            np.mean([row["probability_count"] > 0 for row in proposal_rows])
        ),
        "actual_decoded_rf_candidate_coverage": float(
            np.mean([len(row["eligible"]) > 0 for row in proposal_rows])
        ),
    }

    evaluator, evaluator_label_map = load_evaluator(resolve(args.evaluator_checkpoint))
    if evaluator_label_map != label_map:
        raise ValueError("Evaluator B label map differs from source label map")
    raw_predictions = classify_batches(evaluator, raw_clean)
    codec_predictions = classify_batches(evaluator, decoded_reference)
    projection_config = config["projection"]
    methods = (
        "random_plausible",
        "infill",
        "full_codebook_random_diagnostic",
    )
    results = {}
    for edit_count in config["edit_counts"]:
        count_results = {}
        for method in methods:
            edited_tokens, realized, replacements = edit_tokens(
                reference_tokens,
                proposals,
                edit_count,
                method,
                config["seed"],
                sample_ids,
                graph.codebook_size,
            )
            decoded = decode_batches(codec, edited_tokens)
            pre_metrics_tensor = rf_metric_tensors(
                decoded,
                decoded_reference,
                sample_rate_hz=projection_config["sample_rate_hz"],
                occupied_bandwidth_hz=projection_config["occupied_bandwidth_hz"],
            )
            projected, projection_diagnostics = common_projection_pipeline(
                decoded_reference, decoded, **projection_config
            )
            post_metrics_tensor = rf_metric_tensors(
                projected,
                decoded_reference,
                sample_rate_hz=projection_config["sample_rate_hz"],
                occupied_bandwidth_hz=projection_config["occupied_bandwidth_hz"],
            )
            predictions = classify_batches(evaluator, projected)
            validity = basic_validity_mask(
                post_metrics_tensor,
                min_snr_db=projection_config["min_snr_db"],
                max_normalized_peak_delta=projection_config[
                    "max_normalized_peak_delta"
                ],
            )
            replacement_total = sum(replacements.values())
            probabilities = (
                np.asarray(list(replacements.values()), dtype=np.float64)
                / replacement_total
                if replacement_total
                else np.asarray([], dtype=np.float64)
            )
            count_results[method] = {
                "requested_edit_count": edit_count,
                "realized_edit_count_mean": float(realized.mean()),
                "realized_full_count_fraction": float(np.mean(realized == edit_count)),
                "replacement_token_entropy_nats": (
                    float(-np.sum(probabilities * np.log(probabilities)))
                    if len(probabilities)
                    else 0.0
                ),
                "dominant_replacement_fraction": (
                    max(replacements.values()) / replacement_total
                    if replacement_total
                    else 0.0
                ),
                "pre_projection": aggregate_metric_tensors(pre_metrics_tensor),
                "post_projection": aggregate_metric_tensors(post_metrics_tensor),
                "basic_waveform_valid_fraction": float(validity.float().mean()),
                "projection_changed_fraction": float(
                    projection_diagnostics["changed_fraction"].mean()
                ),
                "evaluator_b": identity_metrics(
                    predictions, codec_predictions, labels, devices
                ),
            }
        results[str(edit_count)] = count_results

    codec_identity = identity_metrics(codec_predictions, raw_predictions, labels, devices)
    infill_validation = json.loads(
        (ROOT / "reports/next_stage/infill_validation.json").read_text(encoding="utf-8")
    )
    uniform_cross_entropy = math.log(graph.codebook_size)
    language_gain = uniform_cross_entropy - float(infill_validation["loss"])
    last = results[str(max(config["edit_counts"]))]
    random_pre = last["full_codebook_random_diagnostic"]["pre_projection"]
    infill_pre = last["infill"]["pre_projection"]
    peak_improvement = 1.0 - (
        infill_pre["normalized_peak_delta"]["p95"]
        / max(random_pre["normalized_peak_delta"]["p95"], 1e-12)
    )
    derivative_improvement = 1.0 - (
        infill_pre["normalized_max_delta_derivative"]["p95"]
        / max(random_pre["normalized_max_delta_derivative"]["p95"], 1e-12)
    )
    identity = last["infill"]["evaluator_b"]
    thresholds = config["gate"]
    checks = {
        "language_cross_entropy_gain": language_gain
        >= thresholds["minimum_cross_entropy_gain_vs_uniform_nats"],
        "language_original_top5": infill_validation["top5_accuracy"]
        >= thresholds["minimum_original_top5_accuracy"],
        "candidate_position_coverage": proposal_metrics[
            "actual_decoded_rf_candidate_coverage"
        ]
        >= thresholds["minimum_candidate_position_coverage"],
        "identity_pooled": identity["pooled_retention_vs_codec_correct"]
        >= thresholds["minimum_identity_pooled"],
        "identity_macro": identity["macro_retention_vs_codec_correct"]
        >= thresholds["minimum_identity_macro"],
        "identity_min_device": identity["minimum_device_retention"]
        >= thresholds["minimum_identity_device_for_10_samples"],
        "waveform_peak_tail_improvement": peak_improvement
        >= thresholds["minimum_waveform_tail_improvement_vs_full_random"],
        "waveform_derivative_tail_improvement": derivative_improvement
        >= thresholds["minimum_waveform_tail_improvement_vs_full_random"],
    }
    gate_passed = all(checks.values())
    source_ids_path = ROOT / "reports/next_stage/p4_development_source_ids.json"
    source_ids_path.write_text(
        json.dumps(
            {
                "role": config["source_role"],
                "seed": config["seed"],
                "sample_ids": sample_ids,
                "policy_gate_used": False,
                "final_test_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": "wifib-infill-waveform-gate-v1",
        "stage": "P4_waveform_validity_and_infill_gate",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_role": config["source_role"],
        "source_count": len(selected_rows),
        "samples_per_device": config["samples_per_device"],
        "policy_gate_used": False,
        "victim_queries": 0,
        "evaluator_b_role": "preliminary offline identity audit only",
        "final_test_used": False,
        "codec_identity_vs_raw_clean": codec_identity,
        "proposal_metrics": proposal_metrics,
        "language_model": {
            "uniform_cross_entropy": uniform_cross_entropy,
            "validation_cross_entropy": infill_validation["loss"],
            "cross_entropy_gain": language_gain,
            "original_top5_accuracy": infill_validation["top5_accuracy"],
        },
        "results": results,
        "waveform_tail_improvement_at_max_edit": {
            "normalized_peak_delta_p95": peak_improvement,
            "normalized_max_delta_derivative_p95": derivative_improvement,
        },
        "gate_checks": checks,
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "reason": (
                "Masked-infill has learned context signal and passes candidate, identity, and waveform checks."
                if gate_passed
                else "At least one pre-registered language, candidate, identity, or waveform check failed; P5 is blocked."
            ),
        },
        "terminology": "digital waveform-validity only; no physical/OTA feasibility claim",
    }
    report_dir = ROOT / "reports/next_stage"
    report_path = report_dir / "p4_infill_gate_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    failed_checks = [name for name, passed in checks.items() if not passed]
    summary = [
        "# P4 Waveform Validity and Infill Gate",
        "",
        "- Gate: **%s**" % report["gate"]["status"],
        "- Development sources: %d (%d/device)" % (len(selected_rows), config["samples_per_device"]),
        "- Victim A queries: **0**",
        "- Policy Gate used: **false**",
        "- Final test used: **false**",
        "- Candidate coverage after latent/transition + probability + actual-decoding checks: %.3f"
        % proposal_metrics["actual_decoded_rf_candidate_coverage"],
        "- Cross-entropy gain over uniform: %.4f nats" % language_gain,
        "- Max-edit B pooled/macro/min-device retention: %.3f / %.3f / %.3f"
        % (
            identity["pooled_retention_vs_codec_correct"],
            identity["macro_retention_vs_codec_correct"],
            identity["minimum_device_retention"],
        ),
        "- Infill peak/derivative p95 improvement over full-codebook random diagnostic: %.3f / %.3f"
        % (peak_improvement, derivative_improvement),
        "- Failed checks: `%s`" % (", ".join(failed_checks) if failed_checks else "none"),
        "",
        "This Gate evaluates digital waveform plausibility only. Evaluator B is a preliminary offline auditor and no over-the-air feasibility is claimed.",
    ]
    summary_path = report_dir / "p4_infill_gate_report.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    stage_report = {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": report["stage"],
        "git_commit": None,
        "config_hash": report["config_sha256"],
        "data_split_hash": split_config["role_assignment_sha256"],
        "checkpoint_hash": infill_validation["checkpoint_sha256"],
        "seeds": [config["seed"]],
        "source_count": len(selected_rows),
        "victim_query_budget": 0,
        "edit_budget": config["edit_counts"],
        "metrics": report,
        "gate": report["gate"],
        "artifacts": {
            "report_json": str(report_path),
            "report_markdown": str(summary_path),
            "source_ids": str(source_ids_path),
            "metric_definition": str(report_dir / "rf_metric_definition.md"),
        },
        "next_allowed_stage": "P5a_screening" if gate_passed else None,
        "prohibited_actions": [
            "run P5 while Gate is FAIL",
            "use policy_gate attack outcomes for development",
            "access final_test signal data",
            "train PPO",
        ],
    }
    (report_dir / "p4_stage_report.json").write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "gate": report["gate"], "failed_checks": failed_checks}, indent=2))


if __name__ == "__main__":
    main()
