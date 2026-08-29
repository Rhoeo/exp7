"""Run the reduced P5a development screening with explicit query accounting."""

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

from rffi_core.attacks.common_projection_pipeline import common_projection_pipeline
from rffi_core.attacks.query_search import (
    normalized_position_prerank,
    select_unique_position_actions,
    true_class_margin,
)
from rffi_core.data.datasets import build_label_map, natural_key
from rffi_core.generators.infill.context_prior import BidirectionalTransitionPrior
from rffi_core.generators.infill.evaluate_waveform_gate import (
    classify_batches,
    decode_batches,
    load_codec,
    load_evaluator,
    load_index,
    load_normalized_iq,
    nested_positions,
    propose_candidates,
)
from rffi_core.generators.infill.models import build_infill_model
from rffi_core.generators.token_candidates import TokenCandidateGraph
from rffi_core.metrics.rf_validity_extended import (
    aggregate_metric_tensors,
    basic_validity_mask,
    rf_metric_tensors,
)
from rffi_core.victim.models import build_model


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(r"E:\data_cache\rffi_v1")


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_integer(seed, namespace, sample_id):
    payload = "%d|%s|%s" % (int(seed), namespace, sample_id)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def load_victim(path):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    label_map = checkpoint["label_map"]
    model = build_model("victim_a", len(label_map))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, label_map


def classify_logits(model, iq, batch_size=64):
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(iq), batch_size):
            outputs.append(model(iq[start : start + batch_size]).cpu())
    return torch.cat(outputs, dim=0)


def select_development_pool(
    rows,
    role_ids,
    victim,
    evaluator,
    label_map,
    samples_per_device,
    seed,
):
    candidates = [row for row in rows if row["sample_id"] in role_ids]
    candidates.sort(
        key=lambda row: (
            natural_key(row["device_id"]),
            stable_integer(seed, "p5a_eligibility", row["sample_id"]),
        )
    )
    iq = load_normalized_iq(
        candidates,
        CACHE_ROOT / "wifib/iq_window_2048_float32.npy",
        CACHE_ROOT / "wifib/power_window_2048_float32.npy",
    )
    labels = torch.tensor(
        [label_map[row["device_id"]] for row in candidates], dtype=torch.long
    )
    victim_predictions = classify_batches(victim, iq)
    evaluator_predictions = classify_batches(evaluator, iq)
    selected = []
    audit = []
    selected_counts = Counter()
    for index, row in enumerate(candidates):
        victim_correct = int(victim_predictions[index]) == int(labels[index])
        evaluator_correct = int(evaluator_predictions[index]) == int(labels[index])
        eligible = victim_correct and evaluator_correct
        choose = eligible and selected_counts[row["device_id"]] < samples_per_device
        if choose:
            selected.append(row)
            selected_counts[row["device_id"]] += 1
        audit.append(
            {
                "sample_id": row["sample_id"],
                "device_id": row["device_id"],
                "cache_index": row["cache_index"],
                "victim_clean_correct": victim_correct,
                "evaluator_clean_correct": evaluator_correct,
                "dual_clean_correct": eligible,
                "selected": choose,
            }
        )
    if len(selected_counts) != 17 or any(
        selected_counts[device_id] != samples_per_device for device_id in selected_counts
    ):
        raise RuntimeError("could not build balanced P5a dual-clean-correct source pool")
    return selected, audit


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nearest_eligible(graph, original, eligible):
    distance = {
        int(candidate): float(value)
        for candidate, value in zip(
            graph.arrays["neighbors"][original],
            graph.arrays["latent_distances"][original],
        )
    }
    return min(eligible, key=lambda candidate: (distance.get(int(candidate), float("inf")), int(candidate)))


def edit_random_method(
    reference_tokens,
    proposals,
    edit_count,
    method,
    sample_ids,
    graph,
    seed,
):
    edited = reference_tokens.copy()
    realized = np.zeros(len(reference_tokens), dtype=np.int64)
    for sample_index, sample_proposals in enumerate(proposals):
        rng = np.random.RandomState(
            stable_integer(seed + edit_count, method, sample_ids[sample_index])
        )
        for proposal in sample_proposals[:edit_count]:
            eligible = proposal["eligible"]
            if not eligible:
                continue
            if method == "random_uniform_plausible":
                candidate = int(eligible[int(rng.randint(0, len(eligible)))])
            elif method == "random_nearest":
                candidate = int(nearest_eligible(graph, proposal["original"], eligible))
            elif method == "random_infill":
                candidate = int(proposal["best"])
            else:
                raise ValueError("unknown random method")
            edited[sample_index, proposal["position"]] = candidate
            realized[sample_index] += 1
    return edited, realized


def query_single_edit_actions(
    victim,
    codec,
    reference_tokens,
    decoded_reference,
    proposals,
    labels,
    projection_config,
    candidates_per_position,
    chunk_size=256,
):
    actions = []
    for sample_index, sample_proposals in enumerate(proposals):
        for position_rank, proposal in enumerate(sample_proposals):
            for candidate in proposal["eligible"][:candidates_per_position]:
                actions.append(
                    {
                        "sample_index": sample_index,
                        "position_rank": position_rank,
                        "position": int(proposal["position"]),
                        "candidate": int(candidate),
                    }
                )
    for start in range(0, len(actions), chunk_size):
        batch_actions = actions[start : start + chunk_size]
        token_batch = np.stack(
            [reference_tokens[action["sample_index"]].copy() for action in batch_actions]
        )
        for batch_index, action in enumerate(batch_actions):
            token_batch[batch_index, action["position"]] = action["candidate"]
        decoded = decode_batches(codec, token_batch)
        sample_indices = [action["sample_index"] for action in batch_actions]
        reference = decoded_reference[sample_indices]
        projected, _ = common_projection_pipeline(
            reference, decoded, **projection_config
        )
        logits = classify_logits(victim, projected, batch_size=64)
        batch_labels = labels[sample_indices]
        margins = true_class_margin(logits, batch_labels)
        for action, margin in zip(batch_actions, margins.tolist()):
            action["margin"] = float(margin)
    by_sample = defaultdict(list)
    for action in actions:
        by_sample[action["sample_index"]].append(action)
    return by_sample, len(actions)


def edit_greedy(
    reference_tokens,
    actions_by_sample,
    edit_count,
    position_limit,
    baseline_margins,
):
    edited = reference_tokens.copy()
    realized = np.zeros(len(reference_tokens), dtype=np.int64)
    query_counts = np.zeros(len(reference_tokens), dtype=np.int64)
    for sample_index in range(len(reference_tokens)):
        actions = [
            action
            for action in actions_by_sample[sample_index]
            if action["position_rank"] < position_limit
        ]
        query_counts[sample_index] = len(actions)
        selected = select_unique_position_actions(
            actions, edit_count, baseline_margins[sample_index]
        )
        for action in selected:
            edited[sample_index, action["position"]] = action["candidate"]
        realized[sample_index] = len(selected)
    return edited, realized, query_counts


def bootstrap_rate(mask, eligible, seed, iterations=2000):
    indices = np.flatnonzero(eligible)
    if len(indices) == 0:
        return [0.0, 0.0]
    rng = np.random.RandomState(seed)
    values = mask.astype(np.float64)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draw = rng.choice(indices, size=len(indices), replace=True)
        estimates[index] = values[draw].mean()
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def evaluate_variant(
    victim,
    evaluator,
    codec,
    edited_tokens,
    decoded_reference,
    labels,
    devices,
    codec_victim_predictions,
    codec_evaluator_predictions,
    codec_margins,
    realized,
    projection_config,
    search_query_counts,
    seed,
    requested_edit_count,
    return_masks=False,
):
    decoded = decode_batches(codec, edited_tokens)
    projected, projection_diagnostics = common_projection_pipeline(
        decoded_reference, decoded, **projection_config
    )
    metric_tensors = rf_metric_tensors(
        projected,
        decoded_reference,
        sample_rate_hz=projection_config["sample_rate_hz"],
        occupied_bandwidth_hz=projection_config["occupied_bandwidth_hz"],
    )
    valid = basic_validity_mask(
        metric_tensors,
        min_snr_db=projection_config["min_snr_db"],
        max_normalized_peak_delta=projection_config["max_normalized_peak_delta"],
    )
    victim_logits = classify_logits(victim, projected)
    victim_predictions = victim_logits.argmax(dim=1)
    victim_margins = true_class_margin(victim_logits, labels)
    evaluator_predictions = classify_batches(evaluator, projected)
    codec_joint = (codec_victim_predictions == labels) & (
        codec_evaluator_predictions == labels
    )
    fooled = victim_predictions != labels
    retained = evaluator_predictions == labels
    valid_hard = codec_joint & valid & fooled & retained
    per_device = {}
    for device_id in sorted(set(devices), key=natural_key):
        device_mask = torch.tensor(
            [value == device_id for value in devices], dtype=torch.bool
        )
        denominator = codec_joint & device_mask
        per_device[device_id] = {
            "eligible": int(denominator.sum()),
            "valid_hard": int((valid_hard & device_mask).sum()),
            "valid_hard_rate": float((valid_hard & device_mask).sum())
            / max(1, int(denominator.sum())),
            "evaluator_retention": float((retained & denominator).sum())
            / max(1, int(denominator.sum())),
        }
    macro = float(np.mean([value["valid_hard_rate"] for value in per_device.values()]))
    pooled = float(valid_hard.sum()) / max(1, int(codec_joint.sum()))
    result = {
        "source_count": len(labels),
        "codec_joint_correct_count": int(codec_joint.sum()),
        "requested_edit_count": int(requested_edit_count),
        "realized_edit_count_mean": float(np.mean(realized)),
        "realized_full_fraction": float(np.mean(realized == int(requested_edit_count))) if len(realized) else 0.0,
        "search_queries_mean": float(np.mean(search_query_counts)),
        "search_queries_max": int(np.max(search_query_counts)),
        "final_audit_queries_per_source": 1,
        "victim_margin_reduction_mean": float(
            (codec_margins[codec_joint] - victim_margins[codec_joint]).mean()
        ),
        "victim_fool_rate_on_codec_correct": float((fooled & codec_joint).sum())
        / max(1, int(codec_joint.sum())),
        "evaluator_retention_on_codec_correct": float((retained & codec_joint).sum())
        / max(1, int(codec_joint.sum())),
        "rf_valid_fraction": float(valid.float().mean()),
        "valid_hard_rate_pooled": pooled,
        "valid_hard_rate_macro": macro,
        "valid_hard_bootstrap_95ci": bootstrap_rate(
            valid_hard.numpy(), codec_joint.numpy(), seed
        ),
        "per_device": per_device,
        "waveform_metrics": aggregate_metric_tensors(metric_tensors),
        "projection_changed_fraction": float(
            projection_diagnostics["changed_fraction"].mean()
        ),
    }
    if return_masks:
        return result, valid_hard.numpy().astype(np.bool_), codec_joint.numpy().astype(np.bool_)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/attack/wifib_p5a_screening_v1.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_config = json.loads(
        (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
    )
    rows = load_index(CACHE_ROOT / "wifib/window_index.csv")
    label_map = build_label_map(row["device_id"] for row in rows)
    victim, victim_label_map = load_victim(
        ROOT / "runs/stage_g0/frozen/wifib_v1/victim_a.pt"
    )
    evaluator, evaluator_label_map = load_evaluator(
        ROOT / "runs/stage_g0/frozen/wifib_v1/evaluator_b.pt"
    )
    if label_map != victim_label_map or label_map != evaluator_label_map:
        raise ValueError("source and classifier label maps differ")
    selected_rows, eligibility_audit = select_development_pool(
        rows,
        set(split_config["roles"][config["source_role"]]),
        victim,
        evaluator,
        label_map,
        config["samples_per_device"],
        config["seed"],
    )
    report_dir = ROOT / "reports/next_stage"
    write_csv(report_dir / "p5a_source_pool.csv", [row for row in eligibility_audit if row["selected"]])
    write_csv(report_dir / "p5a_source_eligibility_audit.csv", eligibility_audit)
    sample_ids = [row["sample_id"] for row in selected_rows]
    devices = [row["device_id"] for row in selected_rows]
    labels = torch.tensor([label_map[value] for value in devices], dtype=torch.long)
    raw_clean = load_normalized_iq(
        selected_rows,
        CACHE_ROOT / "wifib/iq_window_2048_float32.npy",
        CACHE_ROOT / "wifib/power_window_2048_float32.npy",
    )
    token_cache = np.load(
        str(CACHE_ROOT / "tokens/wifib_vq_p1_k1024/tokens.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    reference_tokens = np.stack(
        [np.asarray(token_cache[int(row["cache_index"])], dtype=np.int64) for row in selected_rows]
    )
    codec = load_codec(ROOT / "runs/stage_g1/frozen/wifib_v1/vqvae_p1_k1024.pt")
    decoded_reference = decode_batches(codec, reference_tokens)
    victim_codec_logits = classify_logits(victim, decoded_reference)
    victim_codec_predictions = victim_codec_logits.argmax(dim=1)
    evaluator_codec_predictions = classify_batches(evaluator, decoded_reference)
    codec_margins = true_class_margin(victim_codec_logits, labels)

    graph = TokenCandidateGraph.load(
        ROOT / "artifacts/token_graph/wifib_vq_p1_k1024_neighbors.npz"
    )
    infill_checkpoint = torch.load(
        str(ROOT / "runs/next_stage/p3_infill_hybrid_v2/best.pt"),
        map_location="cpu",
        weights_only=False,
    )
    infill = build_infill_model(infill_checkpoint["model_config"])
    infill.load_state_dict(infill_checkpoint["model_state"])
    infill.eval()
    prior = BidirectionalTransitionPrior(
        graph.transition_counts,
        graph.arrays["usage_counts"],
        alpha=infill_checkpoint["config"]["training"]["transition_prior_alpha"],
    )
    with torch.inference_mode():
        decoded_codebook = codec.decode_code_indices(
            torch.arange(graph.codebook_size, dtype=torch.long)[:, None]
        )[:, :, 0]
    spread = graph.arrays["latent_distances"][:, : config["candidate_top_k"]].mean(axis=1)
    neighbor_values = decoded_codebook[
        torch.from_numpy(graph.arrays["neighbors"][:, : config["candidate_top_k"]]).long()
    ]
    sensitivity = torch.linalg.vector_norm(
        neighbor_values - decoded_codebook[:, None, :], dim=2
    ).mean(dim=1).numpy()
    maximum_positions = max(config["query_budgets"]) // config["candidates_per_position"]
    greedy_positions = normalized_position_prerank(
        reference_tokens,
        prior.log_forward.numpy(),
        spread,
        sensitivity,
        maximum_positions,
        surprisal_weight=config["position_preranking"]["surprisal_weight"],
        candidate_spread_weight=config["position_preranking"][
            "candidate_spread_weight"
        ],
        decoder_sensitivity_weight=config["position_preranking"][
            "decoder_sensitivity_weight"
        ],
    )
    random_positions = nested_positions(
        selected_rows,
        reference_tokens.shape[1],
        max(config["edit_counts"]),
        config["seed"],
    )
    proposal_common = dict(
        model=infill,
        graph=graph,
        reference_tokens=reference_tokens,
        decoded_reference=decoded_reference,
        decoded_codebook=decoded_codebook,
        device_labels=[label_map[value] for value in devices],
        context_length=infill_checkpoint["model_config"]["context_length"],
        delta_ll=config["delta_ll"],
        candidate_top_k=config["candidate_top_k"],
        transition_supported=config["transition_supported"],
        precheck_config=config["local_precheck"],
        transition_prior=prior,
        transition_prior_weight=infill_checkpoint["config"]["training"][
            "transition_prior_weight"
        ],
    )
    random_proposals = propose_candidates(positions=random_positions, **proposal_common)
    greedy_proposals = propose_candidates(positions=greedy_positions, **proposal_common)
    actions_by_sample, total_candidate_queries = query_single_edit_actions(
        victim,
        codec,
        reference_tokens,
        decoded_reference,
        greedy_proposals,
        labels,
        config["projection"],
        config["candidates_per_position"],
    )

    results = {}
    for method in (
        "random_uniform_plausible",
        "random_nearest",
        "random_infill",
    ):
        results[method] = {"q0": {}}
        for edit_count in config["edit_counts"]:
            edited, realized = edit_random_method(
                reference_tokens,
                random_proposals,
                edit_count,
                method,
                sample_ids,
                graph,
                config["seed"],
            )
            results[method]["q0"][str(edit_count)] = evaluate_variant(
                victim,
                evaluator,
                codec,
                edited,
                decoded_reference,
                labels,
                devices,
                victim_codec_predictions,
                evaluator_codec_predictions,
                codec_margins,
                realized,
                config["projection"],
                np.zeros(len(labels), dtype=np.int64),
                config["seed"] + edit_count,
                edit_count,
            )
    results["greedy_infill"] = {}
    for query_budget in config["query_budgets"]:
        key = "q%d" % query_budget
        results["greedy_infill"][key] = {}
        position_limit = query_budget // config["candidates_per_position"]
        for edit_count in config["edit_counts"]:
            edited, realized, query_counts = edit_greedy(
                reference_tokens,
                actions_by_sample,
                edit_count,
                position_limit,
                codec_margins.tolist(),
            )
            results["greedy_infill"][key][str(edit_count)] = evaluate_variant(
                victim,
                evaluator,
                codec,
                edited,
                decoded_reference,
                labels,
                devices,
                victim_codec_predictions,
                evaluator_codec_predictions,
                codec_margins,
                realized,
                config["projection"],
                query_counts,
                config["seed"] + query_budget + edit_count,
                edit_count,
            )

    primary_edit = str(config["selection"]["primary_edit_count"])
    method_primary = {}
    for method in config["methods"]:
        query_key = "q%d" % max(config["query_budgets"]) if method == "greedy_infill" else "q0"
        metrics = results[method][query_key][primary_edit]
        method_primary[method] = {
            "query_key": query_key,
            "valid_hard_rate": metrics["valid_hard_rate_pooled"],
            "evaluator_retention": metrics["evaluator_retention_on_codec_correct"],
            "rf_valid_fraction": metrics["rf_valid_fraction"],
        }
    eligible_methods = [
        method
        for method, metrics in method_primary.items()
        if metrics["evaluator_retention"]
        >= config["selection"]["minimum_b_retention"]
        and metrics["rf_valid_fraction"]
        >= config["selection"]["minimum_rf_valid_fraction"]
    ]
    ranked = sorted(
        eligible_methods,
        key=lambda method: (-method_primary[method]["valid_hard_rate"], method),
    )
    best_rate = method_primary[ranked[0]]["valid_hard_rate"] if ranked else 0.0
    within = [
        method
        for method in ranked
        if best_rate - method_primary[method]["valid_hard_rate"]
        <= config["selection"]["within_best_valid_hard_percentage_points"] / 100.0
    ]
    minimum = config["selection"]["minimum_methods_to_retain"]
    maximum = config["selection"]["maximum_methods_to_retain"]
    retained = list(dict.fromkeys((within + ranked[:minimum])[:maximum]))
    best_random = max(
        method_primary[method]["valid_hard_rate"]
        for method in method_primary
        if method.startswith("random_")
    )
    greedy_rate = method_primary["greedy_infill"]["valid_hard_rate"]
    report = {
        "schema_version": "wifib-p5a-screening-report-v1",
        "stage": "P5a_screening",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_role": config["source_role"],
        "source_count": len(selected_rows),
        "source_condition": "dual-clean-correct on raw clean; attack denominator additionally codec-correct",
        "policy_gate_used": False,
        "final_test_used": False,
        "victim_access": config["victim_access"],
        "evaluator_b_access": config["evaluator_b_access"],
        "eligibility_forward_passes_per_model": len(eligibility_audit),
        "candidate_search_queries_total": total_candidate_queries,
        "position_preranking_uses_victim": False,
        "results": results,
        "primary_screening": method_primary,
        "retained_methods": retained,
        "greedy_minus_best_random_percentage_points": 100.0
        * (greedy_rate - best_random),
        "gate": {
            "status": "PASS" if len(retained) >= minimum else "FAIL",
            "reason": (
                "At least three validity-qualified methods are retained for formal paired evaluation."
                if len(retained) >= minimum
                else "Fewer than three methods meet screening validity and ranking criteria."
            ),
        },
    }
    report_path = report_dir / "p5a_screening_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# P5a Development Screening",
        "",
        "- Gate: **%s**" % report["gate"]["status"],
        "- Sources: %d (10/device), dual-clean-correct development data" % len(selected_rows),
        "- Policy Gate / final test used: **false / false**",
        "- Candidate Victim queries: %d" % total_candidate_queries,
        "- Greedy minus best random at 8 edits: %.2f pp"
        % report["greedy_minus_best_random_percentage_points"],
        "- Retained methods: `%s`" % ", ".join(retained),
        "",
        "Primary 8-edit results:",
        "",
        "| method | query setting | valid-hard | B retention | RF valid |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in config["methods"]:
        value = method_primary[method]
        lines.append(
            "| %s | %s | %.3f | %.3f | %.3f |"
            % (
                method,
                value["query_key"],
                value["valid_hard_rate"],
                value["evaluator_retention"],
                value["rf_valid_fraction"],
            )
        )
    summary_path = report_dir / "p5a_screening_report.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    stage_report = {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": report["stage"],
        "git_commit": None,
        "config_hash": report["config_sha256"],
        "data_split_hash": split_config["role_assignment_sha256"],
        "checkpoint_hash": sha256_file(
            ROOT / "runs/next_stage/p3_infill_hybrid_v2/best.pt"
        ),
        "seeds": [config["seed"]],
        "source_count": len(selected_rows),
        "victim_query_budget": config["query_budgets"],
        "edit_budget": config["edit_counts"],
        "metrics": report,
        "gate": report["gate"],
        "artifacts": {
            "report_json": str(report_path),
            "report_markdown": str(summary_path),
            "source_pool": str(report_dir / "p5a_source_pool.csv"),
            "eligibility_audit": str(report_dir / "p5a_source_eligibility_audit.csv"),
        },
        "next_allowed_stage": "P5b_formal" if report["gate"]["status"] == "PASS" else None,
        "prohibited_actions": [
            "use P5a development rates as policy-Gate conclusions",
            "access final_test signal data",
            "train PPO",
        ],
    }
    (report_dir / "p5a_stage_report.json").write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "gate": report["gate"], "retained_methods": retained}, indent=2))


if __name__ == "__main__":
    main()
