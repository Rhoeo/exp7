"""Run the pre-registered P5b paired experiment on the untouched policy Gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rffi_core.attacks.query_search import normalized_position_prerank, true_class_margin
from rffi_core.attacks.run_p5a_screening import (
    classify_logits,
    edit_greedy,
    edit_random_method,
    evaluate_variant,
    load_victim,
    query_single_edit_actions,
    sha256_file,
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


ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = Path(r"E:\data_cache\rffi_v1")


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_bootstrap_difference(
    first_mask,
    second_masks,
    eligible,
    seed,
    iterations,
):
    second = np.mean(np.stack(second_masks, axis=0).astype(np.float64), axis=0)
    values = first_mask.astype(np.float64) - second
    indices = np.flatnonzero(eligible)
    rng = np.random.RandomState(seed)
    estimates = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draw = rng.choice(indices, size=len(indices), replace=True)
        estimates[index] = values[draw].mean()
    return {
        "difference_percentage_points": 100.0 * float(values[indices].mean()),
        "bootstrap_95ci_percentage_points": [
            100.0 * float(np.percentile(estimates, 2.5)),
            100.0 * float(np.percentile(estimates, 97.5)),
        ],
    }


def aggregate_seed_results(seed_results, masks, eligible):
    keys = (
        "realized_edit_count_mean",
        "search_queries_mean",
        "victim_margin_reduction_mean",
        "victim_fool_rate_on_codec_correct",
        "evaluator_retention_on_codec_correct",
        "rf_valid_fraction",
        "valid_hard_rate_pooled",
        "valid_hard_rate_macro",
    )
    output = {
        key: float(np.mean([result[key] for result in seed_results])) for key in keys
    }
    success_probability = np.mean(np.stack(masks, axis=0).astype(np.float64), axis=0)
    indices = np.flatnonzero(eligible)
    rng = np.random.RandomState(20260919)
    estimates = np.empty(5000, dtype=np.float64)
    for iteration in range(len(estimates)):
        draw = rng.choice(indices, size=len(indices), replace=True)
        estimates[iteration] = success_probability[draw].mean()
    output["valid_hard_bootstrap_95ci"] = [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]
    output["seeds"] = len(seed_results)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/attack/wifib_p5b_formal_v1.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_config = json.loads(
        (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
    )
    all_rows = load_index(CACHE_ROOT / "wifib/window_index.csv")
    row_by_id = {row["sample_id"]: row for row in all_rows}
    pool_rows = read_csv(resolve(config["source_pool"]))
    selected_rows = [row_by_id[row["sample_id"]] for row in pool_rows]
    if len(selected_rows) != 425:
        raise ValueError("formal source pool must contain the P1-balanced 425 records")
    counts = {}
    for row in selected_rows:
        counts[row["device_id"]] = counts.get(row["device_id"], 0) + 1
    if sorted(counts.values()) != [25] * 17:
        raise ValueError("formal source pool is not balanced at 25/device")

    label_map = build_label_map(row["device_id"] for row in all_rows)
    victim, victim_label_map = load_victim(
        ROOT / "runs/stage_g0/frozen/wifib_v1/victim_a.pt"
    )
    evaluator, evaluator_label_map = load_evaluator(
        ROOT / "runs/stage_g0/frozen/wifib_v1/evaluator_b.pt"
    )
    if label_map != victim_label_map or label_map != evaluator_label_map:
        raise ValueError("source and classifier label maps differ")
    sample_ids = [row["sample_id"] for row in selected_rows]
    devices = [row["device_id"] for row in selected_rows]
    labels = torch.tensor([label_map[value] for value in devices], dtype=torch.long)
    raw_clean = load_normalized_iq(
        selected_rows,
        CACHE_ROOT / "wifib/iq_window_2048_float32.npy",
        CACHE_ROOT / "wifib/power_window_2048_float32.npy",
    )
    raw_victim_predictions = classify_batches(victim, raw_clean)
    raw_evaluator_predictions = classify_batches(evaluator, raw_clean)
    if not bool(
        ((raw_victim_predictions == labels) & (raw_evaluator_predictions == labels)).all()
    ):
        raise ValueError("P1 policy source pool no longer satisfies dual-clean-correct")
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
    codec_joint = (
        (victim_codec_predictions == labels) & (evaluator_codec_predictions == labels)
    ).numpy()

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
        candidate_spread_weight=config["position_preranking"]["candidate_spread_weight"],
        decoder_sensitivity_weight=config["position_preranking"][
            "decoder_sensitivity_weight"
        ],
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
    greedy_proposals = propose_candidates(positions=greedy_positions, **proposal_common)
    actions_by_sample, candidate_queries_total = query_single_edit_actions(
        victim,
        codec,
        reference_tokens,
        decoded_reference,
        greedy_proposals,
        labels,
        config["projection"],
        config["candidates_per_position"],
    )

    detailed = {method: {} for method in config["methods"]}
    masks = {method: {} for method in config["methods"]}
    for method in ("random_nearest", "random_infill"):
        for seed in config["seeds"]:
            positions = nested_positions(
                selected_rows,
                reference_tokens.shape[1],
                max(config["edit_counts"]),
                seed,
            )
            proposals = propose_candidates(positions=positions, **proposal_common)
            for edit_count in config["edit_counts"]:
                edited, realized = edit_random_method(
                    reference_tokens,
                    proposals,
                    edit_count,
                    method,
                    sample_ids,
                    graph,
                    seed,
                )
                result, success_mask, eligible = evaluate_variant(
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
                    seed + edit_count,
                    edit_count,
                    return_masks=True,
                )
                key = str(edit_count)
                detailed[method].setdefault(key, []).append(
                    {"seed": seed, "metrics": result}
                )
                masks[method].setdefault(key, []).append(success_mask)

    detailed["greedy_infill"] = {}
    masks["greedy_infill"] = {}
    for query_budget in config["query_budgets"]:
        query_key = "q%d" % query_budget
        detailed["greedy_infill"][query_key] = {}
        masks["greedy_infill"][query_key] = {}
        position_limit = query_budget // config["candidates_per_position"]
        for edit_count in config["edit_counts"]:
            edited, realized, query_counts = edit_greedy(
                reference_tokens,
                actions_by_sample,
                edit_count,
                position_limit,
                codec_margins.tolist(),
            )
            result, success_mask, eligible = evaluate_variant(
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
                config["seeds"][0] + query_budget + edit_count,
                edit_count,
                return_masks=True,
            )
            detailed["greedy_infill"][query_key][str(edit_count)] = result
            masks["greedy_infill"][query_key][str(edit_count)] = success_mask

    aggregate = {"random_nearest": {}, "random_infill": {}, "greedy_infill": {}}
    for method in ("random_nearest", "random_infill"):
        for edit_count in config["edit_counts"]:
            key = str(edit_count)
            aggregate[method][key] = aggregate_seed_results(
                [entry["metrics"] for entry in detailed[method][key]],
                masks[method][key],
                codec_joint,
            )
    aggregate["greedy_infill"] = detailed["greedy_infill"]

    comparisons = {}
    for query_budget in config["query_budgets"]:
        query_key = "q%d" % query_budget
        comparisons[query_key] = {}
        for edit_count in config["edit_counts"]:
            key = str(edit_count)
            comparisons[query_key][key] = {}
            greedy_mask = masks["greedy_infill"][query_key][key]
            for random_method in ("random_nearest", "random_infill"):
                comparisons[query_key][key]["greedy_minus_" + random_method] = (
                    paired_bootstrap_difference(
                        greedy_mask,
                        masks[random_method][key],
                        codec_joint,
                        config["seeds"][0] + query_budget + edit_count,
                        config["bootstrap_iterations"],
                    )
                )

    validity_ok = True
    for query_key, edits in detailed["greedy_infill"].items():
        for result in edits.values():
            validity_ok &= result["rf_valid_fraction"] >= 0.95
            validity_ok &= result["evaluator_retention_on_codec_correct"] >= 0.98
    report = {
        "schema_version": "wifib-p5b-formal-report-v1",
        "stage": "P5b_formal_paired",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_count": len(selected_rows),
        "source_per_device": 25,
        "source_condition": config["source_condition"],
        "codec_joint_correct_count": int(codec_joint.sum()),
        "policy_gate_used": True,
        "policy_gate_used_for_method_or_threshold_selection": False,
        "final_test_used": False,
        "candidate_search_queries_total": candidate_queries_total,
        "detailed": detailed,
        "aggregate": aggregate,
        "paired_comparisons": comparisons,
        "gate": {
            "status": "PASS" if validity_ok else "FAIL",
            "reason": (
                "Pre-registered policy-Gate experiment completed with B/RF validity retained."
                if validity_ok
                else "At least one formal greedy condition violates B/RF validity."
            ),
        },
    }
    report_dir = ROOT / "reports/next_stage"
    report_path = report_dir / "p5b_formal_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# P5b Formal Paired Policy-Gate Experiment",
        "",
        "- Gate: **%s**" % report["gate"]["status"],
        "- Sources: 425 (25/device), fixed untouched-buffer policy Gate",
        "- Codec-joint-correct denominator: %d" % int(codec_joint.sum()),
        "- Final test used: **false**",
        "- Candidate Victim queries: %d" % candidate_queries_total,
        "",
        "| method | Q | edits | valid-hard | 95% CI | B retention | RF valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("random_nearest", "random_infill"):
        for edit_count in config["edit_counts"]:
            value = aggregate[method][str(edit_count)]
            lines.append(
                "| %s | 0 | %d | %.3f | [%.3f, %.3f] | %.3f | %.3f |"
                % (
                    method,
                    edit_count,
                    value["valid_hard_rate_pooled"],
                    value["valid_hard_bootstrap_95ci"][0],
                    value["valid_hard_bootstrap_95ci"][1],
                    value["evaluator_retention_on_codec_correct"],
                    value["rf_valid_fraction"],
                )
            )
    for query_budget in config["query_budgets"]:
        for edit_count in config["edit_counts"]:
            value = detailed["greedy_infill"]["q%d" % query_budget][str(edit_count)]
            lines.append(
                "| greedy_infill | %d | %d | %.3f | [%.3f, %.3f] | %.3f | %.3f |"
                % (
                    query_budget,
                    edit_count,
                    value["valid_hard_rate_pooled"],
                    value["valid_hard_bootstrap_95ci"][0],
                    value["valid_hard_bootstrap_95ci"][1],
                    value["evaluator_retention_on_codec_correct"],
                    value["rf_valid_fraction"],
                )
            )
    summary_path = report_dir / "p5b_formal_report.md"
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
        "seeds": config["seeds"],
        "source_count": len(selected_rows),
        "victim_query_budget": config["query_budgets"],
        "edit_budget": config["edit_counts"],
        "metrics": report,
        "gate": report["gate"],
        "artifacts": {
            "report_json": str(report_path),
            "report_markdown": str(summary_path),
            "source_pool": str(resolve(config["source_pool"])),
        },
        "next_allowed_stage": "P6_query_search_gate" if validity_ok else None,
        "prohibited_actions": [
            "change methods or thresholds based on policy-Gate outcomes",
            "access final_test signal data",
            "train PPO",
        ],
    }
    (report_dir / "p5b_stage_report.json").write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
