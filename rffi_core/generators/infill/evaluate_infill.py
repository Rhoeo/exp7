"""Evaluate P3 candidate probability and diversity without classifier queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rffi_core.generators.infill.data import MaskedTokenWindowDataset
from rffi_core.generators.infill.context_prior import BidirectionalTransitionPrior
from rffi_core.generators.infill.models import build_infill_model, masked_cross_entropy


ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / total
    return float(-np.sum(probabilities * np.log(probabilities)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="runs/next_stage/p3_infill_hybrid_v2/best.pt"
    )
    parser.add_argument(
        "--config", default="configs/generator/wifib_masked_infill_v1.yaml"
    )
    args = parser.parse_args()
    checkpoint_path = resolve(args.checkpoint)
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_infill_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    data = config["data"]
    dataset = MaskedTokenWindowDataset(
        token_cache_path=resolve(data["token_cache"]),
        window_index_path=resolve(data["window_index"]),
        split_config_path=resolve(data["split_config"]),
        role=data["validation_role"],
        candidate_graph_path=resolve(data["candidate_graph"]),
        context_length=config["model"]["context_length"],
        max_samples=config["training"]["max_validation_samples"],
        seed=config["seed"],
        neighbor_corruption_probability=config["training"][
            "neighbor_corruption_probability"
        ],
        transition_supported=config["training"]["transition_supported"],
        candidate_top_k=config["training"]["candidate_top_k"],
    )
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0
    )
    prior = BidirectionalTransitionPrior(
        dataset.graph.transition_counts,
        dataset.graph.arrays["usage_counts"],
        alpha=config["training"]["transition_prior_alpha"],
    )
    prior_weight = float(config["training"]["transition_prior_weight"])
    deltas = [float(value) for value in config["candidate_probability"]["delta_ll_grid"]]
    coverage_counts = {value: [] for value in deltas}
    proposal_counts = {value: Counter() for value in deltas}
    total_loss = 0.0
    total_targets = 0
    top1 = 0
    top5 = 0
    prior_candidate_counts = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["tokens"],
                batch["device_label"],
                batch["global_positions"],
                batch["target_indicator"],
            )
            targets = batch["targets"]
            mask_indices = torch.nonzero(targets != -100, as_tuple=False)
            count = int(mask_indices.shape[0])
            masked_logits = logits[targets != -100]
            masked_targets = targets[targets != -100]
            prior_logits = prior.scores(batch["tokens"], batch["target_indicator"])
            masked_logits = masked_logits + prior_weight * prior_logits
            loss = torch.nn.functional.cross_entropy(masked_logits, masked_targets)
            total_loss += float(loss) * count
            total_targets += count
            top1 += int((masked_logits.argmax(dim=1) == masked_targets).sum())
            top5 += int(
                (masked_logits.topk(5, dim=1).indices == masked_targets[:, None])
                .any(dim=1)
                .sum()
            )
            fused_log_prob = torch.log_softmax(masked_logits, dim=-1)
            for masked_index, (batch_index, position) in enumerate(mask_indices.tolist()):
                original = int(masked_targets[masked_index])
                candidates = dataset.graph.candidates(
                    original,
                    top_k=config["training"]["candidate_top_k"],
                    transition_supported=config["training"]["transition_supported"],
                )
                prior_candidate_counts.append(len(candidates))
                original_log_prob = float(fused_log_prob[masked_index, original])
                for delta in deltas:
                    eligible = [
                        int(candidate)
                        for candidate in candidates
                        if float(fused_log_prob[masked_index, int(candidate)])
                        >= original_log_prob - delta
                    ]
                    coverage_counts[delta].append(len(eligible))
                    if eligible:
                        chosen = max(
                            eligible,
                            key=lambda candidate: float(
                                fused_log_prob[masked_index, candidate]
                            ),
                        )
                        proposal_counts[delta][chosen] += 1
    threshold_metrics = {}
    selected_delta = None
    target_coverage = float(
        config["candidate_probability"]["target_at_least_one_coverage"]
    )
    for delta in deltas:
        counts = np.asarray(coverage_counts[delta], dtype=np.int64)
        coverage = float(np.mean(counts >= 1)) if len(counts) else 0.0
        threshold_metrics[str(delta)] = {
            "at_least_one_coverage": coverage,
            "at_least_four_coverage": float(np.mean(counts >= 4)) if len(counts) else 0.0,
            "mean_candidates": float(np.mean(counts)) if len(counts) else 0.0,
            "median_candidates": float(np.median(counts)) if len(counts) else 0.0,
            "no_op_rate": float(np.mean(counts == 0)) if len(counts) else 1.0,
            "replacement_token_entropy_nats": entropy_from_counts(proposal_counts[delta]),
            "dominant_replacement_fraction": (
                max(proposal_counts[delta].values()) / sum(proposal_counts[delta].values())
                if proposal_counts[delta]
                else 0.0
            ),
        }
        if selected_delta is None and coverage >= target_coverage:
            selected_delta = delta
    finite = math.isfinite(total_loss)
    smoke_passed = finite and total_targets > 0 and np.mean(prior_candidate_counts) > 0
    report = {
        "schema_version": "wifib-infill-validation-v1",
        "stage": "P3_masked_infill_smoke",
        "run_mode": config["run_mode"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "validation_role": data["validation_role"],
        "validation_examples": len(dataset),
        "masked_targets": total_targets,
        "loss": total_loss / max(1, total_targets),
        "top1_accuracy": top1 / max(1, total_targets),
        "top5_accuracy": top5 / max(1, total_targets),
        "mean_prior_candidates": float(np.mean(prior_candidate_counts)),
        "delta_ll_curves": threshold_metrics,
        "selected_delta_ll_smoke": selected_delta,
        "formal_threshold_locked": False,
        "victim_queries": 0,
        "evaluator_queries": 0,
        "policy_gate_attack_feedback": False,
        "final_test_used": False,
        "gate": {
            "status": "SMOKE_PASS" if smoke_passed else "FAIL",
            "reason": (
                "Training/evaluation is finite and the static prior yields alternatives; formal Infill Gate awaits P4."
                if smoke_passed
                else "Masked-infill smoke integrity checks failed."
            ),
        },
    }
    report_dir = ROOT / "reports/next_stage"
    report_path = report_dir / "infill_validation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selected_metrics = threshold_metrics.get(str(selected_delta), {})
    summary = [
        "# P3 Masked-Infill Smoke Validation",
        "",
        "- Gate: **%s**" % report["gate"]["status"],
        "- Validation examples / masked targets: %d / %d" % (len(dataset), total_targets),
        "- Loss: %.4f" % report["loss"],
        "- Original-token top-1 / top-5: %.3f / %.3f"
        % (report["top1_accuracy"], report["top5_accuracy"]),
        "- Mean transition-supported latent candidates before probability filter: %.2f"
        % report["mean_prior_candidates"],
        "- Provisional smoke delta_ll: `%s`" % selected_delta,
        "- Coverage/no-op at provisional threshold: %.3f / %.3f"
        % (
            selected_metrics.get("at_least_one_coverage", 0.0),
            selected_metrics.get("no_op_rate", 1.0),
        ),
        "",
        "No Victim A, Evaluator B, policy-Gate attack outcome, or final-test feedback was used. This is a CPU smoke result, not the formal Infill Gate; the relative-probability threshold remains provisional until P4 waveform checks are available.",
    ]
    summary_path = report_dir / "infill_validation.md"
    summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    stage_report = {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": "P3_masked_infill_smoke",
        "git_commit": None,
        "config_hash": report["config_sha256"],
        "data_split_hash": json.loads(
            (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(encoding="utf-8")
        )["role_assignment_sha256"],
        "checkpoint_hash": report["checkpoint_sha256"],
        "seeds": [config["seed"]],
        "source_count": len(dataset),
        "victim_query_budget": 0,
        "edit_budget": [1, 2, 4],
        "metrics": report,
        "gate": report["gate"],
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "validation_json": str(report_path),
            "validation_markdown": str(summary_path),
        },
        "next_allowed_stage": "P4_waveform_validity" if smoke_passed else None,
        "prohibited_actions": [
            "claim formal Infill Gate from CPU smoke",
            "use policy_gate attack outcomes for threshold selection",
            "access final_test signal data",
            "train PPO",
        ],
    }
    (report_dir / "p3_stage_report.json").write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"report": str(report_path), "gate": report["gate"]}, indent=2))
    if not smoke_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
