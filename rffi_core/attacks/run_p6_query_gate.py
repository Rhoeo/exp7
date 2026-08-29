"""Run the fixed P6 beam-coordinate versus query-matched random Gate Q."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from rffi_core.attacks.common_projection_pipeline import common_projection_pipeline
from rffi_core.attacks.query_search import (
    apply_token_edits,
    interleaved_candidate_actions,
    keep_best_beam_states,
    normalized_position_prerank,
    select_unique_position_actions,
    true_class_margin,
)
from rffi_core.attacks.run_p5a_screening import (
    classify_logits,
    evaluate_variant,
    load_victim,
    sha256_file,
    stable_integer,
)
from rffi_core.attacks.run_p5b_formal import (
    aggregate_seed_results,
    paired_bootstrap_difference,
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


def normalized_edits(edits):
    return tuple(sorted((int(position), int(candidate)) for position, candidate in edits))


def score_edit_records(
    records,
    margin_cache,
    victim,
    codec,
    reference_tokens,
    decoded_reference,
    labels,
    projection_config,
    chunk_size=256,
):
    """Score projected candidate states and cache identical sample/edit pairs."""
    uncached = []
    for record in records:
        record["edits"] = normalized_edits(record["edits"])
        key = (int(record["sample_index"]), record["edits"])
        if key in margin_cache:
            record["margin"] = float(margin_cache[key])
        else:
            uncached.append(record)
    model_calls = 0
    for start in range(0, len(uncached), int(chunk_size)):
        batch_records = uncached[start : start + int(chunk_size)]
        sample_indices = [int(record["sample_index"]) for record in batch_records]
        token_batch = apply_token_edits(
            reference_tokens,
            sample_indices,
            [record["edits"] for record in batch_records],
        )
        decoded = decode_batches(codec, token_batch)
        reference = decoded_reference[sample_indices]
        projected, _ = common_projection_pipeline(
            reference, decoded, **projection_config
        )
        logits = classify_logits(victim, projected, batch_size=64)
        margins = true_class_margin(logits, labels[sample_indices]).tolist()
        for record, margin in zip(batch_records, margins):
            key = (int(record["sample_index"]), record["edits"])
            margin_cache[key] = float(margin)
            record["margin"] = float(margin)
        model_calls += len(batch_records)
    return {
        "logical_requests": len(records),
        "unique_model_calls": model_calls,
        "cache_hits": len(records) - model_calls,
    }


def build_action_pools(proposals, candidates_per_position):
    return [
        interleaved_candidate_actions(sample, candidates_per_position)
        for sample in proposals
    ]


def sample_random_edit_states(
    action_pool,
    edit_count,
    maximum_states,
    seed,
    sample_id,
    attempt_multiplier,
):
    """Sample unique plausible edit sets without using Victim outputs."""
    by_position = defaultdict(list)
    for action in action_pool:
        by_position[int(action["position"])].append(int(action["candidate"]))
    positions = sorted(by_position)
    if not positions:
        return []
    realized_count = min(int(edit_count), len(positions))
    rng = np.random.RandomState(stable_integer(seed, "p6_random_states", sample_id))
    states = []
    seen = set()
    maximum_attempts = max(
        int(maximum_states), int(maximum_states) * int(attempt_multiplier)
    )
    for _ in range(maximum_attempts):
        chosen_positions = rng.choice(
            positions, size=realized_count, replace=False
        ).tolist()
        edits = []
        for position in chosen_positions:
            candidates = by_position[int(position)]
            candidate = candidates[int(rng.randint(0, len(candidates)))]
            edits.append((int(position), int(candidate)))
        key = normalized_edits(edits)
        if key in seen:
            continue
        seen.add(key)
        states.append(key)
        if len(states) == int(maximum_states):
            break
    return states


def choose_best_states(reference_tokens, baseline_margins, states_by_sample):
    edited = np.asarray(reference_tokens).copy()
    realized = np.zeros(len(reference_tokens), dtype=np.int64)
    for sample_index, states in enumerate(states_by_sample):
        best = {"edits": (), "margin": float(baseline_margins[sample_index])}
        for state in states:
            if float(state["margin"]) < float(best["margin"]):
                best = state
        for position, candidate in best["edits"]:
            edited[sample_index, int(position)] = int(candidate)
        realized[sample_index] = len(best["edits"])
    return edited, realized


def run_query_matched_random(
    seed,
    budgets,
    edit_count,
    sample_ids,
    action_pools,
    baseline_margins,
    margin_cache,
    score_common,
    attempt_multiplier,
):
    maximum_budget = max(int(value) for value in budgets)
    sampled = [
        sample_random_edit_states(
            action_pools[index],
            edit_count,
            maximum_budget,
            seed,
            sample_ids[index],
            attempt_multiplier,
        )
        for index in range(len(sample_ids))
    ]
    records = []
    for sample_index, states in enumerate(sampled):
        for order, edits in enumerate(states):
            records.append(
                {
                    "sample_index": sample_index,
                    "order": order,
                    "edits": edits,
                }
            )
    accounting = score_edit_records(records, margin_cache, **score_common)
    scored_by_sample = [[] for _ in sample_ids]
    for record in records:
        scored_by_sample[int(record["sample_index"])].append(record)
    for values in scored_by_sample:
        values.sort(key=lambda value: int(value["order"]))
    snapshots = {}
    for budget in budgets:
        selected = [values[: int(budget)] for values in scored_by_sample]
        edited, realized = choose_best_states(
            score_common["reference_tokens"], baseline_margins, selected
        )
        query_counts = np.asarray([len(values) for values in selected], dtype=np.int64)
        snapshots[int(budget)] = {
            "edited_tokens": edited,
            "realized": realized,
            "query_counts": query_counts,
        }
    accounting["candidate_states_generated"] = len(records)
    accounting["sources_exhausted_before_max_budget"] = int(
        sum(len(values) < maximum_budget for values in scored_by_sample)
    )
    return snapshots, accounting


def run_greedy_coordinate(
    budgets,
    edit_count,
    action_pools,
    baseline_margins,
    margin_cache,
    score_common,
    candidates_per_position,
):
    """Score single-coordinate actions, then select unique improving edits."""
    records = []
    for sample_index, actions in enumerate(action_pools):
        for action in actions:
            records.append(
                {
                    "sample_index": sample_index,
                    "position_rank": int(action["position_rank"]),
                    "position": int(action["position"]),
                    "candidate": int(action["candidate"]),
                    "edits": ((int(action["position"]), int(action["candidate"])),),
                }
            )
    accounting = score_edit_records(records, margin_cache, **score_common)
    by_sample = defaultdict(list)
    for record in records:
        by_sample[int(record["sample_index"])].append(record)
    snapshots = {}
    for budget in budgets:
        position_limit = int(budget) // int(candidates_per_position)
        edited = np.asarray(score_common["reference_tokens"]).copy()
        realized = np.zeros(len(action_pools), dtype=np.int64)
        query_counts = np.zeros(len(action_pools), dtype=np.int64)
        for sample_index, actions in by_sample.items():
            eligible_actions = [
                action
                for action in actions
                if int(action["position_rank"]) < position_limit
            ]
            query_counts[sample_index] = len(eligible_actions)
            selected = select_unique_position_actions(
                eligible_actions,
                edit_count,
                baseline_margins[sample_index],
            )
            for action in selected:
                edited[sample_index, int(action["position"])] = int(
                    action["candidate"]
                )
            realized[sample_index] = len(selected)
        snapshots[int(budget)] = {
            "edited_tokens": edited,
            "realized": realized,
            "query_counts": query_counts,
        }
    accounting["candidate_states_generated"] = len(records)
    return snapshots, accounting


def coordinate_neighbors(state, action_pool, edit_count, branch_per_state, seen, proposed):
    """Generate fixed-cardinality one-coordinate neighbors.

    A coordinate edit either changes the replacement at an already selected
    position or swaps one selected position for an unselected position.  Every
    returned state therefore has exactly ``edit_count`` edits.
    """
    current = normalized_edits(state["edits"])
    current_positions = [int(position) for position, _ in current]
    by_position = defaultdict(list)
    for action in action_pool:
        by_position[int(action["position"])].append(int(action["candidate"]))
    current_candidates = {position: int(candidate) for position, candidate in current}
    neighbors = []

    def add(edits):
        key = normalized_edits(edits)
        if key in seen or key in proposed:
            return False
        proposed.add(key)
        neighbors.append(key)
        return len(neighbors) >= int(branch_per_state)

    # First visit alternative replacements, interleaving positions so the
    # branch is not dominated by one coordinate.
    max_replacements = max(
        [len(by_position[position]) for position in current_positions] + [1]
    )
    for candidate_rank in range(1, max_replacements):
        for position in current_positions:
            candidates = by_position[position]
            if candidate_rank >= len(candidates):
                continue
            edits = [(p, c) for p, c in current if p != position]
            edits.append((position, candidates[candidate_rank]))
            if add(edits):
                return neighbors

    # Then visit position swaps, also interleaved across the current
    # coordinates.  The replacement is drawn from the same precomputed pool.
    unused_actions = [
        action
        for action in action_pool
        if int(action["position"]) not in set(current_positions)
    ]
    for action_index, action in enumerate(unused_actions):
        replacement_position = current_positions[action_index % len(current_positions)]
        edits = [
            (p, c) for p, c in current if p != replacement_position
        ]
        edits.append((int(action["position"]), int(action["candidate"])))
        if add(edits):
            return neighbors
    return neighbors


def run_beam_coordinate(
    budgets,
    edit_count,
    action_pools,
    baseline_margins,
    margin_cache,
    score_common,
    beam_width,
    branch_per_state,
    maximum_rounds,
):
    sample_count = len(action_pools)
    initial_states = []
    initial_records = []
    for sample_index, action_pool in enumerate(action_pools):
        first_by_position = {}
        for action in action_pool:
            first_by_position.setdefault(
                int(action["position"]), int(action["candidate"])
            )
        if len(first_by_position) < int(edit_count):
            raise RuntimeError("insufficient positions for fixed-cardinality beam")
        initial_edits = tuple(
            (position, first_by_position[position])
            for position in sorted(first_by_position)[: int(edit_count)]
        )
        initial_states.append(initial_edits)
        initial_records.append({"sample_index": sample_index, "edits": initial_edits})
    initial_accounting = score_edit_records(initial_records, margin_cache, **score_common)
    beams = [
        [{"edits": initial_states[index], "margin": None}]
        for index in range(sample_count)
    ]
    for sample_index, state in enumerate(beams):
        state[0]["margin"] = float(
            margin_cache[(sample_index, normalized_edits(state[0]["edits"]))]
        )
    seen = [{normalized_edits(initial_states[index])} for index in range(sample_count)]
    query_counts = np.ones(sample_count, dtype=np.int64)
    accounting = Counter()
    accounting.update(initial_accounting)
    snapshots = {}
    rounds_completed = 0
    for target_budget in sorted(int(value) for value in budgets):
        while rounds_completed < int(maximum_rounds):
            round_records = []
            by_sample = defaultdict(list)
            for sample_index in range(sample_count):
                remaining = target_budget - int(query_counts[sample_index])
                if remaining <= 0:
                    continue
                proposed_this_round = set()
                for state in beams[sample_index]:
                    state_neighbors = coordinate_neighbors(
                        state,
                        action_pools[sample_index],
                        edit_count,
                        branch_per_state,
                        seen[sample_index],
                        proposed_this_round,
                    )
                    for edits in state_neighbors:
                        record = {"sample_index": sample_index, "edits": edits}
                        round_records.append(record)
                        by_sample[sample_index].append(record)
                        if len(by_sample[sample_index]) == remaining:
                            break
                    if len(by_sample[sample_index]) == remaining:
                        break
                if len(by_sample[sample_index]) > remaining:
                    del by_sample[sample_index][remaining:]
            if not round_records:
                break
            allowed_ids = {id(record) for values in by_sample.values() for record in values}
            round_records = [record for record in round_records if id(record) in allowed_ids]
            scored = score_edit_records(
                round_records, margin_cache, **score_common
            )
            accounting.update(scored)
            for sample_index, new_states in by_sample.items():
                for state in new_states:
                    seen[sample_index].add(state["edits"])
                query_counts[sample_index] += len(new_states)
                beams[sample_index] = keep_best_beam_states(
                    beams[sample_index] + new_states, beam_width
                )
            rounds_completed += 1
            if bool(np.all(query_counts >= target_budget)):
                break
        edited, realized = choose_best_states(
            score_common["reference_tokens"], baseline_margins, beams
        )
        snapshots[target_budget] = {
            "edited_tokens": edited,
            "realized": realized,
            "query_counts": query_counts.copy(),
            "rounds_completed": rounds_completed,
        }
    accounting["maximum_rounds_completed"] = rounds_completed
    accounting["sources_exhausted_before_max_budget"] = int(
        np.sum(query_counts < max(int(value) for value in budgets))
    )
    return snapshots, dict(accounting)


def minimum_device_retention(result):
    return min(value["evaluator_retention"] for value in result["per_device"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/attack/wifib_p6_query_gate_v1.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    split_config = json.loads(
        (ROOT / "configs/data/wifib_next_stage_splits.json").read_text(
            encoding="utf-8"
        )
    )

    all_rows = load_index(CACHE_ROOT / "wifib/window_index.csv")
    row_by_id = {row["sample_id"]: row for row in all_rows}
    selected_rows = [
        row_by_id[row["sample_id"]] for row in read_csv(resolve(config["source_pool"]))
    ]
    counts = Counter(row["device_id"] for row in selected_rows)
    if len(selected_rows) != 425 or sorted(counts.values()) != [25] * 17:
        raise ValueError("P6 requires the frozen balanced 425-record policy Gate")

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
    raw_joint = (classify_batches(victim, raw_clean) == labels) & (
        classify_batches(evaluator, raw_clean) == labels
    )
    if not bool(raw_joint.all()):
        raise ValueError("frozen policy Gate no longer satisfies dual-clean-correct")

    token_cache = np.load(
        str(CACHE_ROOT / "tokens/wifib_vq_p1_k1024/tokens.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    reference_tokens = np.stack(
        [
            np.asarray(token_cache[int(row["cache_index"])], dtype=np.int64)
            for row in selected_rows
        ]
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
    infill_checkpoint_path = ROOT / "runs/next_stage/p3_infill_hybrid_v2/best.pt"
    infill_checkpoint = torch.load(
        str(infill_checkpoint_path), map_location="cpu", weights_only=False
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
    spread = graph.arrays["latent_distances"][:, : config["candidate_top_k"]].mean(
        axis=1
    )
    neighbor_values = decoded_codebook[
        torch.from_numpy(
            graph.arrays["neighbors"][:, : config["candidate_top_k"]]
        ).long()
    ]
    sensitivity = torch.linalg.vector_norm(
        neighbor_values - decoded_codebook[:, None, :], dim=2
    ).mean(dim=1).numpy()
    positions = normalized_position_prerank(
        reference_tokens,
        prior.log_forward.numpy(),
        spread,
        sensitivity,
        int(config["maximum_preranked_positions"]),
        surprisal_weight=config["position_preranking"]["surprisal_weight"],
        candidate_spread_weight=config["position_preranking"][
            "candidate_spread_weight"
        ],
        decoder_sensitivity_weight=config["position_preranking"][
            "decoder_sensitivity_weight"
        ],
    )
    proposals = propose_candidates(
        model=infill,
        graph=graph,
        reference_tokens=reference_tokens,
        decoded_reference=decoded_reference,
        decoded_codebook=decoded_codebook,
        positions=positions,
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
    action_pools = build_action_pools(
        proposals, config["candidate_actions_per_position"]
    )
    eligible_position_counts = np.asarray(
        [sum(bool(proposal["eligible"]) for proposal in sample) for sample in proposals]
    )
    if int(np.min(eligible_position_counts)) < config["edit_count"]:
        raise RuntimeError("at least one source has fewer eligible positions than edit budget")

    margin_cache = {}
    score_common = {
        "victim": victim,
        "codec": codec,
        "reference_tokens": reference_tokens,
        "decoded_reference": decoded_reference,
        "labels": labels,
        "projection_config": config["projection"],
    }
    random_snapshots = {}
    random_accounting = {}
    for seed in config["random_search_seeds"]:
        snapshots, accounting = run_query_matched_random(
            seed=seed,
            budgets=config["query_budgets"],
            edit_count=config["edit_count"],
            sample_ids=sample_ids,
            action_pools=action_pools,
            baseline_margins=codec_margins.tolist(),
            margin_cache=margin_cache,
            score_common=score_common,
            attempt_multiplier=config["query_matched_random"][
                "maximum_sampling_attempt_multiplier"
            ],
        )
        random_snapshots[int(seed)] = snapshots
        random_accounting[str(seed)] = accounting

    # The P5b Greedy method is pre-registered, but it must be compared to the
    # query-matched random baseline under the same P6 accounting before it can
    # be considered for Gate Q.
    greedy_snapshots, greedy_accounting = run_greedy_coordinate(
        budgets=config["query_budgets"],
        edit_count=config["edit_count"],
        action_pools=action_pools,
        baseline_margins=codec_margins.tolist(),
        margin_cache=margin_cache,
        score_common=score_common,
        candidates_per_position=config["candidate_actions_per_position"],
    )

    beam_snapshots, beam_accounting = run_beam_coordinate(
        budgets=config["query_budgets"],
        edit_count=config["edit_count"],
        action_pools=action_pools,
        baseline_margins=codec_margins.tolist(),
        margin_cache=margin_cache,
        score_common=score_common,
        beam_width=config["beam"]["beam_width"],
        branch_per_state=config["beam"]["branch_per_state"],
        maximum_rounds=config["beam"]["maximum_rounds"],
    )

    random_detailed = {}
    random_masks = {}
    random_aggregate = {}
    beam_detailed = {}
    beam_masks = {}
    greedy_detailed = {}
    greedy_masks = {}
    comparisons = {}
    for budget in config["query_budgets"]:
        key = "q%d" % int(budget)
        random_detailed[key] = []
        random_masks[key] = []
        for seed in config["random_search_seeds"]:
            snapshot = random_snapshots[int(seed)][int(budget)]
            result, success_mask, eligible = evaluate_variant(
                victim,
                evaluator,
                codec,
                snapshot["edited_tokens"],
                decoded_reference,
                labels,
                devices,
                victim_codec_predictions,
                evaluator_codec_predictions,
                codec_margins,
                snapshot["realized"],
                config["projection"],
                snapshot["query_counts"],
                int(seed) + int(budget),
                config["edit_count"],
                return_masks=True,
            )
            random_detailed[key].append({"seed": int(seed), "metrics": result})
            random_masks[key].append(success_mask)
        random_aggregate[key] = aggregate_seed_results(
            [entry["metrics"] for entry in random_detailed[key]],
            random_masks[key],
            codec_joint,
        )

        snapshot = beam_snapshots[int(budget)]
        result, success_mask, eligible = evaluate_variant(
            victim,
            evaluator,
            codec,
            snapshot["edited_tokens"],
            decoded_reference,
            labels,
            devices,
            victim_codec_predictions,
            evaluator_codec_predictions,
            codec_margins,
            snapshot["realized"],
            config["projection"],
            snapshot["query_counts"],
            config["seed"] + int(budget),
            config["edit_count"],
            return_masks=True,
        )
        result["beam_rounds_completed"] = snapshot["rounds_completed"]
        beam_detailed[key] = result
        beam_masks[key] = success_mask
        beam_comparison = paired_bootstrap_difference(
            success_mask,
            random_masks[key],
            codec_joint,
            config["seed"] + int(budget),
            5000,
        )
        lower = beam_comparison["bootstrap_95ci_percentage_points"][0]
        beam_comparison["gate_q1_pass"] = bool(
            beam_comparison["difference_percentage_points"]
            >= config["gate_q1_minimum_percentage_points"]
            and lower > 0.0
        )
        greedy_snapshot = greedy_snapshots[int(budget)]
        greedy_result, greedy_success_mask, _ = evaluate_variant(
            victim,
            evaluator,
            codec,
            greedy_snapshot["edited_tokens"],
            decoded_reference,
            labels,
            devices,
            victim_codec_predictions,
            evaluator_codec_predictions,
            codec_margins,
            greedy_snapshot["realized"],
            config["projection"],
            greedy_snapshot["query_counts"],
            config["seed"] + 1000 + int(budget),
            config["edit_count"],
            return_masks=True,
        )
        greedy_detailed[key] = greedy_result
        greedy_masks[key] = greedy_success_mask
        greedy_comparison = paired_bootstrap_difference(
            greedy_success_mask,
            random_masks[key],
            codec_joint,
            config["seed"] + 1000 + int(budget),
            5000,
        )
        greedy_lower = greedy_comparison["bootstrap_95ci_percentage_points"][0]
        greedy_comparison["gate_q1_pass"] = bool(
            greedy_comparison["difference_percentage_points"]
            >= config["gate_q1_minimum_percentage_points"]
            and greedy_lower > 0.0
        )
        comparisons[key] = {
            "beam_minus_random": beam_comparison,
            "greedy_minus_random": greedy_comparison,
        }

    q2_evidence = []
    for random_budget in config["query_budgets"]:
        random_key = "q%d" % int(random_budget)
        target_rate = random_aggregate[random_key]["valid_hard_rate_pooled"]
        for search_budget in config["query_budgets"]:
            if int(search_budget) > config["gate_q2_maximum_query_fraction"] * int(
                random_budget
            ):
                continue
            search_key = "q%d" % int(search_budget)
            for searcher_name, search_result in (
                ("beam_coordinate", beam_detailed[search_key]),
                ("greedy_coordinate", greedy_detailed[search_key]),
            ):
                if search_result["valid_hard_rate_pooled"] >= target_rate:
                    q2_evidence.append(
                        {
                            "random_budget": int(random_budget),
                            "random_rate": target_rate,
                            "searcher": searcher_name,
                            "searcher_budget": int(search_budget),
                            "searcher_rate": search_result[
                                "valid_hard_rate_pooled"
                            ],
                        }
                    )
                    break
            if q2_evidence and q2_evidence[-1]["random_budget"] == int(random_budget):
                break
    q1_pass = any(
        comparison[method]["gate_q1_pass"]
        for comparison in comparisons.values()
        for method in ("beam_minus_random", "greedy_minus_random")
    )
    q2_pass = bool(q2_evidence)
    gate_pass = q1_pass or q2_pass

    p5b_path = ROOT / "reports/next_stage/p5b_formal_report.json"
    p5b = json.loads(p5b_path.read_text(encoding="utf-8"))
    auxiliary_p5b = {
        "role": "auxiliary only; no-query random is not used for Gate Q",
        "edit_count": config["edit_count"],
        "greedy": {
            key: p5b["aggregate"]["greedy_infill"][key][str(config["edit_count"])]
            for key in ("q64", "q128", "q256")
        },
        "no_query_random": {
            method: p5b["aggregate"][method][str(config["edit_count"])]
            for method in ("random_nearest", "random_infill")
        },
    }
    report = {
        "schema_version": "wifib-p6-query-gate-report-v2",
        "stage": "P6_query_search_gate_v2",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_count": len(selected_rows),
        "source_per_device": 25,
        "source_condition": "conditional on frozen untouched-buffer dual-clean-correct policy Gate",
        "codec_joint_correct_count": int(codec_joint.sum()),
        "edit_count": int(config["edit_count"]),
        "query_budgets": config["query_budgets"],
        "query_budget_definition": "unique projected candidate states per source; clean baseline and final audit excluded",
        "candidate_pool": {
            "reference_frozen_during_search": True,
            "beam_state_cardinality": int(config["edit_count"]),
            "maximum_preranked_positions": config["maximum_preranked_positions"],
            "candidates_per_position": config["candidate_actions_per_position"],
            "eligible_positions_min": int(eligible_position_counts.min()),
            "eligible_positions_mean": float(eligible_position_counts.mean()),
            "eligible_positions_max": int(eligible_position_counts.max()),
            "uses_victim_for_preranking_or_proposal": False,
        },
        "policy_gate_used": True,
        "policy_gate_used_for_method_or_threshold_selection": False,
        "final_test_used": False,
        "evaluator_b_role": "offline final identity audit only",
        "query_matched_random": {
            "detailed": random_detailed,
            "aggregate": random_aggregate,
            "accounting": random_accounting,
        },
        "beam_coordinate": {
            "detailed": beam_detailed,
            "accounting": beam_accounting,
        },
        "greedy_coordinate": {
            "detailed": greedy_detailed,
            "accounting": greedy_accounting,
        },
        "paired_comparisons": comparisons,
        "gate_q2_evidence": q2_evidence,
        "auxiliary_p5b": auxiliary_p5b,
        "shared_candidate_cache": {
            "unique_projected_states": len(margin_cache),
            "scope": "identical sample/edit tuples across P6 methods and seeds",
        },
        "limitations": [
            "The P6 v2 proposal set is recomputed from each clean reference, then frozen during search; affected local infill contexts are not refreshed after each coordinate edit.",
            "Evaluator B is a preliminary same-domain identity auditor, not ground-truth device identity.",
            "Waveform validity is digital-domain only; no over-the-air claim is made.",
        ],
        "gate": {
            "status": "PASS" if gate_pass else "FAIL",
            "q1_pass": q1_pass,
            "q2_pass": q2_pass,
            "reason": (
                "At least one pre-registered project Gate Q condition is met against the query-matched random baseline."
                if gate_pass
                else "Neither the 5-pp paired-effect condition nor the 50%-query condition is met."
            ),
            "ppo_authorized": False,
            "authorization_note": "A PASS permits only preparation and expert review of a new PPO specification; implementation/training remains prohibited.",
        },
    }
    report_dir = ROOT / "reports/next_stage"
    report_path = report_dir / "p6_query_gate_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# P6 Query Search Gate",
        "",
        "- Gate Q: **%s**" % report["gate"]["status"],
        "- Gate Q1/Q2: **%s / %s**"
        % ("PASS" if q1_pass else "FAIL", "PASS" if q2_pass else "FAIL"),
        "- Sources: 425 (25/device), fixed untouched-buffer policy Gate",
        "- Codec-joint-correct denominator: %d" % int(codec_joint.sum()),
        "- Edit budget: %d tokens" % int(config["edit_count"]),
        "- Final test used: **false**",
        "- Evaluator B: offline final audit only",
        "",
        "| method | Q cap | valid-hard | 95% CI | mean queries | B retention | min-device B | RF valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for budget in config["query_budgets"]:
        key = "q%d" % int(budget)
        random_value = random_aggregate[key]
        random_min_b = float(
            np.mean(
                [
                    minimum_device_retention(entry["metrics"])
                    for entry in random_detailed[key]
                ]
            )
        )
        lines.append(
            "| query-matched random | %d | %.3f | [%.3f, %.3f] | %.1f | %.3f | %.3f | %.3f |"
            % (
                budget,
                random_value["valid_hard_rate_pooled"],
                random_value["valid_hard_bootstrap_95ci"][0],
                random_value["valid_hard_bootstrap_95ci"][1],
                random_value["search_queries_mean"],
                random_value["evaluator_retention_on_codec_correct"],
                random_min_b,
                random_value["rf_valid_fraction"],
            )
        )
        greedy_value = greedy_detailed[key]
        lines.append(
            "| greedy-coordinate | %d | %.3f | [%.3f, %.3f] | %.1f | %.3f | %.3f | %.3f |"
            % (
                budget,
                greedy_value["valid_hard_rate_pooled"],
                greedy_value["valid_hard_bootstrap_95ci"][0],
                greedy_value["valid_hard_bootstrap_95ci"][1],
                greedy_value["search_queries_mean"],
                greedy_value["evaluator_retention_on_codec_correct"],
                minimum_device_retention(greedy_value),
                greedy_value["rf_valid_fraction"],
            )
        )
        beam_value = beam_detailed[key]
        lines.append(
            "| beam-coordinate | %d | %.3f | [%.3f, %.3f] | %.1f | %.3f | %.3f | %.3f |"
            % (
                budget,
                beam_value["valid_hard_rate_pooled"],
                beam_value["valid_hard_bootstrap_95ci"][0],
                beam_value["valid_hard_bootstrap_95ci"][1],
                beam_value["search_queries_mean"],
                beam_value["evaluator_retention_on_codec_correct"],
                minimum_device_retention(beam_value),
                beam_value["rf_valid_fraction"],
            )
        )
    lines.extend(
        [
            "",
            "| Q | searcher | searcher - random (pp) | paired 95% CI (pp) | Gate Q1 |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for budget in config["query_budgets"]:
        for searcher, value in comparisons["q%d" % int(budget)].items():
            lines.append(
                "| %d | %s | %.2f | [%.2f, %.2f] | %s |"
                % (
                    budget,
                    searcher.replace("_minus_random", ""),
                    value["difference_percentage_points"],
                    value["bootstrap_95ci_percentage_points"][0],
                    value["bootstrap_95ci_percentage_points"][1],
                    "PASS" if value["gate_q1_pass"] else "FAIL",
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The random baseline receives the same candidate pool, edit budget, projection, and per-source query cap. P5b zero-query random results are auxiliary and do not decide Gate Q. Candidate proposals are frozen from the clean reference during P6 v2, so local infill contexts are not refreshed after accepted edits; this remains a stated limitation.",
            "",
            "A Gate Q PASS does **not** authorize PPO implementation or training. It only permits drafting a new three-head PPO specification for separate expert approval. `final_test` remains sealed.",
        ]
    )
    summary_path = report_dir / "p6_query_gate_report.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stage_report = {
        "schema_version": "rffi-next-stage-report-v1",
        "stage": report["stage"],
        "git_commit": None,
        "config_hash": report["config_sha256"],
        "data_split_hash": split_config["role_assignment_sha256"],
        "checkpoint_hash": sha256_file(infill_checkpoint_path),
        "seeds": config["random_search_seeds"],
        "source_count": len(selected_rows),
        "victim_query_budget": config["query_budgets"],
        "edit_budget": [config["edit_count"]],
        "metrics": report,
        "gate": report["gate"],
        "artifacts": {
            "report_json": str(report_path),
            "report_markdown": str(summary_path),
            "source_pool": str(resolve(config["source_pool"])),
        },
        "next_allowed_stage": (
            "P7_specification_review_only" if gate_pass else "P2_to_P6_diagnosis"
        ),
        "prohibited_actions": [
            "implement or train PPO before separate expert approval",
            "use Evaluator B in proposal, reward, search, or online filtering",
            "access final_test signal data",
            "claim over-the-air or physical realizability",
        ],
    }
    stage_path = report_dir / "p6_stage_report.json"
    stage_path.write_text(
        json.dumps(stage_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "gate": report["gate"],
                "shared_unique_projected_states": len(margin_cache),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
