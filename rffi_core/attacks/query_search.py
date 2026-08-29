"""Query-accounted score-based helpers for discrete token editing."""

from __future__ import annotations

import numpy as np
import torch


def true_class_margin(logits, labels):
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits/labels shapes are incompatible")
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - masked.amax(dim=1)


def normalized_position_prerank(
    token_sequences,
    log_forward,
    candidate_spread_by_token,
    decoder_sensitivity_by_token,
    top_m,
    surprisal_weight=1.0,
    candidate_spread_weight=0.25,
    decoder_sensitivity_weight=0.25,
):
    tokens = np.asarray(token_sequences, dtype=np.int64)
    log_forward = np.asarray(log_forward, dtype=np.float32)
    if tokens.ndim != 2:
        raise ValueError("token sequences must have shape [batch, time]")
    if top_m <= 0 or top_m > tokens.shape[1] - 2:
        raise ValueError("invalid top_m")
    outputs = []
    for sequence in tokens:
        left = sequence[:-2]
        center = sequence[1:-1]
        right = sequence[2:]
        surprisal = -(log_forward[left, center] + log_forward[center, right])
        spread = np.asarray(candidate_spread_by_token)[center]
        sensitivity = np.asarray(decoder_sensitivity_by_token)[center]

        def robust_z(values):
            values = np.asarray(values, dtype=np.float64)
            return (values - np.median(values)) / max(float(values.std()), 1e-8)

        score = (
            float(surprisal_weight) * robust_z(surprisal)
            + float(candidate_spread_weight) * robust_z(spread)
            + float(decoder_sensitivity_weight) * robust_z(sensitivity)
        )
        order = np.argsort(-score, kind="stable")[:top_m] + 1
        outputs.append(order.astype(np.int64))
    return np.stack(outputs, axis=0)


def select_unique_position_actions(actions, edit_count, baseline_margin):
    """Choose improving single-edit actions by ascending queried true-class margin."""
    selected = []
    used_positions = set()
    for action in sorted(actions, key=lambda value: (value["margin"], value["position"], value["candidate"])):
        if action["margin"] >= float(baseline_margin):
            continue
        if action["position"] in used_positions:
            continue
        selected.append(action)
        used_positions.add(action["position"])
        if len(selected) == int(edit_count):
            break
    return selected


def keep_best_beam_states(states, beam_width):
    """Deduplicate edit tuples and keep the lowest-margin states."""
    best = {}
    for state in states:
        key = tuple(sorted((int(position), int(candidate)) for position, candidate in state["edits"]))
        if key not in best or float(state["margin"]) < float(best[key]["margin"]):
            best[key] = state
    return sorted(
        best.values(),
        key=lambda state: (float(state["margin"]), len(state["edits"]), tuple(state["edits"])),
    )[: int(beam_width)]


def interleaved_candidate_actions(proposals, candidates_per_position):
    """Return position-diverse actions before lower-ranked replacements."""
    actions = []
    for candidate_rank in range(int(candidates_per_position)):
        for position_rank, proposal in enumerate(proposals):
            eligible = proposal["eligible"]
            if candidate_rank >= len(eligible):
                continue
            actions.append(
                {
                    "position_rank": int(position_rank),
                    "position": int(proposal["position"]),
                    "candidate_rank": int(candidate_rank),
                    "candidate": int(eligible[candidate_rank]),
                }
            )
    return actions


def apply_token_edits(reference_tokens, sample_indices, edit_tuples):
    """Materialize a batch of sparse edits without modifying the references."""
    reference = np.asarray(reference_tokens)
    if len(sample_indices) != len(edit_tuples):
        raise ValueError("sample_indices and edit_tuples must have equal length")
    output = np.stack(
        [reference[int(sample_index)].copy() for sample_index in sample_indices], axis=0
    )
    for batch_index, edits in enumerate(edit_tuples):
        for position, candidate in edits:
            output[batch_index, int(position)] = int(candidate)
    return output
