"""Common projection applied before every classifier query or validity claim."""

from __future__ import annotations

import math

import torch

from rffi_core.attacks.rf_constraints import bandlimit_complex
from rffi_core.metrics.rf_validity_extended import complex_power


def _scale_to_reference_power(candidate, reference):
    candidate_power = complex_power(candidate).clamp_min(1e-12)
    reference_power = complex_power(reference).clamp_min(1e-12)
    scale = torch.sqrt(reference_power / candidate_power).view(-1, 1, 1)
    return candidate * scale


def project_delta_power(delta, reference, min_snr_db):
    reference_power = complex_power(reference).clamp_min(1e-12)
    delta_power = complex_power(delta).clamp_min(1e-20)
    maximum = reference_power * (10.0 ** (-float(min_snr_db) / 10.0))
    scale = torch.sqrt(maximum / delta_power).clamp_max(1.0).view(-1, 1, 1)
    return delta * scale, scale.flatten()


def project_delta_peak(delta, reference, max_normalized_peak_delta):
    reference_rms = complex_power(reference).clamp_min(1e-12).sqrt()
    peak = torch.sqrt(
        delta[:, 0].square() + delta[:, 1].square() + 1e-20
    ).amax(dim=1)
    maximum = reference_rms * float(max_normalized_peak_delta)
    scale = (maximum / peak.clamp_min(1e-20)).clamp_max(1.0).view(-1, 1, 1)
    return delta * scale, scale.flatten()


def common_projection_pipeline(
    reference,
    candidate,
    sample_rate_hz=35e6,
    occupied_bandwidth_hz=22e6,
    min_snr_db=22.0,
    max_normalized_peak_delta=1.0,
    match_reference_power=True,
    bandlimit_perturbation=True,
):
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate shapes differ")
    reference = reference.float()
    candidate = candidate.float()
    if match_reference_power:
        candidate = _scale_to_reference_power(candidate, reference)
    delta = candidate - reference
    if bandlimit_perturbation:
        delta = bandlimit_complex(
            delta,
            sample_rate_hz=sample_rate_hz,
            occupied_bandwidth_hz=occupied_bandwidth_hz,
        )
    delta, power_scale = project_delta_power(delta, reference, min_snr_db)
    delta, peak_scale = project_delta_peak(
        delta, reference, max_normalized_peak_delta
    )
    projected = reference + delta
    return projected, {
        "power_projection_scale": power_scale,
        "peak_projection_scale": peak_scale,
        "changed_fraction": (
            (power_scale < 1.0) | (peak_scale < 1.0)
        ).float(),
    }


def actual_decoded_local_precheck(
    decoded_reference,
    position,
    decoded_candidate_value,
    max_normalized_peak_delta=0.75,
    max_normalized_jump_increase=1.0,
):
    """Check an actually decoded single-token replacement in local context."""
    if decoded_reference.ndim != 2 or decoded_reference.shape[0] != 2:
        raise ValueError("decoded_reference must have shape [2, time]")
    position = int(position)
    candidate = decoded_candidate_value.reshape(2).to(decoded_reference)
    reference_power = (
        decoded_reference[0].square() + decoded_reference[1].square()
    ).mean().clamp_min(1e-12)
    reference_rms = reference_power.sqrt()
    original = decoded_reference[:, position]
    delta_peak = torch.linalg.vector_norm(candidate - original) / reference_rms
    jumps = []
    baseline_jumps = []
    if position > 0:
        previous = decoded_reference[:, position - 1]
        jumps.append(torch.linalg.vector_norm(candidate - previous))
        baseline_jumps.append(torch.linalg.vector_norm(original - previous))
    if position + 1 < decoded_reference.shape[1]:
        following = decoded_reference[:, position + 1]
        jumps.append(torch.linalg.vector_norm(following - candidate))
        baseline_jumps.append(torch.linalg.vector_norm(following - original))
    jump_increase = (
        (torch.stack(jumps).max() - torch.stack(baseline_jumps).max()) / reference_rms
        if jumps
        else decoded_reference.new_zeros(())
    )
    passed = bool(
        delta_peak <= float(max_normalized_peak_delta)
        and jump_increase <= float(max_normalized_jump_increase)
    )
    return passed, {
        "normalized_peak_delta": float(delta_peak),
        "normalized_jump_increase": float(jump_increase),
    }
