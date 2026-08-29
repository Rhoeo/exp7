"""Per-sample digital waveform-validity metrics for complex baseband IQ."""

from __future__ import annotations

import math

import torch


def _validate(iq, name):
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("%s must have shape [batch, 2, time]" % name)
    return iq.float()


def complex_power(iq):
    iq = _validate(iq, "iq")
    return (iq[:, 0].square() + iq[:, 1].square()).mean(dim=1)


def complex_peak_power(iq):
    iq = _validate(iq, "iq")
    return (iq[:, 0].square() + iq[:, 1].square()).amax(dim=1)


def _power_spectrum(iq):
    iq = _validate(iq, "iq")
    values = torch.complex(iq[:, 0], iq[:, 1])
    spectrum = torch.fft.fft(values, dim=-1, norm="ortho")
    return spectrum.real.square() + spectrum.imag.square()


def _oob_ratio(power_spectrum, sample_rate_hz, occupied_bandwidth_hz):
    frequencies = torch.fft.fftfreq(
        power_spectrum.shape[-1],
        d=1.0 / float(sample_rate_hz),
        device=power_spectrum.device,
    )
    in_band = frequencies.abs() <= float(occupied_bandwidth_hz) / 2.0
    inside = power_spectrum[:, in_band].sum(dim=1)
    outside = power_spectrum[:, ~in_band].sum(dim=1)
    return outside / inside.clamp_min(1e-12)


def rf_metric_tensors(
    candidate,
    reference,
    sample_rate_hz=35e6,
    occupied_bandwidth_hz=22e6,
    clipping_rms_multiple=4.0,
):
    candidate = _validate(candidate, "candidate")
    reference = _validate(reference, "reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference shapes differ")
    delta = candidate - reference
    reference_power = complex_power(reference).clamp_min(1e-12)
    candidate_power = complex_power(candidate).clamp_min(1e-12)
    delta_power = complex_power(delta)
    relative_power = delta_power / reference_power
    reference_rms = reference_power.sqrt()
    candidate_amplitude = torch.sqrt(
        candidate[:, 0].square() + candidate[:, 1].square() + 1e-20
    )
    delta_amplitude = torch.sqrt(delta[:, 0].square() + delta[:, 1].square() + 1e-20)
    flattened_candidate = candidate.flatten(start_dim=1)
    flattened_reference = reference.flatten(start_dim=1)
    correlation = (flattened_candidate * flattened_reference).sum(dim=1) / (
        flattened_candidate.square().sum(dim=1).sqrt().clamp_min(1e-12)
        * flattened_reference.square().sum(dim=1).sqrt().clamp_min(1e-12)
    )
    candidate_psd = _power_spectrum(candidate)
    reference_psd = _power_spectrum(reference)
    candidate_psd_normalized = candidate_psd / candidate_psd.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    reference_psd_normalized = reference_psd / reference_psd.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-12)
    psd_l1 = 0.5 * torch.abs(
        candidate_psd_normalized - reference_psd_normalized
    ).sum(dim=1)
    candidate_diff = candidate[:, :, 1:] - candidate[:, :, :-1]
    delta_diff = delta[:, :, 1:] - delta[:, :, :-1]
    candidate_diff_amplitude = torch.sqrt(
        candidate_diff[:, 0].square() + candidate_diff[:, 1].square() + 1e-20
    )
    delta_diff_amplitude = torch.sqrt(
        delta_diff[:, 0].square() + delta_diff[:, 1].square() + 1e-20
    )
    finite = torch.isfinite(candidate).all(dim=2).all(dim=1)
    return {
        "snr_db": -10.0 * torch.log10(relative_power.clamp_min(1e-12)),
        "relative_perturbation_power": relative_power,
        "evm_like": relative_power.sqrt(),
        "delta_papr": complex_peak_power(delta) / delta_power.clamp_min(1e-12),
        "final_signal_papr": complex_peak_power(candidate) / candidate_power,
        "normalized_peak_delta": delta_amplitude.amax(dim=1) / reference_rms,
        "clipping_fraction": (
            candidate_amplitude
            > float(clipping_rms_multiple) * reference_rms[:, None]
        )
        .float()
        .mean(dim=1),
        "waveform_correlation": correlation,
        "psd_l1_distance": psd_l1,
        "observable_oob_energy_ratio": _oob_ratio(
            candidate_psd, sample_rate_hz, occupied_bandwidth_hz
        ),
        "observable_oob_energy_ratio_db": 10.0
        * torch.log10(
            _oob_ratio(candidate_psd, sample_rate_hz, occupied_bandwidth_hz).clamp_min(
                1e-12
            )
        ),
        "delta_oob_energy_ratio": _oob_ratio(
            _power_spectrum(delta), sample_rate_hz, occupied_bandwidth_hz
        ),
        "normalized_max_final_derivative": candidate_diff_amplitude.amax(dim=1)
        / reference_rms,
        "normalized_max_delta_derivative": delta_diff_amplitude.amax(dim=1)
        / reference_rms,
        "finite": finite.float(),
    }


def aggregate_metric_tensors(metrics):
    output = {}
    for name, values in metrics.items():
        values = values.detach().float().cpu()
        if values.numel() == 0:
            output[name] = {}
            continue
        output[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "p50": float(torch.quantile(values, 0.50)),
            "p95": float(torch.quantile(values, 0.95)),
            "max": float(values.max()),
            "finite_fraction": float(torch.isfinite(values).float().mean()),
        }
    return output


def basic_validity_mask(
    metrics,
    min_snr_db=22.0,
    max_normalized_peak_delta=1.0,
):
    return (
        (metrics["finite"] == 1)
        & (metrics["snr_db"] >= float(min_snr_db) - 1e-4)
        & (
            metrics["normalized_peak_delta"]
            <= float(max_normalized_peak_delta) + 1e-4
        )
    )
