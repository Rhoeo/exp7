"""RF-aware reconstruction objectives without classifier feedback."""

import torch


def mean_complex_power(iq):
    return (iq[:, 0].square() + iq[:, 1].square()).mean(dim=-1)


def normalize_complex_power(iq, epsilon=1e-8):
    scale = mean_complex_power(iq).clamp_min(epsilon).sqrt().view(-1, 1, 1)
    return iq / scale


def complex_fft_magnitude(iq):
    complex_iq = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    spectrum = torch.fft.fft(complex_iq, dim=-1, norm="ortho")
    # Manual magnitude avoids the optional CUDA complex-abs Jiterator path.
    return torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-12)


def reconstruction_loss(reconstruction, target, vq_loss, weights):
    error_power = mean_complex_power(reconstruction - target)
    target_power = mean_complex_power(target).clamp_min(1e-8)
    waveform_nmse = (error_power / target_power).mean()

    flat_reconstruction = reconstruction.flatten(start_dim=1)
    flat_target = target.flatten(start_dim=1)
    correlation = (
        (flat_reconstruction * flat_target).sum(dim=1)
        / (
            flat_reconstruction.square().sum(dim=1).clamp_min(1e-8).sqrt()
            * flat_target.square().sum(dim=1).clamp_min(1e-8).sqrt()
        )
    )
    correlation_loss = (1.0 - correlation).mean()

    reconstructed_spectrum = torch.log1p(complex_fft_magnitude(reconstruction))
    target_spectrum = torch.log1p(complex_fft_magnitude(target))
    spectral_loss = (reconstructed_spectrum - target_spectrum).abs().mean()

    output_power = mean_complex_power(reconstruction)
    power_loss = (output_power - target_power).abs().mean()
    total = (
        float(weights["waveform"]) * waveform_nmse
        + float(weights["correlation"]) * correlation_loss
        + float(weights["spectral"]) * spectral_loss
        + float(weights["power"]) * power_loss
        + float(weights["vq"]) * vq_loss
    )
    return {
        "total": total,
        "waveform_nmse": waveform_nmse,
        "correlation_loss": correlation_loss,
        "spectral_log_l1": spectral_loss,
        "power_loss": power_loss,
        "vq_loss": vq_loss,
        "output_power": output_power.mean(),
    }

