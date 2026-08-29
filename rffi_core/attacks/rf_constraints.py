"""Differentiable, reproducible constraints for complex baseband perturbations."""

import torch


def bandlimit_complex(iq, sample_rate_hz=35e6, occupied_bandwidth_hz=22e6):
    """Keep a centered occupied band and return real/imaginary channels."""
    if iq.ndim != 3 or iq.shape[1] != 2:
        raise ValueError("IQ must have shape [batch, 2, time]")
    complex_iq = torch.complex(iq[:, 0].float(), iq[:, 1].float())
    spectrum = torch.fft.fft(complex_iq, dim=-1, norm="ortho")
    frequencies = torch.fft.fftfreq(
        iq.shape[-1], d=1.0 / float(sample_rate_hz), device=iq.device
    )
    mask = (frequencies.abs() <= float(occupied_bandwidth_hz) / 2.0).to(
        spectrum.real.dtype
    )
    filtered = torch.fft.ifft(spectrum * mask, dim=-1, norm="ortho")
    return torch.stack((filtered.real, filtered.imag), dim=1).to(iq.dtype)

