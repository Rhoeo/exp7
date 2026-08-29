"""Compact 1-D autoencoder and VQ-VAE for complex baseband waveforms."""

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvDown(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ConvUp(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )


class ResizeConvUp(nn.Sequential):
    """Interpolation followed by convolution avoids transposed-conv periodicity."""

    def __init__(self, in_channels, out_channels, activate=True):
        layers = [
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(
                in_channels, out_channels, kernel_size=5, padding=2, bias=not activate
            ),
        ]
        if activate:
            layers.extend(
                [
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                ]
            )
        super().__init__(*layers)


class SubPixelUp(nn.Module):
    """Efficient 1-D sub-pixel upsampler with artifact-resistant ICNR init."""

    def __init__(self, in_channels, out_channels, activate=True):
        super().__init__()
        self.out_channels = out_channels
        self.scale = 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels * self.scale,
            kernel_size=5,
            padding=2,
            bias=True,
        )
        kernel = torch.empty(out_channels, in_channels, 5)
        nn.init.kaiming_normal_(kernel, nonlinearity="linear")
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat_interleave(self.scale, dim=0))
            self.conv.bias.zero_()
        self.post = (
            nn.Sequential(
                nn.GroupNorm(_group_count(out_channels), out_channels), nn.GELU()
            )
            if activate
            else nn.Identity()
        )

    def forward(self, inputs):
        projected = self.conv(inputs)
        batch, _, length = projected.shape
        expanded = projected.view(batch, self.out_channels, self.scale, length)
        expanded = expanded.permute(0, 1, 3, 2).contiguous()
        return self.post(expanded.view(batch, self.out_channels, length * self.scale))


class WaveformEncoder(nn.Module):
    """Reduce the sample rate by 16 while keeping a temporal token sequence."""

    def __init__(self, base_width=16, latent_dim=64):
        super().__init__()
        widths = (base_width, base_width * 2, base_width * 3, base_width * 4)
        self.features = nn.Sequential(
            ConvDown(2, widths[0], 9),
            ConvDown(widths[0], widths[1], 7),
            ConvDown(widths[1], widths[2], 5),
            ConvDown(widths[2], widths[3], 5),
        )
        self.project = nn.Conv1d(widths[3], latent_dim, kernel_size=1)

    def forward(self, inputs):
        return self.project(self.features(inputs))


class PolyphaseEncoder(nn.Module):
    """Losslessly pack adjacent IQ samples before a learnable token projection."""

    def __init__(self, patch_size=4, latent_dim=64):
        super().__init__()
        self.patch_size = int(patch_size)
        self.packed_channels = 2 * self.patch_size
        if latent_dim < self.packed_channels:
            raise ValueError("latent_dim must be at least 2 * patch_size")
        self.project = nn.Conv1d(self.packed_channels, latent_dim, kernel_size=1)
        with torch.no_grad():
            self.project.weight.zero_()
            self.project.bias.zero_()
            identity = torch.eye(self.packed_channels)
            self.project.weight[: self.packed_channels, :, 0].copy_(identity)

    def forward(self, inputs):
        batch, channels, length = inputs.shape
        if channels != 2 or length % self.patch_size != 0:
            raise ValueError("IQ length must be divisible by patch_size")
        tokens = length // self.patch_size
        packed = inputs.view(batch, channels, tokens, self.patch_size)
        packed = packed.permute(0, 1, 3, 2).contiguous()
        return self.project(packed.view(batch, self.packed_channels, tokens))


class PolyphaseDecoder(nn.Module):
    """Inverse projection and exact unpacking paired with PolyphaseEncoder."""

    def __init__(self, patch_size=4, latent_dim=64):
        super().__init__()
        self.patch_size = int(patch_size)
        self.packed_channels = 2 * self.patch_size
        if latent_dim < self.packed_channels:
            raise ValueError("latent_dim must be at least 2 * patch_size")
        self.project = nn.Conv1d(latent_dim, self.packed_channels, kernel_size=1)
        with torch.no_grad():
            self.project.weight.zero_()
            self.project.bias.zero_()
            identity = torch.eye(self.packed_channels)
            self.project.weight[:, : self.packed_channels, 0].copy_(identity)

    def forward(self, latent):
        packed = self.project(latent)
        batch, _, tokens = packed.shape
        unpacked = packed.view(batch, 2, self.patch_size, tokens)
        unpacked = unpacked.permute(0, 1, 3, 2).contiguous()
        return unpacked.view(batch, 2, tokens * self.patch_size)


class WaveformDecoder(nn.Module):
    def __init__(self, base_width=16, latent_dim=64, decoder_type="subpixel"):
        super().__init__()
        if decoder_type not in ("subpixel", "resize_conv", "transpose"):
            raise ValueError(
                "decoder_type must be 'subpixel', 'resize_conv', or 'transpose'"
            )
        widths = (base_width, base_width * 2, base_width * 3, base_width * 4)
        self.project = nn.Sequential(
            nn.Conv1d(latent_dim, widths[3], kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(widths[3]), widths[3]),
            nn.GELU(),
        )
        if decoder_type == "subpixel":
            self.decode = nn.Sequential(
                SubPixelUp(widths[3], widths[2]),
                SubPixelUp(widths[2], widths[1]),
                SubPixelUp(widths[1], widths[0]),
                SubPixelUp(widths[0], 2, activate=False),
            )
        elif decoder_type == "resize_conv":
            self.decode = nn.Sequential(
                ResizeConvUp(widths[3], widths[2]),
                ResizeConvUp(widths[2], widths[1]),
                ResizeConvUp(widths[1], widths[0]),
                ResizeConvUp(widths[0], 2, activate=False),
            )
        else:
            self.decode = nn.Sequential(
                ConvUp(widths[3], widths[2]),
                ConvUp(widths[2], widths[1]),
                ConvUp(widths[1], widths[0]),
                nn.ConvTranspose1d(
                    widths[0], 2, kernel_size=4, stride=2, padding=1, bias=True
                ),
            )

    def forward(self, latent):
        return self.decode(self.project(latent))


class VectorQuantizer(nn.Module):
    """Straight-through vector quantizer with an optimizer-trained codebook."""

    def __init__(self, codebook_size=256, latent_dim=64, commitment_beta=0.25):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.latent_dim = int(latent_dim)
        self.commitment_beta = float(commitment_beta)
        self.embedding = nn.Embedding(self.codebook_size, self.latent_dim)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / self.codebook_size,
            1.0 / self.codebook_size,
        )

    def _initialize_from_batch(self, flat):
        """Seed every code from real latent vectors before gradient updates."""
        count = int(flat.shape[0])
        if count == 0:
            return
        if count >= self.codebook_size:
            indices = torch.randperm(count, device=flat.device)[: self.codebook_size]
        else:
            repeats = (self.codebook_size + count - 1) // count
            indices = torch.arange(count, device=flat.device).repeat(repeats)[
                : self.codebook_size
            ]
        with torch.no_grad():
            self.embedding.weight.copy_(flat[indices])
            self.initialized.fill_(True)

    def forward(self, latent):
        if latent.ndim != 3 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim, time]")
        flat = latent.permute(0, 2, 1).contiguous().view(-1, self.latent_dim)
        if self.training and not bool(self.initialized):
            self._initialize_from_batch(flat.detach())
        codebook = self.embedding.weight
        distances = (
            flat.square().sum(dim=1, keepdim=True)
            + codebook.square().sum(dim=1).unsqueeze(0)
            - 2.0 * torch.matmul(flat, codebook.t())
        )
        indices = distances.argmin(dim=1)
        quantized = F.embedding(indices, codebook).view(
            latent.shape[0], latent.shape[2], self.latent_dim
        )
        quantized = quantized.permute(0, 2, 1).contiguous()
        codebook_loss = F.mse_loss(quantized, latent.detach())
        commitment_loss = F.mse_loss(latent, quantized.detach())
        vq_loss = codebook_loss + self.commitment_beta * commitment_loss
        straight_through = latent + (quantized - latent).detach()
        return straight_through, vq_loss, indices.view(latent.shape[0], -1)


class WaveformAutoencoder(nn.Module):
    """Shared reconstruction backbone; quantization is optional by construction."""

    def __init__(
        self,
        mode="ae",
        base_width=16,
        latent_dim=64,
        codebook_size=256,
        commitment_beta=0.25,
        decoder_type="subpixel",
        architecture="conv",
        patch_size=4,
    ):
        super().__init__()
        if mode not in ("ae", "vqvae"):
            raise ValueError("mode must be 'ae' or 'vqvae'")
        if architecture not in ("conv", "polyphase"):
            raise ValueError("architecture must be 'conv' or 'polyphase'")
        self.mode = mode
        self.architecture = architecture
        self.patch_size = int(patch_size)
        if architecture == "polyphase":
            self.encoder = PolyphaseEncoder(
                patch_size=self.patch_size, latent_dim=latent_dim
            )
        else:
            self.encoder = WaveformEncoder(base_width=base_width, latent_dim=latent_dim)
        self.quantizer = (
            VectorQuantizer(
                codebook_size=codebook_size,
                latent_dim=latent_dim,
                commitment_beta=commitment_beta,
            )
            if mode == "vqvae"
            else None
        )
        if architecture == "polyphase":
            self.decoder = PolyphaseDecoder(
                patch_size=self.patch_size, latent_dim=latent_dim
            )
        else:
            self.decoder = WaveformDecoder(
                base_width=base_width,
                latent_dim=latent_dim,
                decoder_type=decoder_type,
            )

    def forward(self, inputs):
        if inputs.ndim != 3 or inputs.shape[1] != 2:
            raise ValueError("inputs must have shape [batch, 2, time]")
        divisor = self.patch_size if self.architecture == "polyphase" else 16
        if inputs.shape[-1] % divisor != 0:
            raise ValueError("waveform length must be divisible by %d" % divisor)
        encoded = self.encoder(inputs)
        if self.quantizer is None:
            quantized = encoded
            vq_loss = encoded.new_zeros(())
            code_indices = None
        else:
            quantized, vq_loss, code_indices = self.quantizer(encoded)
        reconstruction = self.decoder(quantized)
        if reconstruction.shape != inputs.shape:
            raise RuntimeError(
                "decoder shape %s differs from input shape %s"
                % (tuple(reconstruction.shape), tuple(inputs.shape))
            )
        return {
            "reconstruction": reconstruction,
            "encoded": encoded,
            "quantized": quantized,
            "vq_loss": vq_loss,
            "code_indices": code_indices,
        }

    def encode_code_indices(self, inputs):
        if self.quantizer is None:
            raise RuntimeError("discrete encoding requires a VQ-VAE model")
        encoded = self.encoder(inputs)
        _, _, indices = self.quantizer(encoded)
        return indices

    def decode_code_indices(self, indices):
        if self.quantizer is None:
            raise RuntimeError("discrete decoding requires a VQ-VAE model")
        if indices.ndim != 2:
            raise ValueError("indices must have shape [batch, tokens]")
        latent = F.embedding(indices.long(), self.quantizer.embedding.weight)
        return self.decoder(latent.permute(0, 2, 1).contiguous())


def build_reconstruction_model(model_config):
    required = {"mode", "base_width", "latent_dim", "codebook_size", "commitment_beta"}
    missing = required.difference(model_config)
    if missing:
        raise ValueError("model config is missing: %s" % sorted(missing))
    return WaveformAutoencoder(
        mode=model_config["mode"],
        base_width=int(model_config["base_width"]),
        latent_dim=int(model_config["latent_dim"]),
        codebook_size=int(model_config["codebook_size"]),
        commitment_beta=float(model_config["commitment_beta"]),
        # Checkpoints produced before decoder versioning used transposed convolutions.
        decoder_type=model_config.get("decoder_type", "transpose"),
        architecture=model_config.get("architecture", "conv"),
        patch_size=int(model_config.get("patch_size", 4)),
    )
