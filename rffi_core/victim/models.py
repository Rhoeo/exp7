"""Small 1-D classifiers sized for the local 2 GB GPU."""

import torch
from torch import nn


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=7):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, kernel_size, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.GELU()

    def forward(self, inputs):
        return self.activation(self.conv2(self.conv1(inputs)) + self.skip(inputs))


class VictimRawIQCNN(nn.Module):
    """Victim A: time-domain residual CNN."""

    def __init__(self, num_classes, base_width=16, dropout=0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormAct(2, base_width, 11, stride=4),
            ResidualBlock1D(base_width, base_width, kernel_size=7),
            ResidualBlock1D(base_width, base_width * 2, stride=2, kernel_size=7),
            ResidualBlock1D(base_width * 2, base_width * 3, stride=2, kernel_size=5),
            ResidualBlock1D(base_width * 3, base_width * 4, stride=2, kernel_size=5),
        )
        feature_width = base_width * 4
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_width * 2),
            nn.Dropout(dropout),
            nn.Linear(feature_width * 2, num_classes),
        )

    def embedding(self, inputs):
        features = self.features(inputs)
        pooled_mean = torch.mean(features, dim=-1)
        pooled_max = torch.amax(features, dim=-1)
        return torch.cat((pooled_mean, pooled_max), dim=1)

    def forward(self, inputs):
        return self.classifier(self.embedding(inputs))


class EvaluatorSpectralCNN(nn.Module):
    """Evaluator B: FFT-domain multi-scale CNN with a separate feature bias."""

    def __init__(self, num_classes, branch_width=12, dropout=0.25):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(3, branch_width, 3, stride=4),
                ConvNormAct(3, branch_width, 9, stride=4),
                ConvNormAct(3, branch_width, 17, stride=4),
            ]
        )
        merged = branch_width * 3
        self.features = nn.Sequential(
            ResidualBlock1D(merged, 48, stride=2, kernel_size=7),
            ResidualBlock1D(48, 72, stride=2, kernel_size=5),
            ResidualBlock1D(72, 96, stride=2, kernel_size=5),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Dropout(dropout),
            nn.Linear(192, num_classes),
        )

    @staticmethod
    def spectral_view(inputs):
        complex_iq = torch.complex(inputs[:, 0], inputs[:, 1])
        spectrum = torch.fft.fftshift(
            torch.fft.fft(complex_iq, dim=-1, norm="ortho"), dim=-1
        )
        # Avoid the complex-abs Jiterator path: the local CUDA environment has
        # the FFT runtime but not the optional NVRTC builtins DLL.
        magnitude = torch.log1p(
            torch.sqrt(spectrum.real.square() + spectrum.imag.square() + 1e-12)
        )
        magnitude = (magnitude - magnitude.mean(dim=-1, keepdim=True)) / (
            magnitude.std(dim=-1, keepdim=True).clamp_min(1e-5)
        )
        return torch.stack((spectrum.real, spectrum.imag, magnitude), dim=1)

    def embedding(self, inputs):
        spectral = self.spectral_view(inputs)
        merged = torch.cat([branch(spectral) for branch in self.branches], dim=1)
        features = self.features(merged)
        pooled_mean = torch.mean(features, dim=-1)
        pooled_max = torch.amax(features, dim=-1)
        return torch.cat((pooled_mean, pooled_max), dim=1)

    def forward(self, inputs):
        return self.classifier(self.embedding(inputs))


def build_model(name, num_classes):
    if name == "victim_a":
        return VictimRawIQCNN(num_classes=num_classes)
    if name == "evaluator_b":
        return EvaluatorSpectralCNN(num_classes=num_classes)
    raise ValueError("unknown model architecture: %s" % name)
