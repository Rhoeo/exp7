"""Small bidirectional Transformer for local RF-token infilling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedInfillTransformer(nn.Module):
    def __init__(
        self,
        codebook_size=1024,
        num_devices=17,
        context_length=256,
        d_model=128,
        num_heads=4,
        num_layers=4,
        mlp_ratio=3.0,
        dropout=0.1,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.mask_token_id = self.codebook_size
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.token_embedding = nn.Embedding(self.codebook_size + 1, self.d_model)
        self.device_embedding = nn.Embedding(int(num_devices), self.d_model)
        self.local_position_embedding = nn.Embedding(self.context_length, self.d_model)
        self.global_position_projection = nn.Linear(1, self.d_model)
        self.target_indicator_embedding = nn.Embedding(2, self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.d_model * float(mlp_ratio))),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.output_bias = nn.Parameter(torch.zeros(self.codebook_size))
        # Tied masked-LM heads require small embedding variance. PyTorch's
        # default Embedding N(0, 1) makes 1,024-way logits severely overconfident.
        for embedding in (
            self.token_embedding,
            self.device_embedding,
            self.local_position_embedding,
            self.target_indicator_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.global_position_projection.weight)
        nn.init.zeros_(self.global_position_projection.bias)

    def forward(self, tokens, device_labels, global_positions, target_indicator):
        if tokens.ndim != 2 or tokens.shape[1] != self.context_length:
            raise ValueError("tokens must have shape [batch, context_length]")
        if global_positions.shape != tokens.shape:
            raise ValueError("global_positions shape differs from tokens")
        if target_indicator.shape != tokens.shape:
            raise ValueError("target_indicator shape differs from tokens")
        local_positions = torch.arange(
            self.context_length, device=tokens.device, dtype=torch.long
        )[None, :]
        hidden = self.token_embedding(tokens.long())
        hidden = hidden + self.local_position_embedding(local_positions)
        hidden = hidden + self.device_embedding(device_labels.long())[:, None, :]
        hidden = hidden + self.global_position_projection(
            global_positions.to(dtype=hidden.dtype).unsqueeze(-1)
        )
        hidden = hidden + self.target_indicator_embedding(target_indicator.long())
        hidden = self.norm(self.encoder(hidden))
        return F.linear(
            hidden, self.token_embedding.weight[: self.codebook_size], self.output_bias
        )


def masked_cross_entropy(logits, targets, ignore_index=-100):
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def build_infill_model(config):
    return MaskedInfillTransformer(**config)
