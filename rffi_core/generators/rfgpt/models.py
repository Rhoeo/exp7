"""A compact causal Transformer for RF-token sequences."""

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

    def forward(self, inputs):
        batch, length, width = inputs.shape
        qkv = self.qkv(inputs).view(
            batch, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        inputs = inputs + self.attention_dropout(
            self.attention(self.attention_norm(inputs))
        )
        return inputs + self.mlp(self.mlp_norm(inputs))


class DeviceConditionedRFGPT(nn.Module):
    def __init__(
        self,
        codebook_size,
        num_devices,
        max_sequence_length=2048,
        context_length=256,
        d_model=128,
        num_heads=4,
        num_layers=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        if context_length > max_sequence_length:
            raise ValueError("context_length cannot exceed max_sequence_length")
        self.codebook_size = int(codebook_size)
        self.num_devices = int(num_devices)
        self.max_sequence_length = int(max_sequence_length)
        self.context_length = int(context_length)
        self.token_embedding = nn.Embedding(codebook_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.device_embedding = nn.Embedding(num_devices, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, codebook_size, bias=False)
        self.apply(self._initialize)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, device_labels, position_offset=0):
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, time]")
        batch, length = tokens.shape
        if length > self.context_length:
            raise ValueError("input exceeds configured context_length")
        if device_labels.shape != (batch,):
            raise ValueError("device_labels must have shape [batch]")
        if torch.is_tensor(position_offset):
            offsets = position_offset.to(tokens.device).long().view(batch, 1)
        else:
            offsets = torch.full(
                (batch, 1), int(position_offset), device=tokens.device, dtype=torch.long
            )
        positions = offsets + torch.arange(length, device=tokens.device).view(1, -1)
        if int(positions.min()) < 0 or int(positions.max()) >= self.max_sequence_length:
            raise ValueError("absolute token position is outside the configured sequence")
        hidden = (
            self.token_embedding(tokens.long())
            + self.position_embedding(positions)
            + self.device_embedding(device_labels.long()).unsqueeze(1)
        )
        hidden = self.input_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def build_rfgpt(model_config):
    return DeviceConditionedRFGPT(
        codebook_size=int(model_config["codebook_size"]),
        num_devices=int(model_config["num_devices"]),
        max_sequence_length=int(model_config.get("max_sequence_length", 2048)),
        context_length=int(model_config.get("context_length", 256)),
        d_model=int(model_config.get("d_model", 128)),
        num_heads=int(model_config.get("num_heads", 4)),
        num_layers=int(model_config.get("num_layers", 4)),
        mlp_ratio=float(model_config.get("mlp_ratio", 4.0)),
        dropout=float(model_config.get("dropout", 0.1)),
    )

