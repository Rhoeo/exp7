"""Frozen bidirectional transition prior derived only from generator_train."""

from __future__ import annotations

import torch


class BidirectionalTransitionPrior:
    def __init__(self, transition_counts, usage_counts, alpha=1.0):
        counts = torch.as_tensor(transition_counts, dtype=torch.float32)
        usage = torch.as_tensor(usage_counts, dtype=torch.float32)
        if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
            raise ValueError("transition counts must be a square matrix")
        if usage.shape != (counts.shape[0],):
            raise ValueError("usage counts shape differs from transition counts")
        self.codebook_size = int(counts.shape[0])
        alpha = float(alpha)
        self.log_forward = torch.log(counts + alpha) - torch.log(
            counts.sum(dim=1, keepdim=True) + alpha * self.codebook_size
        )
        self.log_unigram = torch.log(usage + alpha) - torch.log(
            usage.sum() + alpha * self.codebook_size
        )

    def to(self, device):
        self.log_forward = self.log_forward.to(device)
        self.log_unigram = self.log_unigram.to(device)
        return self

    def scores(self, tokens, target_indicator):
        """Return [masked_targets, codebook] context scores in row-major mask order."""
        if tokens.shape != target_indicator.shape:
            raise ValueError("tokens and target_indicator shapes differ")
        indices = torch.nonzero(target_indicator != 0, as_tuple=False)
        scores = []
        for batch_index, position in indices.tolist():
            pieces = []
            if position > 0:
                left = int(tokens[batch_index, position - 1])
                if 0 <= left < self.codebook_size:
                    pieces.append(self.log_forward[left])
            if position + 1 < tokens.shape[1]:
                right = int(tokens[batch_index, position + 1])
                if 0 <= right < self.codebook_size:
                    pieces.append(self.log_forward[:, right])
            scores.append(sum(pieces) if pieces else self.log_unigram)
        if not scores:
            return self.log_forward.new_empty((0, self.codebook_size))
        return torch.stack(scores, dim=0)
