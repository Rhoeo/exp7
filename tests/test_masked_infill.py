import unittest

import torch

from rffi_core.generators.infill.models import (
    MaskedInfillTransformer,
    masked_cross_entropy,
)


class MaskedInfillTests(unittest.TestCase):
    def test_output_shape_and_masked_backward(self):
        model = MaskedInfillTransformer(
            codebook_size=32,
            num_devices=3,
            context_length=16,
            d_model=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        tokens = torch.randint(0, 33, (2, 16))
        targets = torch.full((2, 16), -100, dtype=torch.long)
        targets[:, 3] = torch.tensor([1, 2])
        logits = model(
            tokens,
            torch.tensor([0, 2]),
            torch.linspace(0, 1, 16)[None, :].repeat(2, 1),
            (targets != -100).long(),
        )
        self.assertEqual((2, 16, 32), tuple(logits.shape))
        loss = masked_cross_entropy(logits, targets)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_invalid_context_shape_is_rejected(self):
        model = MaskedInfillTransformer(
            codebook_size=8,
            num_devices=2,
            context_length=4,
            d_model=8,
            num_heads=2,
            num_layers=1,
        )
        with self.assertRaises(ValueError):
            model(
                torch.zeros((1, 3), dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
                torch.zeros((1, 3)),
                torch.zeros((1, 3), dtype=torch.long),
            )


if __name__ == "__main__":
    unittest.main()
