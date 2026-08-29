import unittest

import torch

from rffi_core.generators.rfgpt.models import build_rfgpt
from rffi_core.generators.rfgpt.evaluate_generation import sample_logits


class RFGPTTests(unittest.TestCase):
    def _model(self):
        return build_rfgpt(
            {
                "codebook_size": 32,
                "num_devices": 3,
                "max_sequence_length": 64,
                "context_length": 16,
                "d_model": 32,
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            }
        )

    def test_output_shape_and_backpropagation(self):
        model = self._model()
        tokens = torch.randint(0, 32, (2, 12))
        labels = torch.tensor([0, 2])
        logits = model(tokens, labels, torch.tensor([3, 5]))
        self.assertEqual(tuple(logits.shape), (2, 12, 32))
        logits.square().mean().backward()
        self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_causality(self):
        model = self._model().eval()
        first = torch.randint(0, 32, (1, 12))
        second = first.clone()
        second[:, 8:] = torch.randint(0, 32, (1, 4))
        label = torch.tensor([1])
        with torch.inference_mode():
            first_logits = model(first, label)
            second_logits = model(second, label)
        self.assertTrue(torch.allclose(first_logits[:, :8], second_logits[:, :8]))

    def test_context_limit_is_enforced(self):
        model = self._model()
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 17, dtype=torch.long), torch.tensor([0]))

    def test_top_p_sampling_returns_vocabulary_indices(self):
        torch.manual_seed(3)
        logits = torch.randn(5, 32)
        sampled = sample_logits(logits, temperature=0.8, top_p=0.9)
        self.assertEqual(tuple(sampled.shape), (5,))
        self.assertTrue(bool(((sampled >= 0) & (sampled < 32)).all()))


if __name__ == "__main__":
    unittest.main()
