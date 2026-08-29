import unittest

import numpy as np
import torch

from rffi_core.data.token_datasets import WiFiBTokenDataset
from rffi_core.generators.vqvae.losses import reconstruction_loss
from rffi_core.generators.vqvae.models import build_reconstruction_model


class ReconstructionModelTests(unittest.TestCase):
    def _config(self, mode):
        return {
            "mode": mode,
            "base_width": 8,
            "latent_dim": 16,
            "codebook_size": 32,
            "commitment_beta": 0.25,
        }

    def test_ae_preserves_shape_and_backpropagates(self):
        model = build_reconstruction_model(self._config("ae"))
        inputs = torch.randn(2, 2, 256)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs["reconstruction"].shape), tuple(inputs.shape))
        self.assertIsNone(outputs["code_indices"])
        outputs["reconstruction"].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_vqvae_indices_and_rf_loss_are_finite(self):
        model = build_reconstruction_model(self._config("vqvae"))
        inputs = torch.randn(2, 2, 256)
        outputs = model(inputs)
        self.assertTrue(bool(model.quantizer.initialized))
        self.assertEqual(tuple(outputs["code_indices"].shape), (2, 16))
        decoded = model.decode_code_indices(outputs["code_indices"])
        self.assertTrue(torch.equal(decoded, outputs["reconstruction"]))
        losses = reconstruction_loss(
            outputs["reconstruction"],
            inputs,
            outputs["vq_loss"],
            {"waveform": 1.0, "correlation": 0.1, "spectral": 0.1, "power": 0.05, "vq": 1.0},
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(model.quantizer.embedding.weight.grad)

    def test_invalid_length_is_rejected(self):
        model = build_reconstruction_model(self._config("ae"))
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 250))

    def test_resize_conv_decoder_preserves_shape(self):
        config = self._config("ae")
        config["decoder_type"] = "resize_conv"
        outputs = build_reconstruction_model(config)(torch.randn(1, 2, 256))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (1, 2, 256))

    def test_subpixel_decoder_preserves_shape(self):
        config = self._config("ae")
        config["decoder_type"] = "subpixel"
        outputs = build_reconstruction_model(config)(torch.randn(1, 2, 256))
        self.assertEqual(tuple(outputs["reconstruction"].shape), (1, 2, 256))

    def test_polyphase_ae_starts_as_exact_identity(self):
        config = self._config("ae")
        config.update({"architecture": "polyphase", "patch_size": 4})
        inputs = torch.randn(2, 2, 256)
        reconstruction = build_reconstruction_model(config)(inputs)["reconstruction"]
        self.assertTrue(torch.equal(reconstruction, inputs))


class TokenDatasetTests(unittest.TestCase):
    def test_token_dataset_uses_cache_index_alignment(self):
        import csv
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "cache_index": "0",
                    "device_id": "Device2",
                    "sample_id": "a",
                    "split": "generator_train",
                },
                {
                    "cache_index": "1",
                    "device_id": "Device1",
                    "sample_id": "b",
                    "split": "reward_validation",
                },
            ]
            with (root / "index.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            np.save(
                str(root / "tokens.npy"),
                np.asarray([[4, 5], [8, 9]], dtype=np.uint16),
            )
            dataset = WiFiBTokenDataset(
                root / "index.csv", root / "tokens.npy", "reward_validation"
            )
            self.assertEqual(dataset[0]["tokens"].tolist(), [8, 9])
            self.assertEqual(dataset[0]["label"], 0)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
