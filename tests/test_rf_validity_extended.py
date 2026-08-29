import unittest

import torch

from rffi_core.attacks.common_projection_pipeline import (
    actual_decoded_local_precheck,
    common_projection_pipeline,
)
from rffi_core.metrics.rf_validity_extended import (
    aggregate_metric_tensors,
    basic_validity_mask,
    rf_metric_tensors,
)


class RFValidityExtendedTests(unittest.TestCase):
    def test_identity_metrics_are_finite(self):
        clean = torch.randn(3, 2, 256)
        metrics = rf_metric_tensors(clean, clean)
        self.assertTrue(bool(torch.isfinite(metrics["snr_db"]).all()))
        self.assertTrue(bool((metrics["relative_perturbation_power"] == 0).all()))
        summary = aggregate_metric_tensors(metrics)
        self.assertIn("final_signal_papr", summary)

    def test_projection_enforces_power_and_peak(self):
        torch.manual_seed(3)
        clean = torch.randn(4, 2, 256)
        candidate = clean + 2.0 * torch.randn_like(clean)
        projected, diagnostics = common_projection_pipeline(
            clean,
            candidate,
            min_snr_db=22.0,
            max_normalized_peak_delta=0.8,
        )
        metrics = rf_metric_tensors(projected, clean)
        self.assertTrue(bool(basic_validity_mask(metrics, 22.0, 0.8).all()))
        self.assertEqual((4,), tuple(diagnostics["changed_fraction"].shape))

    def test_local_precheck_uses_actual_values(self):
        reference = torch.zeros(2, 8)
        passed, metrics = actual_decoded_local_precheck(
            reference + 1.0,
            4,
            torch.tensor([1.1, 1.1]),
            max_normalized_peak_delta=0.5,
        )
        self.assertTrue(passed)
        self.assertIn("normalized_jump_increase", metrics)


if __name__ == "__main__":
    unittest.main()
