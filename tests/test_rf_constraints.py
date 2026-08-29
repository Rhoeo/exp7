import unittest

import torch

from rffi_core.attacks.rf_constraints import bandlimit_complex


class RFConstraintTests(unittest.TestCase):
    def test_bandlimit_preserves_shape_and_is_finite(self):
        iq = torch.randn(2, 2, 256)
        filtered = bandlimit_complex(iq, sample_rate_hz=35e6, occupied_bandwidth_hz=22e6)
        self.assertEqual(tuple(filtered.shape), tuple(iq.shape))
        self.assertTrue(bool(torch.isfinite(filtered).all()))


if __name__ == "__main__":
    unittest.main()

