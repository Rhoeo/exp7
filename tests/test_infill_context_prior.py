import unittest

import numpy as np
import torch

from rffi_core.generators.infill.context_prior import BidirectionalTransitionPrior


class InfillContextPriorTests(unittest.TestCase):
    def test_bidirectional_context_prefers_supported_middle(self):
        transitions = np.ones((4, 4), dtype=np.int64)
        transitions[0, 2] = 50
        transitions[2, 1] = 50
        prior = BidirectionalTransitionPrior(transitions, np.ones(4), alpha=1.0)
        tokens = torch.tensor([[0, 4, 1]], dtype=torch.long)
        indicator = torch.tensor([[0, 1, 0]], dtype=torch.long)
        scores = prior.scores(tokens, indicator)
        self.assertEqual((1, 4), tuple(scores.shape))
        self.assertEqual(2, int(scores.argmax(dim=1)))


if __name__ == "__main__":
    unittest.main()
