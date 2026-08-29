import unittest

import numpy as np
import torch

from rffi_core.attacks.query_search import (
    apply_token_edits,
    interleaved_candidate_actions,
    normalized_position_prerank,
    keep_best_beam_states,
    select_unique_position_actions,
    true_class_margin,
)


class QuerySearchTests(unittest.TestCase):
    def test_true_class_margin(self):
        logits = torch.tensor([[3.0, 1.0, 2.0], [0.0, 4.0, 1.0]])
        margins = true_class_margin(logits, torch.tensor([0, 2]))
        torch.testing.assert_close(margins, torch.tensor([1.0, -3.0]))

    def test_prerank_is_deterministic_and_interior(self):
        tokens = np.asarray([[0, 1, 2, 3, 0, 1]], dtype=np.int64)
        log_forward = np.log(np.full((4, 4), 0.25, dtype=np.float32))
        spread = np.arange(4, dtype=np.float32)
        sensitivity = np.arange(4, dtype=np.float32)[::-1]
        first = normalized_position_prerank(tokens, log_forward, spread, sensitivity, 3)
        second = normalized_position_prerank(tokens, log_forward, spread, sensitivity, 3)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(bool(np.all((first >= 1) & (first <= 4))))

    def test_select_actions_uses_unique_positions_and_no_op(self):
        actions = [
            {"position": 1, "candidate": 3, "margin": 0.2},
            {"position": 1, "candidate": 4, "margin": 0.1},
            {"position": 2, "candidate": 5, "margin": 0.3},
            {"position": 3, "candidate": 6, "margin": 1.2},
        ]
        selected = select_unique_position_actions(actions, 3, baseline_margin=1.0)
        self.assertEqual([1, 2], [value["position"] for value in selected])
        self.assertEqual(4, selected[0]["candidate"])

    def test_beam_states_are_deduplicated_by_edits(self):
        states = [
            {"edits": ((2, 4),), "margin": 0.5},
            {"edits": ((2, 4),), "margin": 0.3},
            {"edits": ((1, 3),), "margin": 0.4},
        ]
        kept = keep_best_beam_states(states, 2)
        self.assertEqual([0.3, 0.4], [state["margin"] for state in kept])

    def test_candidate_actions_are_position_diverse(self):
        proposals = [
            {"position": 7, "eligible": [10, 11]},
            {"position": 9, "eligible": [20, 21]},
        ]
        actions = interleaved_candidate_actions(proposals, 2)
        self.assertEqual([7, 9, 7, 9], [action["position"] for action in actions])
        self.assertEqual([10, 20, 11, 21], [action["candidate"] for action in actions])

    def test_apply_token_edits_is_sparse_and_non_mutating(self):
        reference = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        edited = apply_token_edits(reference, [1, 0], [((0, 8),), ((2, 9),)])
        np.testing.assert_array_equal(edited, np.asarray([[8, 5, 6], [1, 2, 9]]))
        np.testing.assert_array_equal(reference, np.asarray([[1, 2, 3], [4, 5, 6]]))


if __name__ == "__main__":
    unittest.main()
