import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rffi_core.generators.token_candidates.token_candidate_graph import (
    TokenCandidateGraph,
    mutual_neighbor_mask,
    nearest_neighbor_arrays,
    undirected_component_sizes,
)


class TokenCandidateGraphTests(unittest.TestCase):
    def test_nearest_neighbors_are_deterministic_and_exclude_self(self):
        embeddings = np.asarray([[0.0], [1.0], [3.0], [7.0]], dtype=np.float32)
        first, distances = nearest_neighbor_arrays(embeddings, 2)
        second, _ = nearest_neighbor_arrays(embeddings, 2)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(bool(np.any(first == np.arange(4)[:, None])))
        self.assertTrue(bool(np.all(np.isfinite(distances))))

    def test_mutual_mask_and_components(self):
        neighbors = np.asarray([[1], [0], [1]], dtype=np.int32)
        mask = mutual_neighbor_mask(neighbors)
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 0]))
        self.assertFalse(bool(mask[2, 0]))
        self.assertEqual([3], undirected_component_sizes(neighbors))

    def test_round_trip_and_candidate_filters(self):
        neighbors = np.asarray([[1, 2], [0, 2], [1, 0]], dtype=np.int32)
        arrays = {
            "neighbors": neighbors,
            "latent_distances": np.ones((3, 2), dtype=np.float32),
            "mutual_neighbors": mutual_neighbor_mask(neighbors),
            "usage_counts": np.ones(3, dtype=np.int64),
            "forward_transition_counts": np.asarray([[1, 0], [1, 1], [0, 0]], dtype=np.int64),
            "backward_transition_counts": np.zeros((3, 2), dtype=np.int64),
            "transition_counts": np.ones((3, 3), dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.npz"
            np.savez_compressed(
                str(path),
                **arrays,
                metadata_json=np.asarray(json.dumps({"version": 1})),
            )
            graph = TokenCandidateGraph.load(path)
            np.testing.assert_array_equal([1], graph.candidates(0, transition_supported=True))
            self.assertTrue(graph.validate())
            self.assertEqual((3, 3), graph.transition_counts.shape)


if __name__ == "__main__":
    unittest.main()
