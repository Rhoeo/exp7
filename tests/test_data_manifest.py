import io
import os
import pickle
import tempfile
import unittest

import numpy as np

from rffi_core.data.build_data_manifest import (
    RestrictedManyTxUnpickler,
    assign_rank_splits,
    assign_receiver_roles,
    choose_manytx_split,
)
from rffi_core.data.convert_manytx_cache import ManyTxCacheSink


class DataManifestTests(unittest.TestCase):
    def test_chronological_split_counts(self):
        segments = [
            {"split": "generator_train", "fraction": 0.53},
            {"split": "buffer", "fraction": 0.01},
            {"split": "reward_validation", "fraction": 0.15},
            {"split": "buffer", "fraction": 0.01},
            {"split": "defense_train", "fraction": 0.15},
            {"split": "buffer", "fraction": 0.01},
            {"split": "final_test", "fraction": 0.14},
        ]
        labels = assign_rank_splits(100, segments)
        self.assertEqual(53, labels.count("generator_train"))
        self.assertEqual(3, labels.count("buffer"))
        self.assertEqual(15, labels.count("reward_validation"))
        self.assertEqual(15, labels.count("defense_train"))
        self.assertEqual(14, labels.count("final_test"))

    def test_receiver_roles_are_deterministic_and_complete(self):
        receivers = ["rx-%02d" % value for value in range(18)]
        partitions = {"generator_train": 12, "defense_train": 3, "final_test": 3}
        first = assign_receiver_roles(receivers, partitions, 20260828)
        second = assign_receiver_roles(receivers, partitions, 20260828)
        self.assertEqual(first, second)
        self.assertEqual(12, list(first.values()).count("generator_train"))
        self.assertEqual(3, list(first.values()).count("defense_train"))
        self.assertEqual(3, list(first.values()).count("final_test"))

    def test_manytx_final_test_requires_held_date_and_receiver(self):
        dates = ["d1", "d2", "d3", "d4"]
        self.assertEqual("final_test", choose_manytx_split("d4", "final_test", dates))
        self.assertEqual(
            "diagnostic_cross_receiver",
            choose_manytx_split("d1", "final_test", dates),
        )
        self.assertEqual(
            "diagnostic_cross_day",
            choose_manytx_split("d4", "generator_train", dates),
        )

    def test_restricted_unpickler_rejects_non_numpy_globals(self):
        payload = pickle.dumps(os.system, protocol=3)
        with self.assertRaises(pickle.UnpicklingError):
            RestrictedManyTxUnpickler(io.BytesIO(payload)).load()

    def test_manytx_cache_sink_converts_and_indexes(self):
        rows = [{"group_id": "g0", "sample_count": "2"}]
        with tempfile.TemporaryDirectory() as directory:
            iq_path = os.path.join(directory, "iq.npy")
            power_path = os.path.join(directory, "power.npy")
            iq = np.lib.format.open_memmap(
                iq_path, mode="w+", dtype=np.float32, shape=(2, 2, 2)
            )
            power = np.lib.format.open_memmap(
                power_path, mode="w+", dtype=np.float32, shape=(2,)
            )
            sink = ManyTxCacheSink(rows, iq, power, (2, 2))
            source = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
            state = (1, source.shape, source.dtype, False, source.tobytes())
            sink.consume_array_state(state)
            np.testing.assert_allclose(source.astype(np.float32), iq)
            np.testing.assert_allclose(
                np.mean(
                    np.sum(source.astype(np.float32) ** 2, axis=2), axis=1
                ),
                power,
            )
            self.assertEqual(0, sink.index_rows[0]["cache_offset"])
            self.assertEqual(2, sink.index_rows[0]["cache_count"])
            iq.flush()
            power.flush()
            del sink
            del iq
            del power


if __name__ == "__main__":
    unittest.main()
