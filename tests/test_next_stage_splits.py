import unittest

from rffi_core.data.build_next_stage_splits import (
    allocate_stratified_counts,
    allocate_with_caps,
    identify_buffer_segments,
    stable_key,
)


class NextStageSplitTests(unittest.TestCase):
    def test_stable_key_is_deterministic_and_namespaced(self):
        first = stable_key(7, "a", "sample")
        self.assertEqual(first, stable_key(7, "a", "sample"))
        self.assertNotEqual(first, stable_key(7, "b", "sample"))

    def test_stratified_allocation_reaches_exact_target(self):
        result = allocate_stratified_counts({"1": 11, "2": 10, "10": 9}, 17)
        self.assertEqual(17, sum(result.values()))
        self.assertTrue(all(result[key] <= value for key, value in {"1": 11, "2": 10, "10": 9}.items()))

    def test_buffer_segments_follow_rank_gaps(self):
        rows = [
            {"split": "buffer", "device_id": "1", "sample_id": "a", "chronological_rank": "10"},
            {"split": "buffer", "device_id": "1", "sample_id": "b", "chronological_rank": "11"},
            {"split": "buffer", "device_id": "1", "sample_id": "c", "chronological_rank": "20"},
            {"split": "buffer", "device_id": "1", "sample_id": "d", "chronological_rank": "30"},
        ]
        self.assertEqual({"a": 0, "b": 0, "c": 1, "d": 2}, identify_buffer_segments(rows))

    def test_segment_allocation_rebalances_capacity(self):
        self.assertEqual([16, 17, 17], allocate_with_caps(50, [20, 20, 20]))
        result = allocate_with_caps(50, [10, 25, 25])
        self.assertEqual(50, sum(result))
        self.assertEqual(10, result[0])
        self.assertTrue(all(result[index] <= [10, 25, 25][index] for index in range(3)))


if __name__ == "__main__":
    unittest.main()
