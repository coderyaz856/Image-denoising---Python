"""Guards for the held-out evaluation split and paired analysis."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_mdp import (  # noqa: E402
    TEST_SEEDS, TEST_SOURCES, TRAIN_SEEDS, TRAIN_SOURCES,
    cluster_bootstrap_difference, source_patches,
)


class ProtocolTests(unittest.TestCase):
    def test_source_images_and_noise_seeds_do_not_cross_splits(self):
        self.assertFalse(set(TRAIN_SOURCES) & set(TEST_SOURCES))
        self.assertFalse(set(TRAIN_SEEDS) & set(TEST_SEEDS))

    def test_each_source_has_three_eighty_pixel_patches(self):
        for name in TRAIN_SOURCES + TEST_SOURCES:
            patches = source_patches(name)
            self.assertEqual(len(patches), 3)
            self.assertTrue(all(patch.shape == (80, 80, 3) for patch in patches))

    def test_paired_difference_uses_source_clusters(self):
        rows = []
        for source, delta in (("first", 1.0), ("second", -1.0)):
            for patch in range(3):
                shared = {"source": source, "patch": patch, "noise": "gaussian",
                          "level": 25.0, "seed": 101}
                rows.append({**shared, "method": "a", "psnr": 20.0 + delta})
                rows.append({**shared, "method": "b", "psnr": 20.0})
        result = cluster_bootstrap_difference(rows, "a", "b", "psnr", repeats=100)
        self.assertEqual(result["source_clusters"], 2)
        self.assertEqual(result["mean_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
