"""Reference-leakage and reproducibility tests for the fitted MDP policy."""

from __future__ import annotations

import sys
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_denoising import add_gaussian_noise, add_salt_pepper_noise  # noqa: E402
from mdp_denoising import (  # noqa: E402
    ACTION_NAMES, FittedMDPPolicy, observable_features,
)


def scene(kind: int) -> np.ndarray:
    image = np.full((32, 32, 3), 70 + 25 * kind, dtype=np.uint8)
    cv2.circle(image, (12, 13), 7, (210, 150, 100), -1)
    cv2.line(image, (0, 26), (31, 26), (30, 240, 50), 2)
    return image


class FeatureTests(unittest.TestCase):
    def test_observation_is_deterministic_and_reference_free(self):
        image = scene(0)
        np.testing.assert_array_equal(observable_features(image), observable_features(image))
        self.assertEqual(observable_features(image).shape, (7,))

    def test_rejects_non_uint8_input(self):
        with self.assertRaises(ValueError):
            observable_features(scene(0).astype(np.float32))


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        clean_a, clean_b = scene(0), scene(1)
        noisy_a = add_salt_pepper_noise(clean_a, amount=0.12, seed=3)
        noisy_b = add_gaussian_noise(clean_b, sigma=25, seed=5)
        cls.noisy = noisy_a
        cls.policy = FittedMDPPolicy(horizon=2, neighbors=2)
        cls.policy.fit([(clean_a, noisy_a), (clean_b, noisy_b)],
                       exploration_paths=2, seed=8)

    def test_enumerates_all_actions_and_bounds_horizon(self):
        self.assertEqual(len(self.policy.values), 2)
        self.assertEqual(self.policy.states[0].shape, (4, 14))
        self.assertEqual(self.policy.values[0].shape[1], len(ACTION_NAMES))
        _, chain = self.policy.denoise(self.noisy)
        self.assertLessEqual(len(chain), 2)
        self.assertTrue(set(chain).issubset(ACTION_NAMES))

    def test_inference_is_deterministic_and_does_not_need_clean_reference(self):
        first, first_chain = self.policy.denoise(self.noisy)
        second, second_chain = self.policy.denoise(self.noisy)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_chain, second_chain)

    def test_explicit_stop_action_can_win(self):
        policy = FittedMDPPolicy(horizon=1, neighbors=1)
        policy.mean = np.zeros(14)
        policy.scale = np.ones(14)
        policy.states = [np.zeros((1, 14))]
        policy.values = [np.full((1, len(ACTION_NAMES)), -0.1)]
        result, chain = policy.denoise(self.noisy)
        np.testing.assert_array_equal(result, self.noisy)
        self.assertEqual(chain, [])

    def test_save_load_preserves_predictions_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            self.policy.save(path)
            loaded = FittedMDPPolicy.load(path)
            a, chain_a = self.policy.denoise(self.noisy)
            b, chain_b = loaded.denoise(self.noisy)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(chain_a, chain_b)

    def test_fitting_rejects_mismatched_references(self):
        with self.assertRaises(ValueError):
            FittedMDPPolicy().fit([(scene(0), np.zeros((4, 4, 3), dtype=np.uint8))])

    def test_fixed_seed_reproduces_fitted_values(self):
        clean = scene(0)
        noisy = add_gaussian_noise(clean, sigma=20, seed=2)
        a = FittedMDPPolicy(horizon=2).fit([(clean, noisy)], seed=14)
        b = FittedMDPPolicy(horizon=2).fit([(clean, noisy)], seed=14)
        for left, right in zip(a.values, b.values):
            np.testing.assert_array_equal(left, right)

    def test_model_rejects_incompatible_feature_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            self.policy.save(path)
            with np.load(path, allow_pickle=False) as archive:
                data = {key: archive[key] for key in archive.files}
            data["feature_names"] = np.array(["changed"])
            np.savez_compressed(path, **data)
            with self.assertRaises(ValueError):
                FittedMDPPolicy.load(path)

    def test_reference_free_command_line_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "noisy.png"
            target = folder / "restored.png"
            model = folder / "policy.npz"
            self.assertTrue(cv2.imwrite(str(source), self.noisy))
            self.policy.save(model)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "mdp_denoising.py"),
                 "--model", str(model), "--input", str(source),
                 "--output", str(target)],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("Selected chain:", completed.stdout)
            self.assertEqual(cv2.imread(str(target)).shape, self.noisy.shape)


if __name__ == "__main__":
    unittest.main()
