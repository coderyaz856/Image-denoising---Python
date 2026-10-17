"""Focused regression tests for the denoising search and benchmark primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_denoising import (  # noqa: E402
    add_gaussian_noise,
    add_salt_pepper_noise,
    calculate_quality_metrics,
    operation_median_blur,
    optimize_image_denoising,
)


def checkerboard(size: int = 72, block: int = 12) -> np.ndarray:
    rows, columns = np.indices((size, size))
    pattern = ((rows // block + columns // block) % 2 * 255).astype(np.uint8)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


class NoiseModelTests(unittest.TestCase):
    def test_gaussian_noise_is_reproducible_for_a_seed(self) -> None:
        image = checkerboard()
        first = add_gaussian_noise(image, sigma=25, seed=9)
        second = add_gaussian_noise(image, sigma=25, seed=9)
        self.assertTrue(np.array_equal(first, second))

    def test_salt_pepper_noise_is_reproducible_for_a_seed(self) -> None:
        image = checkerboard()
        first = add_salt_pepper_noise(image, amount=0.08, seed=9)
        second = add_salt_pepper_noise(image, amount=0.08, seed=9)
        self.assertTrue(np.array_equal(first, second))


class SearchTests(unittest.TestCase):
    def test_metrics_use_float_subtraction_without_uint8_wraparound(self) -> None:
        dark = np.zeros((8, 8), dtype=np.uint8)
        light = np.full((8, 8), 255, dtype=np.uint8)
        metrics = calculate_quality_metrics(dark, light)
        self.assertEqual(metrics["mse"], 255.0**2)

    def test_median_filter_improves_seeded_impulse_noise(self) -> None:
        clean = checkerboard()
        noisy = add_salt_pepper_noise(clean, amount=0.08, seed=4)
        baseline = calculate_quality_metrics(clean, noisy)

        denoised, history = optimize_image_denoising(
            noisy,
            reference_image=clean,
            operations=[operation_median_blur],
            max_iterations=2,
            verbose=False,
        )
        final = calculate_quality_metrics(clean, denoised)

        self.assertGreater(final["psnr"], baseline["psnr"])
        self.assertEqual(history[1]["operation"], "operation_median_blur")


if __name__ == "__main__":
    unittest.main()
