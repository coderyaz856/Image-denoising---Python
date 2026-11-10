"""Finite-horizon denoising policy learned with fitted Q iteration.

The policy observes only the noisy/current image and never receives a clean
reference at inference time. Clean references are used to label transitions
during training and to evaluate held-out cases. This is an approximate MDP:
compact image features alias many underlying images and noise states.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from image_denoising import (
    DENOISING_OPERATIONS,
    calculate_quality_metrics,
    calculate_score,
)


FEATURE_NAMES = (
    "gray_mean", "gray_std", "extreme_fraction", "median_residual",
    "gaussian_residual", "laplacian_energy", "edge_fraction",
)
ACTION_NAMES = tuple(operation.__name__ for operation in DENOISING_OPERATIONS)
MODEL_VERSION = 1


def observable_features(image: np.ndarray) -> np.ndarray:
    """Compute reference-free, deterministic features from a uint8 image."""
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("expected a uint8 grayscale or BGR image")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32) / 255.0
    median = cv2.medianBlur(gray, 3).astype(np.float32) / 255.0
    gaussian = cv2.GaussianBlur(gray_f, (3, 3), 0)
    laplacian = cv2.Laplacian(gray_f, cv2.CV_32F)
    edges = cv2.Canny(gray, 60, 120)
    return np.array(
        [
            gray_f.mean(), gray_f.std(),
            np.mean((gray <= 2) | (gray >= 253)),
            np.mean(np.abs(gray_f - median)),
            np.mean(np.abs(gray_f - gaussian)),
            np.mean(np.abs(laplacian)),
            np.mean(edges != 0),
        ],
        dtype=np.float64,
    )


class FittedMDPPolicy:
    """A small fitted-Q policy with k-nearest-neighbor value approximation.

    The explicit stop action has value zero. Every filter costs ``action_cost``
    units of the repository's composite quality score. The horizon bounds the
    number of filtering steps. Training enumerates every available action at
    each sampled state, so action-value labels are not selected-action-only.
    """

    def __init__(self, horizon: int = 3, neighbors: int = 7,
                 action_cost: float = 0.01):
        if horizon < 1 or neighbors < 1 or action_cost < 0:
            raise ValueError("horizon and neighbors must be positive; cost nonnegative")
        self.horizon = horizon
        self.neighbors = neighbors
        self.action_cost = action_cost
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.states: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

    @staticmethod
    def _state(initial: np.ndarray, current: np.ndarray) -> np.ndarray:
        return np.concatenate((initial, observable_features(current)))

    def _predict(self, depth: int, queries: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or not self.states:
            raise RuntimeError("fit the policy or load a model before prediction")
        queries = np.atleast_2d(queries)
        train = (self.states[depth] - self.mean) / self.scale
        result = np.empty((len(queries), len(ACTION_NAMES)), dtype=np.float64)
        count = min(self.neighbors, len(train))
        for start in range(0, len(queries), 64):
            query = (queries[start:start + 64] - self.mean) / self.scale
            distances = np.sum((query[:, None, :] - train[None, :, :]) ** 2, axis=2)
            nearest = np.argpartition(distances, count - 1, axis=1)[:, :count]
            near_distances = np.take_along_axis(distances, nearest, axis=1)
            weights = 1.0 / (np.sqrt(near_distances) + 0.05)
            near_values = self.values[depth][nearest]
            result[start:start + len(query)] = (
                np.sum(near_values * weights[:, :, None], axis=1)
                / np.sum(weights, axis=1)[:, None]
            )
        return result

    def fit(self, cases: list[tuple[np.ndarray, np.ndarray]],
            exploration_paths: int = 2, seed: int = 7) -> FittedMDPPolicy:
        """Fit on (clean, noisy) pairs; no reference is stored in the model."""
        if not cases or exploration_paths < 1:
            raise ValueError("at least one case and exploration path are required")
        rng = np.random.default_rng(seed)
        states = [[] for _ in range(self.horizon)]
        rewards = [[] for _ in range(self.horizon)]
        successors = [[] for _ in range(self.horizon)]
        for clean, noisy in cases:
            if clean.shape != noisy.shape:
                raise ValueError("clean and noisy images must have the same shape")
            initial = observable_features(noisy)
            for path in range(exploration_paths):
                current = noisy.copy()
                for depth in range(self.horizon):
                    current_score = calculate_score(calculate_quality_metrics(clean, current))
                    next_images = [operation(current) for operation in DENOISING_OPERATIONS]
                    step_rewards = np.array([
                        calculate_score(calculate_quality_metrics(clean, image))
                        - current_score - self.action_cost
                        for image in next_images
                    ], dtype=np.float64)
                    states[depth].append(self._state(initial, current))
                    rewards[depth].append(step_rewards)
                    successors[depth].append(np.stack([
                        self._state(initial, image) for image in next_images
                    ]))
                    # One trajectory follows the training oracle, the others
                    # explore. The oracle is never consulted by predict/denoise.
                    choice = int(np.argmax(step_rewards)) if path == 0 else int(
                        rng.integers(len(DENOISING_OPERATIONS))
                    )
                    current = next_images[choice]
        self.states = [np.stack(stage) for stage in states]
        reward_arrays = [np.stack(stage) for stage in rewards]
        successor_arrays = [np.stack(stage) for stage in successors]
        all_states = np.concatenate(self.states)
        self.mean = all_states.mean(axis=0)
        self.scale = np.maximum(all_states.std(axis=0), 1e-5)
        self.values = [np.zeros_like(stage) for stage in reward_arrays]
        for depth in reversed(range(self.horizon)):
            target = reward_arrays[depth].copy()
            if depth + 1 < self.horizon:
                next_states = successor_arrays[depth].reshape(-1, all_states.shape[1])
                continuation = np.maximum(
                    0.0, self._predict(depth + 1, next_states).max(axis=1)
                )
                target += continuation.reshape(target.shape)
            self.values[depth] = target
        return self

    def choose(self, current: np.ndarray, initial_features: np.ndarray,
               depth: int) -> str | None:
        """Select a filter or stop using only observable image features."""
        if depth >= self.horizon:
            return None
        values = self._predict(depth, self._state(initial_features, current))[0]
        choice = int(np.argmax(values))
        return ACTION_NAMES[choice] if values[choice] > 0 else None

    def denoise(self, noisy: np.ndarray) -> tuple[np.ndarray, list[str]]:
        initial = observable_features(noisy)
        current = noisy.copy()
        history = []
        for depth in range(self.horizon):
            action = self.choose(current, initial, depth)
            if action is None:
                break
            index = ACTION_NAMES.index(action)
            current = DENOISING_OPERATIONS[index](current)
            history.append(action)
        return current, history

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.scale is None or not self.states:
            raise RuntimeError("cannot save an unfitted policy")
        payload = {
            "model_version": np.array(MODEL_VERSION),
            "horizon": np.array(self.horizon),
            "neighbors": np.array(self.neighbors),
            "action_cost": np.array(self.action_cost),
            "mean": self.mean,
            "scale": self.scale,
            "action_names": np.array(ACTION_NAMES),
            "feature_names": np.array(FEATURE_NAMES),
        }
        for depth in range(self.horizon):
            payload[f"states_{depth}"] = self.states[depth]
            payload[f"values_{depth}"] = self.values[depth]
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> FittedMDPPolicy:
        with np.load(path, allow_pickle=False) as archive:
            if int(archive["model_version"]) != MODEL_VERSION:
                raise ValueError("unsupported model version")
            if tuple(archive["action_names"].tolist()) != ACTION_NAMES:
                raise ValueError("model action set differs from this code version")
            if tuple(archive["feature_names"].tolist()) != FEATURE_NAMES:
                raise ValueError("model feature set differs from this code version")
            policy = cls(int(archive["horizon"]), int(archive["neighbors"]),
                         float(archive["action_cost"]))
            policy.mean = archive["mean"]
            policy.scale = archive["scale"]
            policy.states = [archive[f"states_{i}"] for i in range(policy.horizon)]
            policy.values = [archive[f"values_{i}"] for i in range(policy.horizon)]
        return policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a trained, reference-free MDP policy")
    parser.add_argument("--model", required=True, help="NPZ produced by benchmark_mdp.py")
    parser.add_argument("--input", required=True, help="noisy image")
    parser.add_argument("--output", required=True, help="denoised image")
    args = parser.parse_args()
    noisy = cv2.imread(args.input)
    if noisy is None:
        parser.error(f"cannot read image: {args.input}")
    result, actions = FittedMDPPolicy.load(args.model).denoise(noisy)
    if not cv2.imwrite(args.output, result):
        parser.error(f"cannot write image: {args.output}")
    print("Selected chain:", " -> ".join(actions) or "stop without filtering")


if __name__ == "__main__":
    main()
