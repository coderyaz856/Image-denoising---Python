"""Train and evaluate a reference-free finite-horizon denoising policy.

The split is by source image, not by crop or noise seed. No clean test image is
used in fitting, feature normalization, model selection, or action selection.
The existing greedy search is reported separately as a reference-aware oracle.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import skimage
from skimage import data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_denoising import (  # noqa: E402
    add_gaussian_noise, add_salt_pepper_noise, calculate_quality_metrics,
    calculate_score, operation_bilateral_filter, operation_median_blur,
    optimize_image_denoising,
)
from mdp_denoising import FittedMDPPolicy  # noqa: E402


TRAIN_SOURCES = ("astronaut", "camera", "coins", "grass")
TEST_SOURCES = ("chelsea", "coffee", "moon", "brick")
TRAIN_NOISE = (("gaussian", 15.0), ("gaussian", 25.0),
               ("salt_pepper", 0.04), ("salt_pepper", 0.08))
TEST_NOISE = TRAIN_NOISE + (("gaussian", 35.0), ("salt_pepper", 0.12))
TRAIN_SEEDS = (11, 23)
TEST_SEEDS = (101, 211)
PATCH_SIZE = 80
PATCH_ORIGINS = ((0, 0), (40, 40), (80, 80))


def source_patches(name: str) -> list[np.ndarray]:
    raw = getattr(data, name)()
    if raw.ndim == 2:
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR)
    else:
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    side = min(bgr.shape[:2])
    y = (bgr.shape[0] - side) // 2
    x = (bgr.shape[1] - side) // 2
    square = bgr[y:y + side, x:x + side]
    resized = cv2.resize(square, (160, 160), interpolation=cv2.INTER_AREA)
    return [resized[y:y + PATCH_SIZE, x:x + PATCH_SIZE].copy()
            for y, x in PATCH_ORIGINS]


def add_noise(clean: np.ndarray, kind: str, level: float, seed: int) -> np.ndarray:
    if kind == "gaussian":
        return add_gaussian_noise(clean, sigma=level, seed=seed)
    return add_salt_pepper_noise(clean, amount=level, seed=seed)


def cases(sources, noise_levels, seeds):
    for source in sources:
        for patch_index, clean in enumerate(source_patches(source)):
            for kind, level in noise_levels:
                for seed in seeds:
                    noisy = add_noise(clean, kind, level, seed)
                    yield {
                        "source": source, "patch": patch_index, "noise": kind,
                        "level": level, "seed": seed, "clean": clean, "noisy": noisy,
                    }


def rounded_metrics(clean, candidate):
    metrics = calculate_quality_metrics(clean, candidate)
    return {**{key: round(value, 6) for key, value in metrics.items()},
            "score": round(calculate_score(metrics), 6)}


def summarize(rows, method):
    subset = [row for row in rows if row["method"] == method]
    return {
        "cases": len(subset),
        "mean_psnr_db": round(float(np.mean([r["psnr"] for r in subset])), 4),
        "mean_ssim": round(float(np.mean([r["ssim"] for r in subset])), 5),
        "mean_score": round(float(np.mean([r["score"] for r in subset])), 5),
        "mean_actions": round(float(np.mean([r["actions"] for r in subset])), 3),
        "median_elapsed_ms": round(float(np.median([r["elapsed_ms"] for r in subset])), 3),
    }


def cluster_bootstrap_difference(rows, method_a, method_b, metric,
                                 seed=19, repeats=2000):
    """Paired CI resampling source images, not correlated patches as IID cases."""
    by_key = {(r["source"], r["patch"], r["noise"], r["level"], r["seed"]): r
              for r in rows if r["method"] == method_b}
    differences = defaultdict(list)
    for row in rows:
        if row["method"] == method_a:
            key = (row["source"], row["patch"], row["noise"], row["level"], row["seed"])
            differences[row["source"]].append(row[metric] - by_key[key][metric])
    cluster_means = np.array([np.mean(values) for values in differences.values()])
    rng = np.random.default_rng(seed)
    samples = rng.choice(cluster_means, size=(repeats, len(cluster_means)), replace=True)
    return {
        "mean_difference": round(float(cluster_means.mean()), 5),
        "cluster_bootstrap_95pct_ci": [round(float(x), 5) for x in
                                       np.quantile(samples.mean(axis=1), [0.025, 0.975])],
        "source_clusters": len(cluster_means),
    }


def main():
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    training = list(cases(TRAIN_SOURCES, TRAIN_NOISE, TRAIN_SEEDS))
    print(f"Fitting on {len(training)} training cases from {len(TRAIN_SOURCES)} source images...")
    policy = FittedMDPPolicy(horizon=3, neighbors=7, action_cost=0.01)
    policy.fit([(case["clean"], case["noisy"]) for case in training],
               exploration_paths=2, seed=7)
    model_path = result_dir / "mdp_policy.npz"
    policy.save(model_path)
    del training

    rows = []
    for index, case in enumerate(cases(TEST_SOURCES, TEST_NOISE, TEST_SEEDS), start=1):
        clean, noisy = case["clean"], case["noisy"]
        methods = (
            ("noisy", lambda: (noisy, [])),
            ("median_once", lambda: (operation_median_blur(noisy), ["operation_median_blur"])),
            ("bilateral_once", lambda: (operation_bilateral_filter(noisy), ["operation_bilateral_filter"])),
            ("noise_label_oracle", lambda: (
                operation_bilateral_filter(noisy) if case["noise"] == "gaussian"
                else operation_median_blur(noisy),
                ["operation_bilateral_filter" if case["noise"] == "gaussian"
                 else "operation_median_blur"],
            )),
            ("fitted_mdp", lambda: policy.denoise(noisy)),
            ("reference_aware_greedy_oracle", lambda: optimize_image_denoising(
                noisy, reference_image=clean, max_iterations=3, verbose=False)),
        )
        for name, execute in methods:
            start = time.perf_counter()
            result, history = execute()
            elapsed = (time.perf_counter() - start) * 1000
            actions = ([step["operation"] for step in history[1:]]
                       if name == "reference_aware_greedy_oracle" else history)
            rows.append({
                "source": case["source"], "patch": case["patch"],
                "noise": case["noise"], "level": case["level"],
                "seed": case["seed"], "method": name,
                **rounded_metrics(clean, result),
                "actions": len(actions), "chain": actions,
                "elapsed_ms": round(elapsed, 3),
            })
        if index % 24 == 0:
            print(f"Evaluated {index} held-out cases")
    methods = ("noisy", "median_once", "bilateral_once", "noise_label_oracle", "fitted_mdp",
               "reference_aware_greedy_oracle")
    source_hashes = {
        source: hashlib.sha256(b"".join(p.tobytes() for p in source_patches(source))).hexdigest()
        for source in TRAIN_SOURCES + TEST_SOURCES
    }
    payload = {
        "protocol": {
            "training_sources": TRAIN_SOURCES, "heldout_sources": TEST_SOURCES,
            "training_noise": TRAIN_NOISE, "heldout_noise": TEST_NOISE,
            "training_seeds": TRAIN_SEEDS, "heldout_seeds": TEST_SEEDS,
            "patches_per_source": len(PATCH_ORIGINS), "patch_size": PATCH_SIZE,
            "model": {"horizon": policy.horizon, "neighbors": policy.neighbors,
                      "action_cost": policy.action_cost, "exploration_paths": 2,
                      "fit_seed": 7},
            "notes": [
                "Split by original source image; no image or noise seed crosses splits.",
                "The greedy comparator reads clean ground truth for every action and is an oracle, not a deployable baseline.",
                "The noise-label oracle knows the synthetic corruption type and is not a blind deployable baseline.",
                "The learned policy sees only observable image features at inference; clean test images are used for final evaluation only.",
                "Source-cluster confidence intervals are descriptive with four clusters; they do not establish population-level superiority.",
            ],
        },
        "environment": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                        "numpy": np.__version__, "scikit_image": skimage.__version__},
        "source_patch_sha256": source_hashes,
        "summary": {method: summarize(rows, method) for method in methods},
        "summary_by_noise": {
            kind: {method: summarize([r for r in rows if r["noise"] == kind], method)
                   for method in methods}
            for kind in ("gaussian", "salt_pepper")
        },
        "summary_by_shift": {
            label: {method: summarize([r for r in rows if
                                       ((r["noise"], r["level"]) in levels)], method)
                    for method in methods}
            for label, levels in (
                ("trained_levels", set(TRAIN_NOISE)),
                ("unseen_stronger_levels", {("gaussian", 35.0), ("salt_pepper", 0.12)}),
            )
        },
        "paired_psnr": {
            "mdp_vs_median": cluster_bootstrap_difference(rows, "fitted_mdp", "median_once", "psnr"),
            "mdp_vs_bilateral": cluster_bootstrap_difference(rows, "fitted_mdp", "bilateral_once", "psnr"),
            "mdp_vs_noise_label_oracle": cluster_bootstrap_difference(rows, "fitted_mdp", "noise_label_oracle", "psnr"),
            "mdp_vs_oracle": cluster_bootstrap_difference(rows, "fitted_mdp", "reference_aware_greedy_oracle", "psnr"),
        },
        "paired_ssim": {
            "mdp_vs_median": cluster_bootstrap_difference(rows, "fitted_mdp", "median_once", "ssim"),
            "mdp_vs_bilateral": cluster_bootstrap_difference(rows, "fitted_mdp", "bilateral_once", "ssim"),
            "mdp_vs_noise_label_oracle": cluster_bootstrap_difference(rows, "fitted_mdp", "noise_label_oracle", "ssim"),
            "mdp_vs_oracle": cluster_bootstrap_difference(rows, "fitted_mdp", "reference_aware_greedy_oracle", "ssim"),
        },
        "per_case": rows,
    }
    result_path = result_dir / "mdp_benchmark.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": payload["summary"], "paired_psnr": payload["paired_psnr"],
                      "paired_ssim": payload["paired_ssim"]}, indent=2))
    print(f"Wrote {model_path.relative_to(ROOT)} and {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
