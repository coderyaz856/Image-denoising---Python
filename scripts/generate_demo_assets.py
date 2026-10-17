"""Generate the deterministic benchmark figure and its machine-readable data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import skimage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from image_denoising import (  # noqa: E402
    DENOISING_OPERATIONS,
    add_gaussian_noise,
    add_salt_pepper_noise,
    calculate_quality_metrics,
    optimize_image_denoising,
)


ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results"
SEED = 7


def make_reference(width: int = 480, height: int = 320) -> np.ndarray:
    """Build a synthetic scene with gradients, edges, text, and fine detail."""
    x = np.linspace(0, 1, width, dtype=np.float32)
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]

    blue = np.broadcast_to(35 + 135 * x, (height, width))
    green = np.broadcast_to(45 + 120 * y, (height, width))
    red = 55 + 90 * ((x[None, :] + y) / 2)
    image = np.dstack((blue, green, red)).astype(np.uint8)

    cv2.rectangle(image, (28, 30), (208, 142), (28, 35, 46), -1)
    cv2.rectangle(image, (43, 45), (193, 127), (232, 241, 246), 3)
    cv2.circle(image, (330, 92), 61, (40, 177, 235), -1)
    cv2.circle(image, (330, 92), 35, (245, 247, 250), 5)
    cv2.line(image, (24, 205), (455, 205), (248, 248, 248), 4)

    for index in range(10):
        x0 = 28 + index * 22
        color = (235, 235, 235) if index % 2 == 0 else (45, 50, 60)
        cv2.rectangle(image, (x0, 229), (x0 + 12, 282), color, -1)

    cv2.putText(
        image,
        "DENOISE",
        (54, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.84,
        (35, 56, 78),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "edges  gradients  texture",
        (250, 266),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return image


def rounded(metrics: Dict[str, float]) -> Dict[str, float]:
    return {name: round(float(value), 6) for name, value in metrics.items()}


def describe_case(
    name: str,
    clean: np.ndarray,
    noisy: np.ndarray,
    config: Dict[str, Union[float, int, str]],
) -> Tuple[np.ndarray, Dict[str, object]]:
    denoised, history = optimize_image_denoising(
        noisy,
        reference_image=clean,
        max_iterations=5,
        verbose=False,
    )
    noisy_metrics = calculate_quality_metrics(clean, noisy)
    final_metrics = calculate_quality_metrics(clean, denoised)

    single_pass = {}
    for operation in DENOISING_OPERATIONS:
        single_pass[operation.__name__] = rounded(
            calculate_quality_metrics(clean, operation(noisy))
        )

    result = {
        "noise": name,
        "config": config,
        "input": rounded(noisy_metrics),
        "selected_chain": [step["operation"] for step in history[1:]],
        "output": rounded(final_metrics),
        "psnr_gain_db": round(final_metrics["psnr"] - noisy_metrics["psnr"], 6),
        "single_pass": single_pass,
    }
    return denoised, result


def labeled_panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
    width: int = 360,
) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(
        resized, 68, 12, 12, 12, cv2.BORDER_CONSTANT, value=(18, 24, 33)
    )
    cv2.putText(
        panel, title, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
        (246, 248, 250), 2, cv2.LINE_AA
    )
    cv2.putText(
        panel, subtitle, (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
        (158, 174, 192), 1, cv2.LINE_AA
    )
    return panel


def metric_caption(metrics: Dict[str, float]) -> str:
    return f"PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.4f}"


def summarize_chain(chain: List[str]) -> str:
    labels = {
        "operation_gaussian_blur": "Gaussian",
        "operation_median_blur": "median",
        "operation_bilateral_filter": "bilateral",
        "operation_box_filter": "box",
        "operation_non_local_means": "non-local means",
    }
    compressed = []
    for operation in chain:
        label = labels.get(operation, operation)
        if compressed and compressed[-1][0] == label:
            compressed[-1][1] += 1
        else:
            compressed.append([label, 1])
    return " -> ".join(
        f"{label} x{count}" if count > 1 else label
        for label, count in compressed
    ) or "no filter accepted"


def main() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    RESULT_DIR.mkdir(exist_ok=True)

    clean = make_reference()
    cases = [
        (
            "gaussian",
            add_gaussian_noise(clean, sigma=25.0, seed=SEED),
            {"sigma": 25.0, "seed": SEED},
        ),
        (
            "salt_pepper",
            add_salt_pepper_noise(clean, amount=0.08, seed=SEED),
            {"amount": 0.08, "seed": SEED},
        ),
    ]

    rows = []
    results = []
    for display_name, noisy, config in cases:
        denoised, result = describe_case(display_name, clean, noisy, config)
        results.append(result)

        label = "Gaussian noise" if display_name == "gaussian" else "Salt-and-pepper"
        chain = summarize_chain(result["selected_chain"])
        row = np.hstack(
            [
                labeled_panel(clean, "Reference", "deterministic synthetic scene"),
                labeled_panel(noisy, label, metric_caption(result["input"])),
                labeled_panel(
                    denoised,
                    "Selected result",
                    f"{metric_caption(result['output'])} | {chain}",
                ),
            ]
        )
        rows.append(row)

    figure = np.vstack(rows)
    figure_path = ASSET_DIR / "benchmark_overview.png"
    if not cv2.imwrite(str(figure_path), figure):
        raise RuntimeError(f"Could not write {figure_path}")

    payload = {
        "benchmark": "deterministic synthetic image",
        "image_size": {"width": clean.shape[1], "height": clean.shape[0]},
        "search_iterations": 5,
        "scoring": "equal-weight normalized PSNR + SSIM - MSE",
        "environment": {
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_image": skimage.__version__,
        },
        "cases": results,
    }
    metrics_path = RESULT_DIR / "demo_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {figure_path.relative_to(ROOT)}")
    print(f"Wrote {metrics_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
