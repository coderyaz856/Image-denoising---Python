"""Filter-search image denoising with PSNR/SSIM/MSE evaluation.

Two ways to use this module:

* ``--clean IMAGE``  benchmark mode. Synthetic noise is added to a clean image,
  the filter search runs on the noisy version, and every metric is measured
  against the original clean image, which acts as ground truth.
* ``--gui``          interactive mode. Opens a small Tk window, runs the search
  on an image you pick, and displays the result.
"""

import argparse
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

MAX_PIXEL_VALUE = 255


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------

def operation_gaussian_blur(image, ksize=(3, 3), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)


def operation_median_blur(image, ksize=3):
    return cv2.medianBlur(image, ksize)


def operation_bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def operation_box_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)


def operation_non_local_means(image, h=10, template_window=7, search_window=21):
    if image.ndim == 2:
        return cv2.fastNlMeansDenoising(image, None, h, template_window, search_window)
    return cv2.fastNlMeansDenoisingColored(
        image, None, h, h, template_window, search_window
    )


def operation_sharpen(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)


def apply_high_pass_filter(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def apply_laplacian_filter(image):
    return cv2.convertScaleAbs(cv2.Laplacian(image, cv2.CV_64F))


def apply_sobel_filter(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.addWeighted(
        cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0
    )


def apply_threshold(image):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if image.ndim == 3:
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    return thresh


def apply_canny_edge_detection(image):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    if image.ndim == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges


#: Noise-reducing filters, searched over by default.
DENOISING_OPERATIONS = [
    operation_gaussian_blur,
    operation_median_blur,
    operation_bilateral_filter,
    operation_box_filter,
    operation_non_local_means,
]

#: Edge and contrast operators. Available, but excluded from the denoising
#: search because they amplify noise rather than suppress it.
ENHANCEMENT_OPERATIONS = [
    operation_sharpen,
    apply_high_pass_filter,
    apply_laplacian_filter,
    apply_sobel_filter,
    apply_threshold,
    apply_canny_edge_detection,
]


# --------------------------------------------------------------------------
# Noise model
# --------------------------------------------------------------------------

def add_gaussian_noise(image, sigma=25.0, seed=None):
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float64) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, MAX_PIXEL_VALUE).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.05, seed=None):
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    draw = rng.random(image.shape[:2])
    noisy[draw < amount / 2] = 0
    noisy[draw > 1 - amount / 2] = MAX_PIXEL_VALUE
    return noisy


NOISE_MODELS = {"gaussian": add_gaussian_noise, "salt_pepper": add_salt_pepper_noise}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def calculate_quality_metrics(reference, candidate):
    """PSNR, SSIM and MSE of ``candidate`` measured against ``reference``."""
    psnr_value = cv2.PSNR(reference, candidate)

    height, width = reference.shape[:2]
    win_size = min(7, height, width)
    if win_size % 2 == 0:
        win_size -= 1
    win_size = max(win_size, 3)

    ssim_kwargs = {"data_range": MAX_PIXEL_VALUE, "win_size": win_size}
    if reference.ndim == 3:
        # scikit-image renamed `multichannel` to `channel_axis` in 0.19 and
        # removed the old name in 0.23; support both.
        try:
            ssim_value = ssim(reference, candidate, channel_axis=-1, **ssim_kwargs)
        except TypeError:
            ssim_value = ssim(reference, candidate, multichannel=True, **ssim_kwargs)
    else:
        ssim_value = ssim(reference, candidate, **ssim_kwargs)

    # Cast before subtracting: uint8 arithmetic wraps around on negatives.
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    mse_value = float(np.mean(diff ** 2))

    return {"psnr": float(psnr_value), "ssim": float(ssim_value), "mse": mse_value}


def normalize_metric(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


PSNR_RANGE = (0.0, 100.0)
SSIM_RANGE = (0.0, 1.0)
MSE_RANGE = (0.0, float(MAX_PIXEL_VALUE ** 2))
DEFAULT_WEIGHTS = {"psnr": 1.0, "ssim": 1.0, "mse": 1.0}


def calculate_score(metrics, weights=None):
    """Weighted composite: reward PSNR and SSIM, penalise MSE."""
    weights = weights or DEFAULT_WEIGHTS
    # An identical pair gives infinite PSNR; cap it so the score stays finite.
    psnr = min(metrics["psnr"], PSNR_RANGE[1])
    return (
        weights["psnr"] * normalize_metric(psnr, *PSNR_RANGE)
        + weights["ssim"] * normalize_metric(metrics["ssim"], *SSIM_RANGE)
        - weights["mse"] * normalize_metric(metrics["mse"], *MSE_RANGE)
    )


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------

def optimize_image_denoising(
    noisy_image,
    reference_image=None,
    operations=None,
    max_iterations=5,
    weights=None,
    verbose=True,
):
    """Greedy hill-climbing over a set of filters.

    Each iteration applies every candidate filter to the current image and
    keeps the highest-scoring result. The search stops when no filter improves
    on the current score, or when ``max_iterations`` is reached.

    ``reference_image`` is what metrics are measured against. Pass the clean
    image for a true denoising benchmark; when it is omitted the noisy input is
    used, which only measures how little a filter alters its input.
    """
    operations = operations or DENOISING_OPERATIONS
    reference = noisy_image if reference_image is None else reference_image

    current_image = noisy_image.copy()
    current_metrics = calculate_quality_metrics(reference, current_image)
    current_score = calculate_score(current_metrics, weights)
    history = [{"iteration": 0, "operation": "input", **current_metrics,
                "score": current_score}]

    if verbose:
        print(f"Start   PSNR={current_metrics['psnr']:.2f} "
              f"SSIM={current_metrics['ssim']:.4f} "
              f"MSE={current_metrics['mse']:.2f} score={current_score:.4f}")

    for iteration in range(1, max_iterations + 1):
        best_score = -float("inf")
        best_image = None
        best_metrics = None
        best_name = None

        for operation in operations:
            candidate = operation(current_image)
            metrics = calculate_quality_metrics(reference, candidate)
            score = calculate_score(metrics, weights)

            if verbose:
                print(f"  iter {iteration} {operation.__name__:<30} "
                      f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} "
                      f"MSE={metrics['mse']:.2f} score={score:.4f}")

            if score > best_score:
                best_score, best_image = score, candidate
                best_metrics, best_name = metrics, operation.__name__

        # Compare against the score carried into this iteration, not against
        # the best candidate of this iteration.
        if best_score <= current_score:
            if verbose:
                print(f"  iter {iteration}: no improvement, stopping.")
            break

        current_image, current_metrics, current_score = (
            best_image, best_metrics, best_score
        )
        history.append({"iteration": iteration, "operation": best_name,
                        **best_metrics, "score": best_score})
        if verbose:
            print(f"  iter {iteration}: accepted {best_name} "
                  f"(score={best_score:.4f})")

    return current_image, history


# --------------------------------------------------------------------------
# Benchmark
# --------------------------------------------------------------------------

def run_benchmark(clean_path, noise="gaussian", sigma=25.0, amount=0.05,
                  max_iterations=5, seed=0, output=None):
    clean = cv2.imread(clean_path)
    if clean is None:
        raise FileNotFoundError(f"Could not read image: {clean_path}")

    if noise == "gaussian":
        noisy = add_gaussian_noise(clean, sigma=sigma, seed=seed)
    else:
        noisy = add_salt_pepper_noise(clean, amount=amount, seed=seed)

    print(f"Image: {clean_path}  ({clean.shape[1]}x{clean.shape[0]})")
    print(f"Noise: {noise}"
          + (f", sigma={sigma}" if noise == "gaussian" else f", amount={amount}"))

    baseline = calculate_quality_metrics(clean, noisy)
    print(f"\nNoisy input vs clean: PSNR={baseline['psnr']:.2f} dB  "
          f"SSIM={baseline['ssim']:.4f}  MSE={baseline['mse']:.2f}")

    print("\nSingle-pass filter comparison (each applied to the noisy image):")
    print(f"  {'filter':<32}{'PSNR (dB)':>11}{'SSIM':>9}{'MSE':>11}")
    for operation in DENOISING_OPERATIONS:
        metrics = calculate_quality_metrics(clean, operation(noisy))
        print(f"  {operation.__name__:<32}{metrics['psnr']:>11.2f}"
              f"{metrics['ssim']:>9.4f}{metrics['mse']:>11.2f}")

    print("\nGreedy filter search:")
    best_image, history = optimize_image_denoising(
        noisy, reference_image=clean, max_iterations=max_iterations
    )

    final = history[-1]
    print(f"\nSelected chain: {' -> '.join(h['operation'] for h in history[1:]) or 'none'}")
    print(f"Final: PSNR={final['psnr']:.2f} dB  SSIM={final['ssim']:.4f}  "
          f"MSE={final['mse']:.2f}")
    print(f"PSNR gain over noisy input: {final['psnr'] - baseline['psnr']:+.2f} dB")

    if output:
        cv2.imwrite(output, best_image)
        print(f"Wrote denoised image to {output}")

    return best_image, history


# --------------------------------------------------------------------------
# GUI
# --------------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, ttk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Denoising Tool")
    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12))

    notebook = ttk.Notebook(root)
    notebook.pack(pady=10, padx=10)

    tab_file = tk.Frame(notebook)
    tab_about = tk.Frame(notebook)
    notebook.add(tab_file, text="File")
    notebook.add(tab_about, text="About")

    tk.Label(
        tab_about,
        text="Applies a greedy search over OpenCV denoising filters and shows\n"
             "the highest-scoring result. Metrics are measured against the\n"
             "image you open, so use the command line benchmark for results\n"
             "measured against a clean ground truth.",
        bg="#ffffff",
        justify="left",
    ).pack(padx=10, pady=10)

    image_label = tk.Label(tab_file, bg="#ffffff")
    image_label.pack(padx=10, pady=10)
    status = tk.Label(tab_file, text="No image loaded", bg="#f0f0f0")
    status.pack(pady=4)

    def open_image():
        filepath = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not filepath:
            return
        image = cv2.imread(filepath)
        if image is None:
            status.config(text="Could not read that file")
            return

        denoised, history = optimize_image_denoising(image, verbose=False)
        chain = " -> ".join(h["operation"] for h in history[1:]) or "none"

        photo = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
        )
        image_label.config(image=photo)
        image_label.image = photo  # keep a reference alive
        status.config(text=f"Applied: {chain}")

    ttk.Button(tab_file, text="Open Image", command=open_image).pack(pady=10)
    ttk.Button(tab_file, text="Exit", command=root.destroy).pack(pady=5)

    root.mainloop()


# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clean", metavar="IMAGE",
                        help="clean image to benchmark against")
    parser.add_argument("--noise", choices=sorted(NOISE_MODELS), default="gaussian")
    parser.add_argument("--sigma", type=float, default=25.0,
                        help="gaussian noise standard deviation (default 25)")
    parser.add_argument("--amount", type=float, default=0.05,
                        help="salt-and-pepper corruption fraction (default 0.05)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="maximum search iterations (default 5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="noise seed, for reproducible runs (default 0)")
    parser.add_argument("--output", metavar="IMAGE",
                        help="write the denoised image here")
    parser.add_argument("--gui", action="store_true", help="launch the Tk interface")
    args = parser.parse_args(argv)

    if args.gui or not args.clean:
        if not args.gui:
            parser.print_help()
            print("\nNo --clean image given; launching the GUI.")
        launch_gui()
        return 0

    run_benchmark(args.clean, noise=args.noise, sigma=args.sigma,
                  amount=args.amount, max_iterations=args.iterations,
                  seed=args.seed, output=args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
