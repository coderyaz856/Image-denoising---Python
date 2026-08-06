# Image Denoising — Filter Search with PSNR/SSIM Evaluation

A small image-denoising tool that searches over classical OpenCV filters and picks the one that best restores a noisy image, scored with a weighted combination of PSNR, SSIM and MSE.

Status: course project, runnable from the command line or a small Tk interface.

## Problem

Different noise types call for different filters. A median filter handles salt-and-pepper corruption well but wastes detail on Gaussian noise; a bilateral filter is the reverse. Rather than choosing by hand, this project treats filter selection as a **greedy search problem**: at each step, try every candidate filter, keep whichever scores highest, and repeat until no filter improves the score.

The difficulty is defining that score. Measuring a filtered image against the *noisy input* only shows how little the filter changed it — which rewards doing nothing. So the benchmark mode starts from a **clean image**, adds synthetic noise to it, and measures every candidate against the clean original, which serves as ground truth.

## Approach

```
clean image ──► add synthetic noise ──► noisy image
     │                                       │
     │                                       ▼
     │                            greedy filter search
     │                          (try each filter, keep best)
     │                                       │
     └──────────► PSNR / SSIM / MSE ◄────────┘
                   ground truth
```

Each candidate is scored with

```
score = w_psnr · norm(PSNR) + w_ssim · SSIM − w_mse · norm(MSE)
```

where PSNR is normalised over 0–100 dB and MSE over 0–255². The search accepts a filter only if it beats the score carried into the current iteration, so filters can be chained across iterations (applying the bilateral filter several times, for instance) and the loop terminates as soon as it stops gaining.

### Filters searched

`GaussianBlur` · `medianBlur` · `bilateralFilter` · 5×5 box filter · fast non-local means

Sharpening, high-pass, Laplacian, Sobel, Otsu threshold and Canny are also implemented and exported as `ENHANCEMENT_OPERATIONS`, but are deliberately excluded from the denoising search — they amplify noise rather than suppress it.

### Noise models

Additive Gaussian (configurable σ) and salt-and-pepper (configurable corruption fraction). Both take a seed, so runs are reproducible.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. The GUI additionally needs Tk, which ships with most CPython installs.

## Usage

Benchmark against a clean image:

```bash
python image_denoising.py --clean path/to/image.png --noise gaussian --sigma 25
python image_denoising.py --clean path/to/image.png --noise salt_pepper --amount 0.08
```

Useful flags: `--iterations` (search budget, default 5), `--seed` (noise seed, default 0), `--output` (write the denoised image to a file).

Interactive mode:

```bash
python image_denoising.py --gui
```

The GUI runs the same search on an image you pick and reports the filter chain it applied. Because it has no clean reference, its metrics are measured against the opened image — use the command-line benchmark for meaningful numbers.

## Results

The search recovers the appropriate filter for each noise type without being told which. Measured on a 240×160 synthetic test pattern (your numbers will vary by image):

**Gaussian noise, σ = 25** — noisy input at 20.60 dB / SSIM 0.29

| Filter | PSNR (dB) | SSIM |
| --- | --- | --- |
| Gaussian blur | 25.23 | 0.5668 |
| Median blur | 26.72 | 0.5384 |
| **Bilateral filter** | **31.36** | **0.7407** |
| Box filter | 22.49 | 0.7199 |
| Non-local means | 27.02 | 0.7283 |

Selected chain: bilateral × 4 → **36.65 dB / SSIM 0.9846**, a gain of **+16.05 dB** over the noisy input.

**Salt-and-pepper, 8% corruption** — noisy input at 15.69 dB / SSIM 0.25

| Filter | PSNR (dB) | SSIM |
| --- | --- | --- |
| Gaussian blur | 22.14 | 0.4245 |
| **Median blur** | **31.20** | **0.9835** |
| Bilateral filter | 15.74 | 0.2595 |
| Box filter | 21.49 | 0.5792 |
| Non-local means | 16.92 | 0.2977 |

The ordering flips exactly as theory predicts: bilateral filtering is the strongest choice for Gaussian noise and among the weakest for impulse noise, where the median filter dominates.

## Repository structure

```
image_denoising.py    Filters, noise models, metrics, search, CLI and GUI
requirements.txt      Dependencies
```

Key entry points: `optimize_image_denoising()` for the search, `calculate_quality_metrics()` for PSNR/SSIM/MSE, and `run_benchmark()` for the full clean-versus-noisy comparison.

## Scope

This is a search over classical filters, not a learned denoiser — there is no training and no comparison against neural baselines such as DnCNN. The composite score weights PSNR, SSIM and MSE equally by default; other weightings will select different filters, and the weights are exposed as a parameter rather than tuned. The synthetic noise models are independent per pixel, which is a simplification of real sensor noise.

## Technologies

Python · OpenCV · NumPy · scikit-image · Pillow · Tkinter
