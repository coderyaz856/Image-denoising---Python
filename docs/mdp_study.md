# Reference-free sequential denoising with a fitted MDP policy

This experiment asks whether a policy can choose and chain classical filters
without seeing a clean reference image at inference time. It extends, but does
not replace, the original ground-truth-guided greedy search.

## Decision problem

One decision step is one filtering operation on a static image. The noise
process is **not** assumed to change over time. The policy has a horizon of
three operations and an explicit stop action.

- Observation: seven image-only features (mean and standard deviation of
  grayscale intensity; near-black/white fraction; median and Gaussian
  residuals; Laplacian energy; edge fraction). The feature vector of the
  original noisy image is retained alongside the current image's features.
- Actions: Gaussian, median, bilateral, box and non-local means denoisers, or
  stop. Parameters are the repository's existing fixed filter parameters.
- Training reward for filter action `a`: `score(next image, clean reference)
  - score(current image, clean reference) - 0.01`. Stop returns zero. The
  score is the repository's fixed PSNR/SSIM/MSE composite. The 0.01 term is
  a constant **action-count penalty**, not a measured latency cost.
- Training: enumerate all five filter outcomes from each sampled state, then
  use backward fitted Q iteration. Seven-nearest-neighbor regression
  approximates each stage's action values. One rollout path follows an
  immediate-reward oracle; another explores with a fixed random seed.
- Inference: the saved policy receives only a noisy image. It predicts
  action values from observable features, stops if no action has positive
  estimated value, and never calls a reference-dependent metric.

This is a finite-horizon **approximate MDP** with a compressed observation
state. The feature map can alias different underlying images or noise
conditions, so a formal Markov property is not established. It is not an
exact POMDP solution, HMM, DQN or trilevel optimization method.

## Evaluation protocol

The training and held-out sets are split by original `skimage.data` source
image, not by crop. Training uses astronaut, camera, coins and grass; testing
uses chelsea, coffee, moon and brick. Each source supplies three deterministic
80 x 80 crops. Training covers Gaussian sigma 15/25 and salt-and-pepper
amount 0.04/0.08 with seeds 11/23; testing uses distinct seeds 101/211 and
adds stronger unseen levels, sigma 35 and amount 0.12. The full benchmark has
96 training and 144 held-out corrupted crops. The JSON artifact records
source-image hashes, library versions, every case and every filter chain.

For the fitted policy and fixed filters, test references are used **only** for
scoring after an answer is produced. The comparison includes unchanged noisy input, fixed median and
bilateral filters, the learned policy, a selector given the synthetic noise
label, and the existing greedy method. The label selector knows whether the
corruption is Gaussian or salt-and-pepper, which a blind user would not know.
The greedy method reads the clean reference for every candidate action.
These two oracles indicate headroom, not fair deployable competitors.

The primary metrics are PSNR, SSIM and action count. Per-case elapsed time is
recorded but is machine-dependent; a constant action penalty should not be
confused with runtime optimization. Paired 95% bootstrap intervals resample
original test source images, rather than treating correlated crops as
independent. With only four source clusters, these intervals are descriptive.

## Results

| Method | Mean PSNR | Mean SSIM | Mean actions |
| --- | ---: | ---: | ---: |
| Noisy input | 18.80 dB | 0.3764 | 0.00 |
| Median once | 28.47 dB | 0.7634 | 1.00 |
| Bilateral once | 22.31 dB | 0.5034 | 1.00 |
| Noise-label oracle | 29.46 dB | 0.8086 | 1.00 |
| Fitted MDP policy | 28.67 dB | 0.7933 | 1.22 |
| Reference-aware greedy oracle | 30.33 dB | 0.8607 | 1.43 |

"Actions" counts applied or accepted filters. The reference-aware greedy
oracle also evaluates rejected candidate filters, so its accepted-action
count does not represent its computation cost.

Across all 144 held-out cases, the policy averaged +0.20 dB PSNR and +0.030
SSIM against a single median pass. Its PSNR cluster-bootstrap interval
was [-0.61, 0.85] dB; the SSIM interval was [-0.0096, 0.0717]. Both include
zero, so this is **not** evidence of reliable superiority over median.
It remains 1.66 dB below the reference-aware oracle on average. The policy
is also 0.79 dB below the noise-label oracle; that paired interval is
[-1.30, -0.29] dB over the four held-out source clusters. The policy
adapts its action mix to corruption type, but it does not dominate the best
single filter for each known type. In particular, performance weakens on the
unseen stronger Gaussian level; across both unseen stronger levels it averaged
26.48 dB versus 26.54 dB for median. Consult `results/mdp_benchmark.json` for
noise-stratified results and paired intervals.

## Reproduction

```bash
pip install -r requirements.txt
python scripts/benchmark_mdp.py
python -m unittest discover -s tests -v
python mdp_denoising.py --model results/mdp_policy.npz \
  --input path/to/noisy.png --output path/to/restored.png
```

`benchmark_mdp.py` regenerates `results/mdp_policy.npz` and
`results/mdp_benchmark.json`. Quality values and action chains are
deterministic for the recorded library versions. Two complete local runs
produced byte-identical model files and identical quality metrics, chains,
intervals and provenance when timing fields were excluded. Timing values
varied; `scripts/verify_mdp_evidence.py` can compare two saved JSON runs.

## Limitations and next experiment

Four training and four test images, correlated crops, and synthetic independent
noise do not establish broad generalization to photographs, sensors or unseen
corruptions. The finite feature map is hand-designed and may miss perceptual
structure. A stronger next study would reserve more independent source images,
compare against a simple noise-regime classifier that chooses among fixed
filters, and evaluate mixed/real noise with confidence intervals fixed before
model changes. HMM-guided DQN or a POMDP approximation should be attempted
only if that simpler baseline reveals a reproducible uncertainty gap.
