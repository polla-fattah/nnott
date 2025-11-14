---
title: Noise & Cutout
---

Two powerful regularizers live in `common/augment.py`: additive Gaussian noise and cutout masks. Both are dataset-friendly when tuned carefully.

## Gaussian Noise

- `--augment-noise-std`: standard deviation of the noise sample.  
- `--augment-noise-prob`: probability of applying noise per image.  
- `--augment-noise-clip`: clamp range after noise injection (helps when data are standardized).

Sample run (vectorized trainer):

```bash
python3 vectorized/main.py --epochs 1 --batch-size 64 \
  --augment-noise-std 0.04 --augment-noise-prob 0.6 \
  --augment-noise-clip 3.0
```

Tips:

- Standardize inputs first (the default `DataUtility.load_data()` already zero-centers and scales MNIST images).
- Use `plot_loss` to monitor whether noise slows convergence; reduce `noise_prob` if the model struggles early on.

## Cutout

- `--augment-cutout-prob` toggles random occlusion per sample.  
- `--augment-cutout-size` measures the square patch length in pixels.  
- The helper zeros out a square centered at a random coordinate, clamped to image boundaries.

Scalar trainer example:

```bash
python3 scalar/main.py --epochs 2 --batch-size 64 \
  --augment-cutout-prob 0.3 --augment-cutout-size 5 \
  --augment-noise-std 0.03 --augment-noise-prob 0.5
```

Convolutional trainer example (using both digits-safe settings):

```bash
python3 convolutional/main.py baseline --epochs 2 --batch-size 64 \
  --augment-cutout-prob 0.25 --augment-cutout-size 4 \
  --augment-noise-std 0.02 --augment-noise-prob 0.4
```

Practical advice:

1. Start with small cutout sizes (3–5 pixels for MNIST) so digits remain identifiable.
2. When training on larger datasets (e.g., CIFAR-10), gradually increase both size and probability.
3. Combine with `--no-augment` baseline runs to quantify the benefit in your experiment logs.
