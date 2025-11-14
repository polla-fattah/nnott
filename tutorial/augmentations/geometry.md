[Overview](overview.md) | [Tutorial Hub](../README.md)

# Geometric Augmentations

**Breadcrumb:** [Home](../README.md) / [Augmentation Playground](../augmentation-playground.md) / Geometry

This page covers the spatial transforms supported by `common/augment.py`: pixel shifts, rotations, and flips.

## Pixel Shifts

- Controlled via `--augment-max-shift` (integer radius).  
- Images are rolled along height/width before any other op.
- Shifts are label-safe for centered digits and general-purpose image datasets.

Example:

```bash
python3 vectorized/main.py --epochs 1 --batch-size 64 \
  --augment-max-shift 3 --augment-rotate-deg 0 \
  --augment-hflip-prob 0 --augment-vflip-prob 0
```

## Rotations

- `--augment-rotate-deg` sets the maximum absolute angle.
- `--augment-rotate-prob` gates how often we rotate.
- The shared helper uses nearest-neighbor sampling per channel and works on CPU or GPU transparently.

Validation tip: start with small angles (≤12°) on MNIST to avoid turning a “6” into a “9”. When experimenting with natural images, combine rotations with flips to simulate viewpoint changes.

CLI example across all trainers:

```bash
python3 scalar/main.py --epochs 1 --batch-size 64 \
  --augment-rotate-deg 12 --augment-rotate-prob 0.7 \
  --augment-max-shift 2 --augment-hflip-prob 0

python3 vectorized/main.py --epochs 1 --batch-size 64 \
  --augment-rotate-deg 15 --augment-rotate-prob 0.5

python3 convolutional/main.py baseline --epochs 1 --batch-size 64 \
  --augment-rotate-deg 20 --augment-rotate-prob 0.5
```

## Flips

- `--augment-hflip-prob` mirrors across the vertical axis.
- `--augment-vflip-prob` mirrors across the horizontal axis.
- Useful for natural images (CIFAR, ImageNet) where mirrored scenes keep the same label.
- **Digits caution:** flipping can change semantics (mirrored “2” isn’t a valid “2”). For MNIST, keep flip probabilities at 0 unless running an ablation.

Mixed example (safe for CIFAR-style data):

```bash
python3 convolutional/main.py resnet18 --epochs 2 --batch-size 128 \
  --augment-hflip-prob 0.5 --augment-vflip-prob 0.1 \
  --augment-rotate-deg 5 --augment-rotate-prob 0.3
```

Remember to visualize augmented samples using the plotting utilities before committing to a policy.
