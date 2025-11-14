---
title: Overview
---

[Tutorial Hub](../index.md) | [Augmentation Playground](../augmentation-playground.md)

# Augmentation Overview


All three training stacks (scalar, vectorized, convolutional) now consume a shared configuration from `common/augment.py`. That means every CLI exposes the same switches—once you learn a flag, you can use it everywhere.

## Core Flags

| Flag | Meaning | Notes |
| --- | --- | --- |
| `--augment-max-shift` | Pixel jitter radius | Applies before other ops. |
| `--augment-rotate-deg` / `--augment-rotate-prob` | Max rotation + probability | Small angles preserve MNIST semantics. |
| `--augment-hflip-prob` / `--augment-vflip-prob` | Flip probabilities | Use carefully for digits. |
| `--augment-noise-std` / `--augment-noise-prob` / `--augment-noise-clip` | Additive Gaussian noise | Works best on normalized data. |
| `--augment-cutout-prob` / `--augment-cutout-size` | Random occlusion masks | Caution on small digits. |
| `--augment-cutmix-prob` / `--augment-cutmix-alpha` | CutMix patches + label mixing | Automatically disabled by the scalar trainer. |
| `--augment-randaug-layers` / `--augment-randaug-magnitude` | RandAugment policy layers | Randomly stacks ops using the shared magnitude. |
| `--no-augment` | Disables everything | Handy for ablations/tests. |

## One-Liner Cookbook

```bash
# Scalar (no CutMix, but full geometric/noise stack)
python3 scalar/main.py --epochs 2 --batch-size 64 \
  --augment-rotate-deg 12 --augment-rotate-prob 0.6 \
  --augment-cutout-prob 0.25 --augment-cutout-size 5 \
  --augment-randaug-layers 1 --augment-randaug-magnitude 0.4

# Vectorized MLP with CutMix + RandAugment
python3 vectorized/main.py --epochs 2 --batch-size 64 \
  --augment-cutmix-prob 0.3 --augment-cutmix-alpha 0.8 \
  --augment-randaug-layers 2 --augment-randaug-magnitude 0.5 \
  --augment-noise-std 0.03 --augment-cutout-prob 0.3

# Convolutional trainer (baseline CNN)
python3 convolutional/main.py baseline --epochs 2 --batch-size 64 \
  --augment-hflip-prob 0.4 --augment-cutmix-prob 0.3 \
  --augment-randaug-layers 2 --augment-randaug-magnitude 0.5
```

Each command exercises the exact same augmentation backend; only the model/optimizer stacks differ.

## Suggested Reading

- [Geometry Ops (shift/rotate/flip)](geometry.md)
- [Noise + Cutout](noise-and-cutout.md)
- [CutMix + RandAugment](cutmix-and-randaugment.md)

Keep the [Augmentation Playground](../augmentation-playground.md) open for quick experiments and lab ideas.
