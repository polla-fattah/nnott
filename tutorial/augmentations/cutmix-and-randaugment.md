[Overview](overview.md) | [Tutorial Hub](../README.md)

# CutMix & RandAugment

**Breadcrumb:** [Home](../README.md) / [Augmentation Playground](../augmentation-playground.md) / CutMix & RandAugment

Advanced policies such as CutMix and RandAugment are now first-class citizens inside `common/augment.py`. This page explains how to configure and validate them.

## CutMix

- Flags: `--augment-cutmix-prob`, `--augment-cutmix-alpha`.
- Logic:
  1. Sample two images and draw a random rectangle sized according to `lambda ~ Beta(alpha, alpha)`.
  2. Copy the rectangle from the second image into the first.
  3. Mix labels using `λ * y_a + (1-λ) * y_b`. Trainers compute a weighted sum of losses/gradients automatically.
- The scalar trainer automatically disables CutMix (`allow_label_mix=False`) to keep its simple per-sample label flow intact.

Vectorized trainer example:

```bash
python3 vectorized/main.py --epochs 3 --batch-size 64 \
  --augment-cutmix-prob 0.4 --augment-cutmix-alpha 0.7 \
  --augment-noise-std 0.02 --augment-cutout-prob 0.2
```

Convolutional trainer example:

```bash
python3 convolutional/main.py resnet18 --epochs 3 --batch-size 64 \
  --augment-cutmix-prob 0.35 --augment-cutmix-alpha 1.0 \
  --augment-randaug-layers 1 --augment-randaug-magnitude 0.4
```

Validation tips:

- Log both `λ` values and sample visualizations occasionally to ensure the bounding boxes make sense.
- Combine with `--show-misclassified` to inspect whether CutMix is making labels ambiguous.

## RandAugment

- Flags: `--augment-randaug-layers`, `--augment-randaug-magnitude`.
- Implementation:
  - For each layer, pick one op from `{rotate, shift_x, shift_y, flip_lr, flip_ud, noise, cutout}` at random.
  - Apply it using the provided magnitude scaled into the op’s natural units (degrees, pixels, probability, etc.).
- Works on all trainers; magnitude ∈ [0, 1].

Cookbook command (shared policy across stacks):

```bash
python3 scalar/main.py --epochs 2 --batch-size 64 \
  --augment-randaug-layers 1 --augment-randaug-magnitude 0.4 --augment-cutout-prob 0.2

python3 vectorized/main.py --epochs 2 --batch-size 64 \
  --augment-randaug-layers 2 --augment-randaug-magnitude 0.5 \
  --augment-cutmix-prob 0.3 --augment-cutmix-alpha 0.9

python3 convolutional/main.py baseline --epochs 2 --batch-size 64 \
  --augment-randaug-layers 2 --augment-randaug-magnitude 0.5
```

Practical advice:

1. Start with one RandAugment layer on MNIST-class tasks; more layers may over-randomize digits.
2. Combine with deterministic flags (e.g., keep `--augment-rotate-deg` low while RandAugment explores additional rotations).
3. Always benchmark against `--no-augment` to quantify the gain.
