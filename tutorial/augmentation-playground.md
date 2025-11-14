**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# Augmentation Playground

**Breadcrumb:** [Home](README.md) / Augmentation Playground


All three trainers now share the same augmentation stack via [`common/augment.py`](../common/augment.py). The CLI front-ends expose identical knobs (see the new [Augmentation Overview](augmentations/overview.md) plus detailed guides on [geometry](augmentations/geometry.md), [noise & cutout](augmentations/noise-and-cutout.md), and [CutMix/RandAugment](augmentations/cutmix-and-randaugment.md)).

```bash
# scalar MLP
python scalar/main.py --epochs 5 --augment-rotate-deg 12 --augment-cutout-prob 0.2

# vectorized MLP
python vectorized/main.py --epochs 2 --augment-cutmix-prob 0.5 --augment-cutmix-alpha 0.75

# convolutional CNNs
python convolutional/main.py resnet18 --augment-randaug-layers 2 --augment-randaug-magnitude 0.5
```

Every flag maps directly to the shared config (`max_shift`, `rotate_deg`, `noise_std`, `cutout_prob`, `cutmix_prob`, `randaugment_layers`, etc.). The scalar trainer disables CutMix internally to keep labels single-class, but it otherwise runs the same codepath as the higher-performance pipelines.

Use `--no-augment` on any entry point to run with raw data (useful for ablations or unit tests).

## 1. Understanding the Existing Augmentor

```python
def augment_image_batch(batch, cfg, xp_module=None, labels=None, allow_label_mix=True):
    if batch.ndim not in (3, 4):
        return batch, None
    ...
```

- Inputs can be `(N, C, H, W)` (conv/vectorized) or `(N, H, W)` (scalar) and the helper automatically reshapes as needed.
- Image ops (shift, rotate, flips, noise, cutout) run per-sample; CutMix optionally mixes labels if `allow_label_mix=True`.
- `augment_flat_batch` reshapes `(N, 784)` vectors into `28×28` grids for the vectorized trainer.

Under the hood the trainers call:

```python
xb, mix_meta = augment_image_batch(xb, cfg, xp_module=xp, labels=yb)
logits = model.forward(xb)
loss, grad = combine_losses(logits, yb, mix_meta)
```

- When `mix_meta` is `None`, the loss behaves like standard cross-entropy.
- When `mix_meta` contains CutMix metadata, both targets are blended during forward/backward passes.
- Scalar training requests `allow_label_mix=False`, so CutMix is skipped while other transforms still apply.

---

## 2. Adding Rotations (Small Angles)

For MNIST, minor rotations (±10°) keep label semantics intact. You can extend `_augment_batch`:

```python
import math

def _augment_batch(self, xb, max_shift=2, max_rotate_deg=10):
    if xb.ndim != 4:
        return xb
    shifted = xp.empty_like(xb)
    for i in range(len(xb)):
        dx = int(xp.random.randint(-max_shift, max_shift + 1))
        dy = int(xp.random.randint(-max_shift, max_shift + 1))
        img = xp.roll(xp.roll(xb[i], dy, axis=1), dx, axis=2)
        if max_rotate_deg > 0 and xp.random.rand() < 0.5:
            angle = float(xp.random.uniform(-max_rotate_deg, max_rotate_deg))
            img = rotate_nearest(img, math.radians(angle))
        shifted[i] = img
    return shifted

def rotate_nearest(img, angle):
    # xp-based nearest-neighbor rotation (adapted from scalar trainer for GPU support)
    C, H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = xp.meshgrid(xp.arange(H), xp.arange(W), indexing="ij")
    x0, y0 = xx - cx, yy - cy
    c, s = math.cos(angle), math.sin(angle)
    xr = c * x0 + s * y0 + cx
    yr = -s * x0 + c * y0 + cy
    xi = xp.clip(xp.rint(xr).astype(int), 0, W - 1)
    yi = xp.clip(xp.rint(yr).astype(int), 0, H - 1)
    out = xp.empty_like(img)
    for ch in range(C):
        out[ch] = img[ch, yi, xi]
    return out
```

**Warning:** Rotations beyond ~15° may distort digits into different classes (e.g., “6” vs “9”). Always sanity-check a few augmented images using `plot_image_grid`.

---

## 3. Horizontal/Vertical Flips (Use with Care)

- **Digits:** Horizontal flips can change semantics (“2” vs mirrored “2”). Avoid flips unless labels and dataset allow it.
- **Natural images:** For CIFAR-10 or larger datasets, flips often help. Implement by toggling axes:

```python
if xp.random.rand() < 0.5:
    img = img[:, :, ::-1]  # horizontal flip
if xp.random.rand() < 0.1:
    img = img[:, ::-1, :]  # vertical flip (rarely used for digits)
```

Always ask: *Does a flipped image still belong to the same class?* If not, skip the augmentation.

---

## 4. Additive Noise

```python
noise_std = 0.05
if noise_std > 0:
    img = img + noise_std * xp.random.randn(*img.shape).astype(img.dtype)
    img = xp.clip(img, -3, 3)  # keep standardized range reasonable
```

- Works well when data are standardized (mean≈0, std≈1). Adjust clipping bounds accordingly.

---

## 5. Cutout (Random Occlusion)

```python
def apply_cutout(img, size=5):
    C, H, W = img.shape
    x = xp.random.randint(0, W)
    y = xp.random.randint(0, H)
    x1 = max(0, x - size // 2)
    x2 = min(W, x1 + size)
    y1 = max(0, y - size // 2)
    y2 = min(H, y1 + size)
    img[:, y1:y2, x1:x2] = 0
    return img
```

- **Caution:** On small digits, large cutouts might erase the entire number. Start with size 3–5.

---

## 6. CutMix and RandAugment

**CutMix (vectorized + convolutional trainers):**

- Controlled by `--augment-cutmix-prob` (probability per sample) and `--augment-cutmix-alpha` (Beta distribution for mixing ratio).
- Two images share a rectangular patch and the labels are blended using the CutMix ratio.
- The trainers compute `λ * CE(y_a) + (1 - λ) * CE(y_b)` and combine gradients accordingly.
- Disabled automatically for the scalar trainer (labels stay single-class there).

**RandAugment:**

- `--augment-randaug-layers` picks how many random ops to stack per sample.
- `--augment-randaug-magnitude` acts as a 0–1 scaling factor for those ops.
- Available primitives: rotate, X/Y translations, flips, additive noise, cutout. Each layer randomly chooses one operation and scales its magnitude.
- Combine with the deterministic knobs (e.g., keep `--augment-rotate-deg` small so RandAugment explores meaningful ranges).

Example:

```python
# more aggressive digits run
python vectorized/main.py \
  --epochs 5 \
  --augment-randaug-layers 2 \
  --augment-randaug-magnitude 0.6 \
  --augment-cutout-prob 0.3 \
  --augment-cutmix-prob 0.2
```

Tune probabilities/thresholds per dataset and always visualize random batches to ensure semantics hold.

---

## Lab Challenge

1. Use `--no-augment` as a baseline, then enable rotations + flips on each trainer. Record accuracy changes in the [Experiment Log](experiment-log-template.md).
2. Enable CutMix on the vectorized or convolutional trainer (`--augment-cutmix-prob 0.4`). Plot 10 mixed samples and explain why the label mixing math is required.
3. Experiment with RandAugment (layers/magnitude sweep) and discuss whether it helps small digit datasets or if it simply injects too much randomness.

Remember: augmentations are only beneficial when they preserve label semantics. Evaluate carefully before enabling them in production runs.

**Navigation:**
[Back to Project Tour](../project-tour.md)

---

MIT License | [About](about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
