# Augmentation Playground

**Breadcrumb:** [Home](README.md) / Augmentation Playground


The convolutional trainer currently applies random pixel shifts (±2) to each batch via `ConvTrainer._augment_batch`. This page shows how to extend that mechanism with additional transforms—rotations, flips, noise, cutout—while emphasizing when such augmentations make sense (label semantics matter!).

## 1. Understanding the Existing Augmentor

```python
def _augment_batch(self, xb, max_shift=2):
    if xb.ndim != 4:
        return xb
    shifted = xp.empty_like(xb)
    for i in range(len(xb)):
        dx = int(xp.random.randint(-max_shift, max_shift + 1))
        dy = int(xp.random.randint(-max_shift, max_shift + 1))
        shifted[i] = xp.roll(xp.roll(xb[i], dy, axis=1), dx, axis=2)
    return shifted
```

- Inputs are `(N, C, H, W)` arrays; the code uses `xp` to support NumPy or CuPy.
- Augmentation is applied *per batch* after shuffling but before the forward pass.

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

## 6. Putting It Together

Combine the above snippets:

```python
def _augment_batch(self, xb, max_shift=2, max_rotate_deg=10, noise_std=0.05, cutout_size=5):
    if xb.ndim != 4:
        return xb
    out = xp.empty_like(xb)
    for i in range(len(xb)):
        img = xb[i]
        # shifts
        dx = int(xp.random.randint(-max_shift, max_shift + 1))
        dy = int(xp.random.randint(-max_shift, max_shift + 1))
        img = xp.roll(xp.roll(img, dy, axis=1), dx, axis=2)
        # rotation
        if max_rotate_deg > 0 and xp.random.rand() < 0.5:
            angle = float(xp.random.uniform(-max_rotate_deg, max_rotate_deg))
            img = rotate_nearest(img, math.radians(angle))
        # noise
        if noise_std > 0:
            img = img + noise_std * xp.random.randn(*img.shape).astype(img.dtype)
        # optional cutout
        if cutout_size > 0 and xp.random.rand() < 0.3:
            img = apply_cutout(img, cutout_size)
        out[i] = img
    return xp.clip(out, -3, 3)
```

Tune probabilities/thresholds per dataset.

---

## Lab Challenge

1. Clone `ConvTrainer._augment_batch` into a branch, add one new transform (rotation, noise, flip, or cutout), and train BaselineCNN for 2 epochs. Does it help or hurt accuracy?
2. Visualize 10 augmented samples using `plot_image_grid` to ensure labels still make sense. Document your findings in the [Experiment Log](experiment-log-template.md).
3. For CIFAR-10, try combining horizontal flips and color jitter (via additive noise). Note how training time or stability changes.

Remember: augmentations are only beneficial when they preserve label semantics. Evaluate carefully before enabling them in production runs.

**Navigation:**
[Back to Project Tour](../project-tour.md)
