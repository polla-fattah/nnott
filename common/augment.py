import math
from typing import Dict, Optional, Tuple

import numpy as np


DEFAULT_CONFIG = {
    "max_shift": 2,
    "rotate_deg": 10.0,
    "rotate_prob": 0.5,
    "hflip_prob": 0.5,
    "vflip_prob": 0.0,
    "noise_std": 0.02,
    "noise_prob": 0.3,
    "noise_clip": 3.0,
    "cutout_prob": 0.0,
    "cutout_size": 4,
    "cutmix_prob": 0.0,
    "cutmix_alpha": 1.0,
    "randaugment_layers": 0,
    "randaugment_magnitude": 0.0,
}


def build_augment_config(overrides: Optional[Dict[str, float]] = None, **kwargs) -> Dict[str, float]:
    """Return a sanitized augmentation config shared across trainers."""
    cfg = dict(DEFAULT_CONFIG)
    source = {}
    if overrides:
        source.update(overrides)
    if kwargs:
        source.update(kwargs)

    for key, value in source.items():
        if key not in cfg or value is None:
            continue
        cfg[key] = value

    def clamp_prob(val):
        return float(max(0.0, min(1.0, float(val))))

    cfg["max_shift"] = max(0, int(cfg["max_shift"]))
    cfg["rotate_deg"] = max(0.0, float(cfg["rotate_deg"]))
    cfg["rotate_prob"] = clamp_prob(cfg["rotate_prob"])
    cfg["hflip_prob"] = clamp_prob(cfg["hflip_prob"])
    cfg["vflip_prob"] = clamp_prob(cfg["vflip_prob"])
    cfg["noise_std"] = max(0.0, float(cfg["noise_std"]))
    cfg["noise_prob"] = clamp_prob(cfg["noise_prob"])
    cfg["noise_clip"] = max(0.0, float(cfg["noise_clip"]))
    cfg["cutout_prob"] = clamp_prob(cfg["cutout_prob"])
    cfg["cutout_size"] = max(1, int(cfg["cutout_size"])) if cfg["cutout_prob"] > 0 else int(cfg["cutout_size"])
    cfg["cutmix_prob"] = clamp_prob(cfg["cutmix_prob"])
    cfg["cutmix_alpha"] = max(1e-6, float(cfg["cutmix_alpha"]))
    cfg["randaugment_layers"] = max(0, int(cfg["randaugment_layers"]))
    cfg["randaugment_magnitude"] = clamp_prob(cfg["randaugment_magnitude"])
    return cfg


def augment_image_batch(
    batch,
    cfg: Dict[str, float],
    xp_module=None,
    labels=None,
    allow_label_mix: bool = True,
):
    """Apply channel-first augmentations to a batch shaped (B,C,H,W) or (B,H,W)."""
    xp = xp_module or np
    if batch.ndim not in (3, 4):
        return batch, None
    squeeze_channel = False
    if batch.ndim == 3:
        batch = batch[:, None, :, :]
        squeeze_channel = True
    B, C, H, W = batch.shape
    rng = xp.random
    out = xp.empty_like(batch)
    for i in range(B):
        img = batch[i]
        img = _apply_basic_ops(img, cfg, xp, rng)
        img = _apply_randaugment(img, cfg, xp, rng)
        out[i] = img

    mix_meta = None
    if allow_label_mix:
        out, mix_meta = _apply_cutmix(out, labels, cfg, xp, rng)

    if squeeze_channel:
        out = out[:, 0]
    return out, mix_meta


def augment_flat_batch(
    flat_batch,
    cfg: Dict[str, float],
    xp_module=None,
    labels=None,
    image_side: Optional[int] = None,
    channels: int = 1,
    allow_label_mix: bool = True,
):
    """Augment flattened images (B, D) by reshaping to square grids."""
    xp = xp_module or np
    if flat_batch.ndim != 2:
        return flat_batch, None

    B, D = flat_batch.shape
    side = image_side or int(round(math.sqrt(D / max(1, channels))))
    if side * side * channels != D:
        return flat_batch, None

    reshaped = flat_batch.reshape(B, channels, side, side)
    aug, mix = augment_image_batch(reshaped, cfg, xp_module=xp, labels=labels, allow_label_mix=allow_label_mix)
    return aug.reshape(B, D), mix


def _scalar(value):
    """Convert NumPy/CuPy scalar arrays to native Python scalars."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _apply_basic_ops(img, cfg, xp, rng):
    if cfg["max_shift"] > 0:
        dx = int(_scalar(rng.randint(-cfg["max_shift"], cfg["max_shift"] + 1)))
        dy = int(_scalar(rng.randint(-cfg["max_shift"], cfg["max_shift"] + 1)))
        img = xp.roll(xp.roll(img, dy, axis=-2), dx, axis=-1)
    if cfg["rotate_deg"] > 0 and float(_scalar(rng.rand())) < cfg["rotate_prob"]:
        angle = float(_scalar(rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"])))
        img = _rotate_image(img, math.radians(angle), xp)
    if cfg["hflip_prob"] > 0 and float(_scalar(rng.rand())) < cfg["hflip_prob"]:
        img = img[..., :, ::-1]
    if cfg["vflip_prob"] > 0 and float(_scalar(rng.rand())) < cfg["vflip_prob"]:
        img = img[..., ::-1, :]
    if cfg["noise_std"] > 0 and float(_scalar(rng.rand())) < cfg["noise_prob"]:
        noise = rng.normal(0.0, cfg["noise_std"], size=img.shape).astype(img.dtype, copy=False)
        img = img + noise
    if cfg["cutout_prob"] > 0 and float(_scalar(rng.rand())) < cfg["cutout_prob"]:
        img = _apply_cutout(img, cfg["cutout_size"], xp, rng)
    if cfg["noise_clip"] > 0:
        img = xp.clip(img, -cfg["noise_clip"], cfg["noise_clip"])
    return img


def _apply_cutout(img, size, xp, rng):
    c, h, w = img.shape
    half = max(1, size // 2)
    cx = int(_scalar(rng.randint(0, w)))
    cy = int(_scalar(rng.randint(0, h)))
    x1 = max(0, cx - half)
    x2 = min(w, cx + half)
    y1 = max(0, cy - half)
    y2 = min(h, cy + half)
    img[:, y1:y2, x1:x2] = 0
    return img


def _apply_randaugment(img, cfg, xp, rng):
    layers = cfg["randaugment_layers"]
    if layers <= 0:
        return img
    magnitude = cfg["randaugment_magnitude"]
    if magnitude <= 0:
        return img

    ops = ("rotate", "shift_x", "shift_y", "flip_lr", "flip_ud", "noise", "cutout")
    for _ in range(layers):
        op = ops[int(_scalar(rng.randint(0, len(ops))))]
        img = _randaugment_op(img, op, magnitude, cfg, xp, rng)
    return img


def _randaugment_op(img, op, magnitude, cfg, xp, rng):
    if op == "rotate":
        max_deg = max(cfg["rotate_deg"], 1.0)
        angle = (magnitude * max_deg) * (1 if float(_scalar(rng.rand())) < 0.5 else -1)
        return _rotate_image(img, math.radians(angle), xp)
    if op == "shift_x":
        max_shift = max(cfg["max_shift"], 1)
        dx = int(round(magnitude * max_shift))
        dx = dx if dx != 0 else 1
        return xp.roll(img, dx, axis=-1)
    if op == "shift_y":
        max_shift = max(cfg["max_shift"], 1)
        dy = int(round(magnitude * max_shift))
        dy = dy if dy != 0 else 1
        return xp.roll(img, dy, axis=-2)
    if op == "flip_lr":
        if float(_scalar(rng.rand())) < magnitude:
            return img[..., :, ::-1]
        return img
    if op == "flip_ud":
        if float(_scalar(rng.rand())) < magnitude:
            return img[..., ::-1, :]
        return img
    if op == "noise":
        std = magnitude * max(cfg["noise_std"], 1e-4)
        noise = rng.normal(0.0, std, size=img.shape).astype(img.dtype, copy=False)
        out = img + noise
        if cfg["noise_clip"] > 0:
            out = xp.clip(out, -cfg["noise_clip"], cfg["noise_clip"])
        return out
    if op == "cutout" and cfg["cutout_size"] > 0:
        size = max(1, int(round(magnitude * cfg["cutout_size"])))
        return _apply_cutout(img, size, xp, rng)
    return img


def _rotate_image(img, angle_rad, xp):
    squeeze = False
    if img.ndim == 2:
        img = img[None, :, :]
        squeeze = True
    c, h, w = img.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    yy, xx = xp.meshgrid(
        xp.arange(h, dtype=xp.float32),
        xp.arange(w, dtype=xp.float32),
        indexing="ij",
    )
    x0 = xx - cx
    y0 = yy - cy
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    xr = cos_a * x0 + sin_a * y0 + cx
    yr = -sin_a * x0 + cos_a * y0 + cy
    xi = xp.clip(xp.rint(xr), 0, w - 1).astype(xp.int32)
    yi = xp.clip(xp.rint(yr), 0, h - 1).astype(xp.int32)
    rotated = xp.empty_like(img)
    for ch in range(c):
        rotated[ch] = img[ch, yi, xi]
    return rotated[0] if squeeze else rotated


def _apply_cutmix(batch, labels, cfg, xp, rng):
    prob = cfg["cutmix_prob"]
    if prob <= 0 or labels is None or len(batch) < 2:
        return batch, None
    B, _, H, W = batch.shape
    idx_perm = xp.random.permutation(B)
    lam = xp.ones(B, dtype=xp.float32)
    targets_b = labels.copy()
    applied = False
    for i in range(B):
        if float(_scalar(rng.rand())) >= prob:
            continue
        j = int(idx_perm[i])
        lam_sample = float(_scalar(rng.beta(cfg["cutmix_alpha"], cfg["cutmix_alpha"])))
        x1, y1, x2, y2 = _rand_bbox(W, H, lam_sample, xp, rng)
        batch[i, :, y1:y2, x1:x2] = batch[j, :, y1:y2, x1:x2]
        area = (x2 - x1) * (y2 - y1)
        lam[i] = 1.0 - area / float(H * W)
        targets_b[i] = labels[j]
        applied = True
    if not applied:
        return batch, None
    return batch, {
        "targets_a": labels,
        "targets_b": targets_b,
        "lam": lam,
    }


def _rand_bbox(width, height, lam_sample, xp, rng) -> Tuple[int, int, int, int]:
    cut_ratio = math.sqrt(1.0 - lam_sample)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = int(_scalar(rng.randint(0, width)))
    cy = int(_scalar(rng.randint(0, height)))
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(width, x1 + cut_w)
    y2 = min(height, y1 + cut_h)
    return x1, y1, x2, y2
