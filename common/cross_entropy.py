import numpy as _np
import common.backend as backend
from common.softmax import softmax

xp = backend.xp


def logsumexp(x, axis=-1, keepdims=False):
    arr = xp.asarray(x, dtype=xp.float64)
    m = xp.max(arr, axis=axis, keepdims=True)
    y = xp.log(xp.sum(xp.exp(arr - m), axis=axis, keepdims=True)) + m
    return y if keepdims else xp.squeeze(y, axis=axis)


def cross_entropy_from_logits(logits, targets, reduction="mean"):
    """Cross-entropy for multi-class from logits (stable).

    targets can be class indices (int array) or one-hot vectors.
    logits: shape (N, C) or (C,)
    returns scalar if reduction is 'mean' or 'sum'; per-sample otherwise.
    """
    logits = xp.asarray(logits, dtype=xp.float64)
    # Ensure batch dimension
    if logits.ndim == 1:
        logits = logits[None, :]

    N, C = logits.shape

    # Convert targets to indices
    t = _np.asarray(targets)
    if t.ndim == 0:
        t = _np.array([int(t)])
    if t.ndim == 1 and t.size == N and _np.issubdtype(t.dtype, _np.integer):
        target_idx = t.astype(_np.int64, copy=False)
    else:
        t = _np.asarray(t, dtype=_np.float64)
        if t.ndim == 1:
            t = t[None, :]
        target_idx = t.argmax(axis=1).astype(_np.int64, copy=False)
    target_idx_xp = backend.to_device(target_idx, dtype=xp.int64)

    # Stable CE: -z_y + logsumexp(z)
    lse = logsumexp(logits, axis=1)
    z_y = logits[xp.arange(N), target_idx_xp]
    losses = -z_y + lse

    if reduction == "mean":
        return float(backend.to_cpu(losses.mean()))
    if reduction == "sum":
        return float(backend.to_cpu(losses.sum()))
    return losses


def cross_entropy_grad_logits(logits, targets):
    """Gradient dL/dlogits for CE from logits: softmax(logits) - one_hot(target)."""
    logits = xp.asarray(logits, dtype=xp.float64)
    probs = softmax(logits, axis=-1)

    # shape handling
    if probs.ndim == 1:
        probs = probs[None, :]

    N, C = probs.shape

    t = _np.asarray(targets)
    if t.ndim == 0:
        t = _np.array([int(t)])
    if t.ndim == 1 and t.size == N and _np.issubdtype(t.dtype, _np.integer):
        target_idx = t.astype(_np.int64, copy=False)
    else:
        t = _np.asarray(t, dtype=_np.float64)
        if t.ndim == 1:
            t = t[None, :]
        target_idx = t.argmax(axis=1).astype(_np.int64, copy=False)
    target_idx_xp = backend.to_device(target_idx, dtype=xp.int64)

    one_hot = xp.zeros_like(probs)
    one_hot[xp.arange(N), target_idx_xp] = 1.0

    grad = probs - one_hot
    return grad if logits.ndim > 1 else grad[0]


def _run_checks():
    # Value checks
    logits = _np.array([0.0, 0.0, 0.0])
    # uniform probs => CE = -log(1/3)
    ce = cross_entropy_from_logits(logits, 1)
    assert _np.allclose(ce, -_np.log(1 / 3), atol=1e-12)

    # Perfect prediction should be small
    logits = _np.array([0.0, 10.0, 0.0])
    ce = cross_entropy_from_logits(logits, 1)
    assert ce < 1e-4

    # Gradient check (finite difference)
    rng = _np.random.default_rng(0)
    logits = rng.normal(size=(4, 3)) * 0.5
    targets = _np.array([0, 1, 2, 1])
    eps = 1e-5
    analytic = cross_entropy_grad_logits(logits, targets)
    numeric = _np.zeros_like(logits)
    base = cross_entropy_from_logits(logits, targets, reduction="mean")
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            Zp = logits.copy(); Zp[i, j] += eps
            Zm = logits.copy(); Zm[i, j] -= eps
            fp = cross_entropy_from_logits(Zp, targets, reduction="mean")
            fm = cross_entropy_from_logits(Zm, targets, reduction="mean")
            numeric[i, j] = (fp - fm) / (2 * eps)
    assert _np.allclose(analytic, numeric, atol=1e-6, rtol=1e-4)

    print("cross-entropy checks passed ✔")


if __name__ == "__main__":
    _run_checks()


# OOP wrapper for integration with Trainer
class CrossEntropyLoss:
    """Cross-entropy loss operating on logits (stable).

    Usage:
        ce = CrossEntropyLoss(reduction="mean")
        loss = ce.forward(logits, targets)
        grad = ce.backward(logits, targets)
    """

    def __init__(self, reduction: str = "mean"):
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction

    def forward(self, logits, targets):
        red = self.reduction if self.reduction != "none" else None
        if red is None:
            return cross_entropy_from_logits(logits, targets, reduction="none")
        return cross_entropy_from_logits(logits, targets, reduction=red)

    def backward(self, logits, targets):
        return cross_entropy_grad_logits(logits, targets)
