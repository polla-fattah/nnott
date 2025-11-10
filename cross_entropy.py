import numpy as np
from softmax import softmax


def logsumexp(x, axis=-1, keepdims=False):
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return y if keepdims else np.squeeze(y, axis=axis)


def cross_entropy_from_logits(logits, targets, reduction="mean"):
    """Cross-entropy for multi-class from logits (stable).

    targets can be class indices (int array) or one-hot vectors.
    logits: shape (N, C) or (C,)
    returns scalar if reduction is 'mean' or 'sum'; per-sample otherwise.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # Ensure batch dimension
    if logits.ndim == 1:
        logits = logits[None, :]

    N, C = logits.shape

    # Convert targets to indices
    t = np.asarray(targets)
    if t.ndim == 0:
        t = np.array([int(t)])
    if t.ndim == 1 and t.size == N and np.issubdtype(t.dtype, np.integer):
        target_idx = t.astype(int)
    else:
        # assume one-hot-like (N, C)
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 1:
            t = t[None, :]
        target_idx = np.argmax(t, axis=1).astype(int)

    # Stable CE: -z_y + logsumexp(z)
    lse = logsumexp(logits, axis=1)
    z_y = logits[np.arange(N), target_idx]
    losses = -z_y + lse

    if reduction == "mean":
        return float(np.mean(losses))
    if reduction == "sum":
        return float(np.sum(losses))
    return losses


def cross_entropy_grad_logits(logits, targets):
    """Gradient dL/dlogits for CE from logits: softmax(logits) - one_hot(target)."""
    logits = np.asarray(logits, dtype=np.float64)
    probs = softmax(logits, axis=-1)

    # shape handling
    if probs.ndim == 1:
        probs = probs[None, :]

    N, C = probs.shape

    t = np.asarray(targets)
    if t.ndim == 0:
        t = np.array([int(t)])
    if t.ndim == 1 and t.size == N and np.issubdtype(t.dtype, np.integer):
        target_idx = t.astype(int)
    else:
        t = np.asarray(t, dtype=np.float64)
        if t.ndim == 1:
            t = t[None, :]
        target_idx = np.argmax(t, axis=1).astype(int)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), target_idx] = 1.0

    grad = probs - one_hot
    return grad if logits.ndim > 1 else grad[0]


def _run_checks():
    # Value checks
    logits = np.array([0.0, 0.0, 0.0])
    # uniform probs => CE = -log(1/3)
    ce = cross_entropy_from_logits(logits, 1)
    assert np.allclose(ce, -np.log(1/3), atol=1e-12)

    # Perfect prediction should be small
    logits = np.array([0.0, 10.0, 0.0])
    ce = cross_entropy_from_logits(logits, 1)
    assert ce < 1e-4

    # Gradient check (finite difference)
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 3)) * 0.5
    targets = np.array([0, 1, 2, 1])
    eps = 1e-5
    analytic = cross_entropy_grad_logits(logits, targets)
    numeric = np.zeros_like(logits)
    base = cross_entropy_from_logits(logits, targets, reduction="mean")
    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            Zp = logits.copy(); Zp[i, j] += eps
            Zm = logits.copy(); Zm[i, j] -= eps
            fp = cross_entropy_from_logits(Zp, targets, reduction="mean")
            fm = cross_entropy_from_logits(Zm, targets, reduction="mean")
            numeric[i, j] = (fp - fm) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-4)

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
