import numpy as np
# moved to common package


def softmax(logits, axis=-1):
    """Numerically stable softmax.

    Accepts 1D or ND arrays and computes along `axis`.
    Returns probabilities that sum to 1 along `axis`.
    """
    logits = np.asarray(logits, dtype=np.float64)
    # subtract max for numerical stability
    max_logits = np.max(logits, axis=axis, keepdims=True)
    shifted = logits - max_logits
    exp = np.exp(shifted)
    sum_exp = np.sum(exp, axis=axis, keepdims=True)
    probs = exp / sum_exp
    return probs.astype(np.float64)


def _run_checks():
    # 1D basic properties
    z = np.array([1.0, 2.0, 3.0])
    p = softmax(z)
    assert p.shape == z.shape
    assert np.all(p >= 0) and np.allclose(p.sum(), 1.0, atol=1e-12)

    # shift invariance
    c = 123.456
    p_shift = softmax(z + c)
    assert np.allclose(p, p_shift, atol=1e-12)

    # zeros -> uniform
    z0 = np.zeros(5)
    p0 = softmax(z0)
    assert np.allclose(p0, np.ones(5) / 5.0, atol=1e-12)

    # large numbers should not overflow
    z_big = np.array([1000.0, 1001.0, 1002.0])
    p_big = softmax(z_big)
    assert np.isfinite(p_big).all() and np.allclose(p_big.sum(), 1.0)

    # 2D along last axis
    Z = np.array([[0.0, 0.0], [1.0, 2.0]])
    P = softmax(Z, axis=1)
    assert np.allclose(P.sum(axis=1), np.ones(2))

    print("softmax checks passed ✔")


if __name__ == "__main__":
    _run_checks()
