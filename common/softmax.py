import numpy as _np
import common.backend as backend

xp = backend.xp


def softmax(logits, axis=-1):
    """Numerically stable softmax along a given axis (supports GPU backend)."""
    arr = xp.asarray(logits, dtype=xp.float64)
    max_logits = xp.max(arr, axis=axis, keepdims=True)
    shifted = arr - max_logits
    exp = xp.exp(shifted)
    sum_exp = xp.sum(exp, axis=axis, keepdims=True)
    probs = exp / sum_exp
    return probs


def _run_checks():
    # 1D basic properties
    z = _np.array([1.0, 2.0, 3.0])
    p = backend.to_cpu(softmax(z))
    assert p.shape == z.shape
    assert _np.all(p >= 0) and _np.allclose(p.sum(), 1.0, atol=1e-12)

    # shift invariance
    c = 123.456
    p_shift = softmax(z + c)
    assert _np.allclose(p, p_shift, atol=1e-12)

    # zeros -> uniform
    z0 = _np.zeros(5)
    p0 = softmax(z0)
    assert _np.allclose(p0, _np.ones(5) / 5.0, atol=1e-12)

    # large numbers should not overflow
    z_big = _np.array([1000.0, 1001.0, 1002.0])
    p_big = softmax(z_big)
    assert _np.isfinite(p_big).all() and _np.allclose(p_big.sum(), 1.0)

    # 2D along last axis
    Z = _np.array([[0.0, 0.0], [1.0, 2.0]])
    P = backend.to_cpu(softmax(Z, axis=1))
    assert _np.allclose(P.sum(axis=1), _np.ones(2))

    print("softmax checks passed ✔")


if __name__ == "__main__":
    _run_checks()
