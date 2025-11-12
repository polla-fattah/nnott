import numpy as _np

try:
    import cupy as _cp  # type: ignore
except Exception:  # CuPy might be unavailable
    _cp = None

_current_xp = _np
_using_gpu = False


class _XPProxy:
    """Proxy that forwards attribute access to the active array module."""

    def __getattr__(self, name):
        return getattr(_current_xp, name)


xp = _XPProxy()


def gpu_available():
    return _cp is not None


def is_gpu_enabled():
    return _using_gpu


def use_gpu():
    global _current_xp, _using_gpu
    if _cp is None:
        raise RuntimeError("CuPy is not installed; cannot enable GPU backend.")
    _current_xp = _cp
    _using_gpu = True


def use_cpu():
    global _current_xp, _using_gpu
    _current_xp = _np
    _using_gpu = False


def to_device(array, dtype=None):
    """Convert array-like to the active backend (GPU/CPU)."""
    if _using_gpu:
        if isinstance(array, _cp.ndarray):
            out = array
        else:
            out = _cp.asarray(array)
    else:
        if isinstance(array, _np.ndarray):
            out = array
        else:
            out = _np.asarray(array)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    return out


def to_cpu(array):
    """Ensure the returned array is a NumPy ndarray on host memory."""
    if _cp is not None and isinstance(array, _cp.ndarray):
        return _cp.asnumpy(array)
    return _np.asarray(array)


def current_module():
    return _current_xp


def get_array_module(array):
    if _cp is not None and isinstance(array, _cp.ndarray):
        return _cp
    return _np
