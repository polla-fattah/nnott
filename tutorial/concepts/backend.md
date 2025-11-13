**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](../..) | [Architectures](../architecture-gallery.md)


# Backend & Device Utilities

**Breadcrumb:** [Home](../README.md) / [Core Concepts](../core-concepts.md) / Backend & Device Utilities


The sandbox abstracts CPU and GPU execution through a lightweight backend layer so the same code can run on either NumPy or CuPy. Understanding this layer helps you debug device issues and write backend-agnostic modules.

## XP Proxy

- **File:** `common/backend.py`
- **Class:** `_XPProxy`
- **Behavior:** Routes attribute access (`xp.exp`, `xp.zeros`) to the currently active array module (`numpy` by default, `cupy` when GPU is enabled).
- **Usage:** All modules import `xp = backend.xp` so they don’t need to care about the underlying device.

## Switching Devices

- `backend.use_gpu()` sets `xp = cupy` and marks `_using_gpu = True`.
- `backend.use_cpu()` reverts back to NumPy.
- `convolutional/main.py --gpu` calls `backend.use_gpu()` automatically, with a fallback if CuPy is unavailable.

## Helper Functions

- `to_device(array, dtype=None)`: Converts arrays to the active backend, optionally casting to `dtype`.
- `to_cpu(array)`: Ensures the output is a NumPy array (used for logging, plotting, checkpointing).
- `get_array_module(array)`: Returns `numpy` or `cupy` depending on the array type—useful when writing utilities that accept either.

## Memory Safety

- Optimizers and modules allocate state (e.g., zeros-like arrays) via `xp.zeros_like`, ensuring buffers live on the right device.
- Always avoid mixing NumPy and CuPy arrays in arithmetic; use `backend.to_device` and `backend.to_cpu` to convert explicitly.

## Diagnostics

- `scripts/test_cupy.py` exercises allocations, kernels, matmuls, and custom elementwise code. Run it whenever you change CUDA drivers or hardware.
- If you see `cupy.cuda.memory.OutOfMemoryError`, reduce batch sizes or enable batching in visualization functions (already done for misclassification plots).

## Best Practices

- Import `backend` once per module to avoid circular dependencies.
- Keep CPU-only utilities (like plotting) wrapped with `backend.to_cpu` so they don’t accidentally receive CuPy arrays.
- When writing new modules, test them with both `backend.use_cpu()` and `backend.use_gpu()` to ensure portability.

[Previous (Activation Functions)](activations.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Convolution Mechanics & Advanced Blocks)](convolution.md)

**Navigation:**
[Back to Core Concepts](../core-concepts.md)

---
Return to [Tutorial Hub](../README.md)
