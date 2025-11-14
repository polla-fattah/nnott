[MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# 02 · Implementations & Hardware

**Breadcrumb:** [Home](README.md) / 02 · Implementations & Hardware


One of the sandbox’s teaching pillars is showing how the same neural network evolves from naïve loops to GPU-accelerated kernels. This module explains each stage and how to toggle between CPU and GPU execution.

## Scalar (“Loop-Based”) Implementation

- **Location:** `scalar/`
- **What it teaches:** Each neuron, layer, and gradient update is implemented with explicit Python loops. Files like `scalar/neuron.py`, `scalar/layer.py`, and `scalar/trainer.py` print intermediate tensors so you can trace forward/backward propagation exactly as described in lecture.
- **Takeaway:** Perfect for debugging your intuition. You can literally follow how the derivative of the loss with respect to a weight is computed.
- **Trade-off:** Extremely slow—only intended for tiny batches, but that is the point.

## Vectorized Implementation

- **Location:** `vectorized/`
- **What it teaches:** Rewrites the scalar math using NumPy arrays (and optionally CuPy). Files such as `vectorized/modules.py` and `vectorized/optim.py` replace loops with matrix multiplications and broadcasted operations.
- **Benefits:** Orders-of-magnitude faster while remaining mathematically identical to the scalar version. Demonstrates why linear algebra primitives are the workhorse of deep learning.
- **Tip:** Run `python vectorized/main.py --epochs 5 --batch-size 128` and compare runtime against the scalar script.

## Convolutional Stack

- **Location:** `convolutional/`
- **What it teaches:** Real-world CNNs built from custom Conv2D, pooling, normalization, activation, and dropout layers (`convolutional/modules.py`). Complex architectures such as LeNet, AlexNet, VGG16, ResNet18, EfficientNet-Lite0, and ConvNeXt-Tiny live in `convolutional/architectures/`.
- **Trainer:** `convolutional/trainer.py` unifies data shuffling, augmentation, LR schedules, backprop, and evaluation for every architecture. It respects the active backend (CPU vs GPU) automatically.

## Backend Switching (CPU ↔ GPU)

### The `xp` Abstraction

- `common/backend.py` exposes an `xp` proxy that points to NumPy by default.
- Calling `backend.use_gpu()` swaps `xp` to CuPy (if available) and flips helper functions (`to_device`, `to_cpu`) accordingly.
- All downstream layers/optimizers import `xp`, so the implementation code does not need to care whether it’s running on CPU or GPU.

### When to Use `--gpu`

- Any `convolutional/main.py` run accepts `--gpu`. If CuPy is installed and a CUDA device is visible, training happens on the GPU; otherwise the script gracefully falls back to CPU.
- Vectorized scripts can also benefit by calling `backend.use_gpu()` before constructing modules (or by setting an environment variable if you prefer).

### Verifying Your GPU Setup

Run the provided stress test to sanity-check drivers, CUDA, and CuPy:

```bash
python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096
```

You can dial up `--stress-seconds` or `--stress-size` to keep the GPU busy longer. The script reports device info, runs elementwise ops, matmuls, custom kernels, and a configurable GEMM loop, exiting non-zero if anything fails.

## Summary Table

| Implementation | Location | Highlights |
| --- | --- | --- |
| Scalar MLP | `scalar/` | Pedagogical loops, verbose printing, easiest to debug. |
| Vectorized MLP | `vectorized/` | NumPy/CuPy arrays, fast batch math, same equations as scalar. |
| CNN Suite | `convolutional/` | Custom layers, modern architectures, unified trainer. |
| Backend Layer | `common/backend.py` | `xp` proxy, `use_gpu/use_cpu`, device transfers. |
| GPU Health Check | `scripts/test_cupy.py` | Confirms CuPy can allocate, compute, and stress your GPU. |

Understanding these tiers equips you to reason about both the *what* (network math) and the *how* (performance engineering).

### Lab Challenges

1. **Benchmark the tiers:** Time one epoch of the scalar, vectorized, and convolutional implementations (use `scripts/quickstart_*` if you prefer). Record the wall-clock duration and explain the speed differences in your own words.
2. **Backend toggle drill:** Run `python convolutional/main.py resnet18 --epochs 1 --batch-size 64` twice—once with `--gpu`, once without. Note the runtime, GPU utilization (via `nvidia-smi`), and any startup warnings. Summarize the steps the backend layer took to select the device.

---

MIT License | [About](about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
