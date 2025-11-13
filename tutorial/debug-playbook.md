**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# Debug Playbook

**Breadcrumb:** [Home](README.md) / Debug Playbook


When experiments go sideways, start here. Each section lists common symptoms, likely causes, and quick fixes so you can get back to learning instead of fighting the environment.

---

## 1. Environment & Installation

| Symptom | Likely Cause | Quick Fix |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'cupy'` | CuPy not installed or wrong CUDA version | Install the wheel matching your CUDA driver (`pip install cupy-cuda12x`). On CPU-only machines, omit `--gpu`. |
| `ImportError: libcudart.so not found` | CUDA toolkit not in PATH/LD_LIBRARY_PATH | Install the CUDA runtime (matching the CuPy build) or add `/usr/local/cuda/lib64` to your `LD_LIBRARY_PATH`. |
| `matplotlib` backend errors on headless servers | Using default interactive backend without display | Set `matplotlib.use("Agg")` before importing pyplot or run the script with `--no-plot` / omit `--plot`. |

### Diagnostics

- `python scripts/test_system.py`: Basic sanity check of Python packages.
- `python scripts/test_cupy.py --stress-seconds 5 --stress-size 2048`: Verifies CuPy can allocate memory, run kernels, and synchronize.

---

## 2. GPU-Specific Issues

### A. GPU OOM (Out-Of-Memory)

**Symptoms**
- Training runs crash with `cupy.cuda.memory.OutOfMemoryError`.
- GPU utilization stays high after training (e.g., misclassification pass still computing).

**Fixes**
1. Reduce batch size (`--batch-size` flag) or switch to a smaller architecture.
2. Disable optional post-processing (e.g., run `convolutional/main.py` without `--plot`/`--show-misclassified` to skip large visualization buffers).
3. Clear lingering CuPy arrays by allowing scripts to exit, or manually call `backend.use_cpu()` when post-processing on CPU.
4. Use the quick-start scripts with smaller scenarios to reproduce the issue quickly.

### B. GPU Utilization Stays at 100% After Training

Likely cause: The script is still running a heavy pass (e.g., collecting misclassifications). Remove plotting flags or run with `--show-misclassified` only when needed.

### C. CuPy Fallback

If `--gpu` prints `[WARN] CuPy is not installed` and continues on CPU:
- Ensure `pip show cupy` returns a version matching your CUDA drivers.
- Test with `python scripts/test_cupy.py`; if it fails, reinstall CuPy or update drivers.

---

## 3. Data & Shape Mismatches

| Symptom | Likely Cause | Quick Fix |
| --- | --- | --- |
| `ValueError: Axis dimension mismatch` in fully connected layers | Flattened size doesn’t match Dense layer input (e.g., VGG16 pooling stride mismatch) | Recompute the flattened feature size; ensure the final Conv/Pool stages produce the expected spatial dimensions. The tutorial already patches VGG16 to track spatial size—check similar math if you customize architectures. |
| `RuntimeError: cannot reshape array of size ...` | Loading dataset with different image shapes (e.g., CIFAR-10) but still reshaping to `(N, 1, 28, 28)` | Update reshape logic to match the dataset: `(N, 3, 32, 32)` for RGB, `(N, height*width)` for flattened MLP inputs. |
| Labels look incorrect (e.g., floats instead of ints) | `.npy` files saved with wrong dtype | Ensure labels are `int64` before saving. `DataUtility` casts to `int64`, but double-check custom datasets. |

### Tips

- Print tensor shapes after each major step (especially before Dense layers).
- Use `scripts/quickstart_* --scenario dataset-swap` to validate new datasets end-to-end.

---

## 4. Training Instability

| Symptom | Likely Cause | Quick Fix |
| --- | --- | --- |
| Loss becomes `nan` or explodes | Learning rate too high, no gradient clipping | Lower `--lr`, enable `--grad-clip`, or switch to Adam for initial experiments. |
| Accuracy stuck at chance | Architecture too large, insufficient epochs, or data not normalized | Verify `common/data_utils` normalization ran (call `DataUtility.load_data`), start with simpler nets, and ensure labels are correct. |
| Training diverges when enabling GPU | Uninitialized CuPy memory or mismatched dtype | Confirm all inputs are `float32` (`DataUtility` already enforces this) and re-run the CuPy stress test. |

### Process Checklist

1. Run a **scalar** or **vectorized** quick-start scenario to ensure the math works.
2. Move to the **convolutional** script once the dataset + optimizer combination is stable.
3. Only then enable `--gpu`, `--lookahead`, or other advanced flags.

---

## 5. Checkpoint & Serialization Problems

| Symptom | Likely Cause | Quick Fix |
| --- | --- | --- |
| `KeyError` when loading a checkpoint | Architecture changed between save/load | Make sure you rebuild the exact same architecture (same `arch` flag, same hidden sizes) before calling `load_model`. |
| Checkpoint files huge or missing metadata | Saving every epoch or omitting metadata | Use `--save` only when needed and pass `metadata` (already done in main scripts). Clean unused checkpoints from `checkpoints/`. |

### Resume Workflow

- Save after an initial run (`--save checkpoints/demo.npz`).
- Later, resume with `--load checkpoints/demo.npz --skip-train` to evaluate or `--epochs 1` to continue training.

---

## 6. Quick Reference Commands

| Goal | Command |
| --- | --- |
| Validate CPU environment | `python scripts/test_system.py` |
| Stress-test GPU/CuPy | `python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096` |
| Baseline scalar run | `python scripts/quickstart_scalar.py --scenario basic --plot` |
| Optimizer comparison (vectorized) | `python scripts/quickstart_vectorized.py --scenario optimizer-compare --plot` |
| GPU-ready CNN demo | `python scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead --plot` |
| Fashion-MNIST swap smoke test | `python scripts/quickstart_scalar.py --scenario dataset-swap --plot` |

Keep this playbook handy whenever you run into trouble—most issues trace back to one of the scenarios above. If you discover a new pitfall, add it here (noting the symptoms, cause, and fix) to help the next learner.

---

MIT License | [About](about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
