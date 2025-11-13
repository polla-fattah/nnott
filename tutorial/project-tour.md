# 01 · Project Tour

This project is designed as a teaching sandbox. Each directory demonstrates a different perspective on neural networks—from slow-but-transparent scalar math to production-style CNN stacks. Use this tour to map the theoretical lectures to the exact files you should inspect.

## Top-Level Layout

| Path | Why it matters |
| --- | --- |
| `scalar/` | Loop-based multilayer perceptron (MLP). Every neuron, gradient, and update is spelled out with Python `for` loops so you can follow the math line by line. |
| `vectorized/` | NumPy/CuPy version of the same MLP. Shows how replacing loops with matrix operations unlocks massive speed-ups while keeping the math identical. |
| `convolutional/` | CNN modules (`convolutional/modules.py`), a unified trainer, and the architecture definitions living under `convolutional/architectures/`. |
| `common/` | Shared backend selector, losses, softmax, model I/O, and data utilities. Every other package imports from here to stay consistent. |
| `data/` | Stores the MNIST `.npy` files (`train_images.npy`, `train_labels.npy`, etc.) plus `data/readme.md` describing the dataset. |
| `scripts/` | One-off tools such as GPU diagnostics (`test_cupy.py`), experiment helpers, and sanity checks. |
| `checkpoints/` | Optional folder (create it as needed) where training runs can save `.npz` weight files. |

### Shared Utilities (`common/`)

- **Backend selector** (`common/backend.py`): provides the `xp` alias that points to either NumPy (CPU) or CuPy (GPU) plus helper functions (`to_device`, `to_cpu`, `gpu_available`, etc.).
- **Data loader** (`common/data_utils.py`): loads the MNIST `.npy` arrays, normalizes them (mean/std), reshapes images, and exposes quick sample-plotting helpers.
- **Loss & softmax** (`common/cross_entropy.py`, `common/softmax.py`): numerically stable log-sum-exp implementation of cross-entropy and the associated gradients.
- **Model I/O** (`common/model_io.py`): serialize weights/metadata to `.npz` files and reload them later—used by all trainers.

### Dataset Storage (`data/`)

MNIST is packaged as NumPy arrays to keep the focus on neural-network mechanics. Files include:

- `train_images.npy` / `train_labels.npy`: 60k training samples (28×28 grayscale digits).
- `test_images.npy` / `test_labels.npy`: 10k evaluation samples.
- `readme.md`: explains provenance, shapes, and how to create validation splits if desired.

### Helper Scripts (`scripts/`)

- `test_cupy.py`: stress-tests your CUDA+CuPy setup with basic ops, matmuls, kernel launches, and configurable GEMM stress loops (`--stress-seconds`, `--stress-size`).
- `test_system.py`, `sanity_linear_toy.py`, `experiments.py`: minimal scripts for system diagnostics or quick experiments.

## Required Packages

The sandbox intentionally keeps dependencies light. Install the following inside your Conda/virtualenv:

- [Python ≥ 3.10](https://docs.python.org/3/tutorial/)
- [NumPy](tools/numpy.md)
- [CuPy](tools/cupy.md) (optional, for GPU acceleration)
- [Matplotlib](tools/matplotlib.md) (for training curves + misclassification grids)
- [tqdm](tools/tqdm.md) (progress bars)

Standard-library modules (`argparse`, `pathlib`, `math`, `time`, etc.) round out the tooling.

## Why This Structure Matters

The directory split mirrors the learning journey:

1. **Understand the math** in `scalar/`.
2. **Learn vectorization** in `vectorized/`.
3. **Scale up to CNNs** in `convolutional/`.
4. **Rely on shared utilities** in `common/` so all implementations behave the same.

Refer back to this map whenever you encounter a concept in the tutorial—the table above tells you which file to inspect next.
