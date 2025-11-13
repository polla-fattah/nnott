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

#### Swapping in Other Datasets

You can drop additional datasets into the same folder as long as you convert them to `.npy` files. Two quick options:

- **Fashion-MNIST (drop-in replacement)**  
  1. Download via `torchvision.datasets.FashionMNIST` or the official website.  
  2. Save the tensors to `.npy`:
     ```python
     np.save("data/fashion_train_images.npy", train_images.numpy())
     np.save("data/fashion_train_labels.npy", train_labels.numpy())
     ```  
  3. Run `DataUtility("data").load_data(train_images_file="fashion_train_images.npy", ...)`.
  The scalar/vectorized/convolutional scripts will work without further changes because Fashion-MNIST shares MNIST’s 28×28×1 shape.

- **CIFAR-10 (RGB, 32×32)**  
  1. Convert the training/test splits to `.npy` (e.g., via `torchvision.datasets.CIFAR10`).  
  2. Update the reshape lines in `vectorized/main.py` and `convolutional/main.py` to use `(len(X), 3, 32, 32)` for CNNs or `len(X), -1` for MLPs.  
  3. Adjust the first convolutional layer to accept 3 input channels (e.g., modify the architecture builder or create a new entry in `ARCH_REGISTRY`).  
  4. Normalize using dataset-specific mean/std to keep training stable.

For any other dataset, follow the same pattern: convert to `.npy`, point `DataUtility` at the new filenames, and ensure model input shapes/first-layer channels match the data.

##### Fashion-MNIST Walk-Through

Convert and run:

```python
# convert_fashion_mnist.py
import numpy as np
from torchvision.datasets import FashionMNIST

train = FashionMNIST(root="data/raw", train=True, download=True)
test = FashionMNIST(root="data/raw", train=False, download=True)

np.save("data/fashion_train_images.npy", train.data.numpy())
np.save("data/fashion_train_labels.npy", train.targets.numpy())
np.save("data/fashion_test_images.npy", test.data.numpy())
np.save("data/fashion_test_labels.npy", test.targets.numpy())
```

```bash
python scripts/quickstart_scalar.py --scenario dataset-swap --plot \
    --alt-train-images fashion_train_images.npy \
    --alt-train-labels fashion_train_labels.npy \
    --alt-test-images fashion_test_images.npy \
    --alt-test-labels fashion_test_labels.npy
```

No additional changes needed because the images are still 28×28×1.

##### CIFAR-10 Walk-Through

Convert RGB 32×32 data and reshape for CNNs:

```python
# convert_cifar10.py
import numpy as np
from torchvision.datasets import CIFAR10

train = CIFAR10(root="data/raw", train=True, download=True)
test = CIFAR10(root="data/raw", train=False, download=True)

np.save("data/cifar10_train_images.npy", train.data.transpose(0, 3, 1, 2))
np.save("data/cifar10_train_labels.npy", np.array(train.targets))
np.save("data/cifar10_test_images.npy", test.data.transpose(0, 3, 1, 2))
np.save("data/cifar10_test_labels.npy", np.array(test.targets))
```

```bash
python scripts/quickstart_convolutional.py --scenario dataset-swap \
    --arch resnet18 --epochs 1 --batch-size 64 --plot \
    --alt-train-images cifar10_train_images.npy \
    --alt-train-labels cifar10_train_labels.npy \
    --alt-test-images cifar10_test_images.npy \
    --alt-test-labels cifar10_test_labels.npy \
    --image-shape 3,32,32
```

ResNet18 already handles 3-channel inputs when the data reshape matches `(3,32,32)`; confirm your custom architectures do the same.

### Helper Scripts (`scripts/`)

- `test_cupy.py`: stress-tests your CUDA+CuPy setup with basic ops, matmuls, kernel launches, and configurable GEMM stress loops (`--stress-seconds`, `--stress-size`).
- `test_system.py`, `sanity_linear_toy.py`, `experiments.py`: minimal scripts for system diagnostics or quick experiments.

### Lab Challenges

1. **Map the repo:** Create a diagram (digital or hand-drawn) showing each top-level folder, two representative files inside it, and one sentence about what they do. Share it with a peer or TA to verify you captured every component.
2. **Dataset swap dry run:** Convert Fashion-MNIST to `.npy`, place the files in `data/`, and run `scripts/quickstart_scalar.py --scenario dataset-swap --plot`. Record the resulting accuracy and any code/config changes you had to make.

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
