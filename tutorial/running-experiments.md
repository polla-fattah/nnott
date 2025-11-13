# 03 · Running Experiments

This guide walks you through training the provided models, timing CPU vs GPU runs, and saving/loading checkpoints. Mix and match the commands to design your own lab exercises.

## Prerequisites

1. Activate the Conda/virtualenv where NumPy, CuPy (optional), matplotlib, and tqdm are installed.
2. Place the MNIST `.npy` files inside `data/` (already provided in this repo).
3. From the project root, run the commands below.

---

## Fully Connected Networks

### Scalar Debug Mode

```bash
python scalar/main.py
```

- Trains a tiny MLP using pure Python loops.
- Prints intermediate tensors and gradients so you can follow every step.
- Ideal for tracing forward/backward propagation by hand.

### Vectorized (NumPy/CuPy) Mode

```bash
python vectorized/main.py --epochs 5 --batch-size 128      # add --gpu to use CuPy
```

- Uses batched matrix operations—orders of magnitude faster.
- Displays per-epoch progress via `tqdm`.
- Demonstrates that the vectorized math matches the scalar results.

---

## Convolutional Architectures

All CNNs share the entry point `convolutional/main.py`. The `arch` positional argument selects the model:

```
baseline, lenet, alexnet, vgg16, resnet18, efficientnet_lite0, convnext_tiny
```

### Quick Comparison (2 epochs each)

```bash
python convolutional/main.py baseline            --epochs 2 --batch-size 64 --gpu
python convolutional/main.py lenet               --epochs 2 --batch-size 64 --gpu
python convolutional/main.py alexnet             --epochs 2 --batch-size 64 --gpu
python convolutional/main.py vgg16               --epochs 2 --batch-size 64 --gpu
python convolutional/main.py resnet18            --epochs 2 --batch-size 64 --gpu
python convolutional/main.py efficientnet_lite0  --epochs 2 --batch-size 64 --gpu
python convolutional/main.py convnext_tiny       --epochs 2 --batch-size 64 --gpu
```

Observations to make:

- Training time vs architecture depth.
- Test accuracy after two epochs.
- GPU utilization compared to CPU runs (rerun without `--gpu` to see the difference).

### Save / Resume Workflow

```bash
mkdir -p checkpoints
python convolutional/main.py resnet18 --epochs 2 --batch-size 64 --gpu \
    --save checkpoints/resnet18_e2.npz

python convolutional/main.py resnet18 --epochs 1 --batch-size 64 --gpu \
    --load checkpoints/resnet18_e2.npz \
    --save checkpoints/resnet18_e3.npz
```

- The first command trains for two epochs and saves the weights.
- The second loads the checkpoint, trains one extra epoch, and writes a new file.
- Metadata (arch name, epoch count) is embedded via `common/model_io.py`.

---

## Quick-Start Scripts

Prefer a guided walkthrough? Each module has a scenario-driven helper in `scripts/`:

- `python scripts/quickstart_scalar.py --scenario basic --plot`  
  Loads MNIST, previews a handful of samples, trains the scalar MLP, and optionally plots loss/prediction grids. Alternate scenarios compare optimizers or plug in Fashion-MNIST style `.npy` files.

- `python scripts/quickstart_vectorized.py --scenario hidden-sweep --plot`  
  Sweeps over several hidden-layer configurations, printing accuracies and (optionally) plotting curves. Use `--scenario optimizer-compare` to benchmark SGD vs Adam in minutes.

- `python scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead --plot`  
  Trains ResNet18 with Lookahead + gradient clipping on GPU (falls back to CPU). Additional scenarios demonstrate CPU baselines, checkpoint resume flows, and dataset swaps (e.g., CIFAR-10 shaped data via `--image-shape 3,32,32`).

Each script exposes flags (`--epochs`, `--batch-size`, dataset overrides, `--plot`, etc.) so students can experiment interactively without editing the main entrypoints.

---

## Suggested Experiments

| Idea | What to measure |
| --- | --- |
| **CPU vs GPU timing** | Run the same architecture with and without `--gpu`. Record epoch time to quantify the hardware boost. |
| **Scalar vs Vectorized accuracy** | Train both implementations for one epoch and compare final accuracy to confirm mathematical equivalence. |
| **Architecture bake-off** | Train multiple CNNs for a fixed epoch budget and compare test accuracy vs parameter count. |
| **Hyperparameter tweaks** | Modify `--batch-size`, learning rate (edit optimizer), or augmentation toggle (`--no-augment`) to see their effect. |
| **Dropout/normalization ablations** | Temporarily disable layers (e.g., comment out BatchNorm) to witness training instability or overfitting. |

Document your findings—timing tables, accuracy plots, or misclassification grids make great lab reports.

---

## Troubleshooting Checklist

- **CuPy import error:** ensure CUDA toolkit + matching CuPy wheel are installed; run `python scripts/test_cupy.py` to confirm.
- **FileNotFoundError for data:** verify the `.npy` files exist under `data/`.
- **Out-of-memory during misclassification plots:** reduce the visualization batch size or run on CPU; the helper in `convolutional/main.py` already requests batched predictions, but huge grids can still consume memory.
- **Long training time on large nets:** start with `--epochs 1` to smoke-test your setup before longer runs.

With these commands and experiments, you can turn the sandbox into a full hands-on lab sequence.
