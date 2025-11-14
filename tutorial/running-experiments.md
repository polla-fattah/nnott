---
title: Running Experiments
---

[MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# 03 - Running Experiments



This guide walks you through training the provided models, timing CPU vs GPU runs, and saving/loading checkpoints. Mix and match the commands to design your own lab exercises.

## Prerequisites

1. Activate the Conda/virtualenv where NumPy, CuPy (optional), matplotlib, and tqdm are installed.
2. Place the MNIST `.npy` files inside `data/` (already provided in this repo).
3. From the project root, run the commands below.

> Label sanity checks now run automatically. If your labels are one-hot matrices or include values outside `[0, 9]`, the trainer raises a descriptive error before training starts.

---

## Reproducibility & Monitoring

- All entry points accept `--seed` (default `42`). Set it explicitly (`--seed 1337`) to make shuffling, augmentation, and optimizer initialization deterministic across runs.
- `--val-split` (default `0.1`) automatically carves out a validation subset from the training data; epoch logs now include both train and validation loss/accuracy.
- Append `--confusion-matrix` to dump class-by-class counts after the final test evaluation—handy for spotting systematic mistakes.

Example one-liners:

```bash
# Scalar MLP with deterministic split + confusion matrix
python scalar/main.py --epochs 2 --batch-size 64 \
    --seed 7 --val-split 0.15 --confusion-matrix

# Vectorized MLP on GPU with CutMix + RandAugment, logging val metrics
python vectorized/main.py --epochs 3 --batch-size 128 --gpu \
    --seed 2024 --val-split 0.2 --augment-cutmix-prob 0.3 \
    --augment-randaug-layers 2 --confusion-matrix

# CNN baseline with validation monitoring and confusion matrix
python convolutional/main.py baseline --epochs 3 --batch-size 64 --gpu \
    --seed 99 --val-split 0.1 --confusion-matrix
```

---

## Fully Connected Networks

### Scalar Debug Mode

```bash
python scalar/main.py --epochs 1 --batch-size 64 \
    --hidden-sizes 256,128,64 \
    --hidden-activations relu,leaky_relu,tanh \
    --hidden-dropout 0.3,0.2,0.1 \
    --plot
```

- Still the pure-Python teaching model, now with activation-aware initialization (He for ReLU/LeakyReLU/GELU, Xavier for tanh/sigmoid) tied to each layer's selected activation.
- `--hidden-activations` lets you mix activations per layer without touching source files; provide a comma-separated list that matches `--hidden-sizes`.
- `--hidden-dropout` inserts dropout after every hidden block so you can demo regularization directly in the scalar path.
- `--plot` remains opt-in; leave it off whenever you want to skip matplotlib windows.

### Vectorized (NumPy/CuPy) Mode

```bash
python vectorized/main.py --epochs 3 --batch-size 128 --gpu \
    --hidden-sizes 512,256,128 \
    --hidden-activations relu,gelu,tanh \
    --dropout 0.2 \
    --batchnorm
```

- Shares all scalar flags plus `--batchnorm` (adds `BatchNorm1D` between Linear and activation) and `--bn-momentum` to tune running-stat updates.
- `--dropout` accepts either a single value or comma-separated list; semantics match `--hidden-dropout`.
- `--leaky-negative-slope` controls the slope when you include LeakyReLU in the activation list.
- `--lr-schedule cosine` / `--lr-schedule reduce_on_plateau` plus `--min-lr`, `--reduce-factor`, `--reduce-patience`, and `--reduce-delta` let you test advanced learning-rate policies; pair them with `--early-stopping --early-patience 4 --early-delta 5e-4` to halt when validation loss stalls (validation split driven by `--val-split`, default 0.1).
- Augmentation knobs are now built-in: tweak `--augment-max-shift`, `--augment-rotate-deg`, `--augment-hflip-prob`, `--augment-vflip-prob`, `--augment-noise-std`, etc., or disable everything with `--no-augment`.
- Note: the vectorized trainer is CPU/NumPy-only today. Use the convolutional entrypoint (`convolutional/main.py --gpu ...`) when you need CuPy acceleration.

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
- Want heavier augmentation? Add flags such as `--augment-rotate-deg 15 --augment-hflip-prob 0.5 --augment-noise-std 0.03` (or turn everything off with `--no-augment`).

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

- `python scripts/quickstart_scalar.py --scenario basic --epochs 1 --hidden-activations relu,gelu,tanh --hidden-dropout 0.25 --plot`  
  Loads MNIST, previews a handful of samples, trains the scalar MLP, and optionally plots loss/prediction grids. Alternate scenarios compare optimizers or plug in Fashion-MNIST style `.npy` files, and every scenario now honors the `--hidden-sizes`, `--hidden-activations`, and `--hidden-dropout` flags.

- `python scripts/quickstart_vectorized.py --scenario hidden-sweep --batchnorm --dropout 0.2 --lr-schedule cosine --val-split 0.1 --early-stopping --augment-rotate-deg 15 --augment-noise-std 0.03 --plot`  
  Sweeps over several hidden-layer configurations, printing accuracies and (optionally) plotting curves. Use `--scenario optimizer-compare` to benchmark SGD vs Adam; mix activations via `--hidden-activations relu,tanh,gelu`, tune learning-rate schedules via `--lr-schedule ...`, toggle richer augmentation via the `--augment-*` flags, and enable early stopping without editing code. Answer "yes" to the prompt before misclassification plotting (it triggers an extra pass).

- `python scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead --plot`  
  Trains ResNet18 with Lookahead + gradient clipping on GPU (falls back to CPU). Additional scenarios cover CPU baselines, checkpoint resume flows, and dataset swaps (e.g., CIFAR-10 shaped data via `--image-shape 3,32,32`). You will be prompted before the misclassification plots run; decline if you want to skip the extra inference sweep.

Each script exposes flags (`--epochs`, `--batch-size`, dataset overrides, `--plot`, etc.) so students can experiment interactively without editing the main entrypoints.

---

## Advanced Training Recipes

| Goal | Command |
| --- | --- |
| Cosine LR + early stop on full MLP | `python vectorized/main.py --epochs 15 --batch-size 128 --hidden-sizes 512,256,128 --hidden-activations relu,gelu,tanh --dropout 0.2 --batchnorm --lr-schedule cosine --min-lr 5e-5 --val-split 0.1 --early-stopping --early-patience 4 --early-delta 5e-4 --gpu` |
| Plateau LR schedule via quick-start | `python scripts/quickstart_vectorized.py --scenario optimizer-compare --epochs 6 --lr-schedule reduce_on_plateau --reduce-factor 0.4 --reduce-patience 2 --val-split 0.1 --early-stopping --plot` |
| Compare stopping behavior across architectures | `python scripts/quickstart_vectorized.py --scenario hidden-sweep --epochs 8 --hidden-options "512,256;256,128,64" --hidden-activations relu,gelu,tanh --dropout 0.2 --batchnorm --lr-schedule cosine --val-split 0.1 --early-stopping` |
| Resume CNN run with scheduler | `python convolutional/main.py resnet18 --epochs 3 --batch-size 128 --gpu --save checkpoints/resnet18_cosine.npz --lr-schedule cosine --lr-decay-min 1e-5` (then rerun with `--load checkpoints/resnet18_cosine.npz --epochs 2` to continue) |

Tips:
- Watch the printed LR each epoch to confirm the schedule is working.
- Validation loss only appears when `--val-split > 0` (or the quick-start scenario carries a validation set); early stopping relies on that signal.
- For rapid prototyping, drop `--epochs` to 2–3 and disable plotting. Once a schedule looks promising, bump epochs and re-enable visualizations.

---

## Suggested Experiments

| Idea | What to measure |
| --- | --- |
| **CPU vs GPU timing** | Run the same architecture with and without `--gpu`. Record epoch time to quantify the hardware boost. |
| **Scalar vs Vectorized accuracy** | Train both implementations for one epoch and compare final accuracy to confirm mathematical equivalence. |
| **Architecture bake-off** | Train multiple CNNs for a fixed epoch budget and compare test accuracy vs parameter count. |
| **Hyperparameter tweaks** | Modify `--batch-size`, learning rate (edit optimizer), or augmentation toggle (`--no-augment`) to see their effect. |
| **Dropout/normalization ablations** | Temporarily disable layers (e.g., comment out BatchNorm) to witness training instability or overfitting. |

Document your findings: timing tables, accuracy plots, or misclassification grids make great lab reports.

## Lab Challenges

1. **Optimizer notebook:** Use `scripts/quickstart_vectorized.py --scenario optimizer-compare --plot` to capture loss curves for SGD and Adam on the same dataset. Write a short paragraph explaining which optimizer converged faster and why.
2. **ResNet stress test:** Run `python scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead --plot` on a GPU (or CPU fallback). Measure runtime, GPU memory usage, and final accuracy, then describe one tweak that could reduce memory pressure.
---

## Troubleshooting Checklist

- **CuPy import error:** ensure CUDA toolkit + matching CuPy wheel are installed; run `python scripts/test_cupy.py` to confirm.
- **FileNotFoundError for data:** verify the `.npy` files exist under `data/`.
- **Out-of-memory during misclassification plots:** reduce the visualization batch size or run on CPU; the helper in `convolutional/main.py` already requests batched predictions, but huge grids can still consume memory.
- **Long training time on large nets:** start with `--epochs 1` to smoke-test your setup before longer runs.

With these commands and experiments, you can turn the sandbox into a full hands-on lab sequence.

---

