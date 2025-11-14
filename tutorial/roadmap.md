---
title: Learning Roadmap
---

Use this step-by-step progression to work through the NNFS sandbox. Each stage links to the relevant tutorial page(s) and suggests commands to run before moving on.

---

## Stage 0 – Environment & Data

1. **Install dependencies**

   ```bash
   pip install numpy matplotlib tqdm cupy  # CuPy optional if you have CUDA
   ```

2. **Verify GPU health**

   ```bash
   python scripts/test_cupy.py --stress-seconds 5 --stress-size 2048
   ```

3. **Check dataset**
   - Ensure `data/train_images.npy`, `data/train_labels.npy`, `data/test_images.npy`, `data/test_labels.npy` exist (see the [data README](https://github.com/polla-fattah/nnott/tree/main/data#readme) for details).

## Stage 1 – Project Orientation

1. Read **[Project Tour](project-tour.md)** to understand the directory layout and shared utilities.
2. Skim **[Implementations & Hardware](implementations-and-hardware.md)** for the scalar → vectorized → GPU progression.
3. Review **[Core Concepts](core-concepts.md)** as needed (activations, losses, optimizers, regularization).

## Stage 2 – Scalar Baseline

1. Read **[Running Experiments](running-experiments.md#fully-connected-networks)**, scalar section.
2. Run the scalar trainer with defaults:

   ```bash
    python scalar/main.py --epochs 1 --batch-size 64 --plot
   ```

3. Experiment with CLI flags: `--hidden-activations`, `--hidden-dropout`, `--no-augment`.
4. Fill an entry in **[Experiment Log Template](experiment-log-template.md)** capturing loss, accuracy, runtime.

## Stage 3 – Vectorized MLP

1. Read the vectorized section in **[Running Experiments](running-experiments.md#vectorized-numpycupy-mode)**.
2. Train with validation split + confusion matrix:

   ```bash
   python vectorized/main.py --epochs 3 --batch-size 128 \
     --val-split 0.2 --confusion-matrix \
     --augment-cutmix-prob 0.3 --augment-randaug-layers 1
   ```

3. Compare results to scalar runs; note the effect of batchnorm/dropout toggles.
4. Review the **[Augmentation Playground](augmentation-playground.md)** and mini-guides under `tutorial/augmentations/` for policy ideas.

## Stage 4 – Optimization Lab

1. Work through **[Optimization Lab](optimization-lab.md)** in Jupyter (learning-rate sweeps, gradient clipping, Lookahead).
2. Use the provided code snippets to instrument loss curves, then document findings in your experiment log.

## Stage 5 – Convolutional Architectures

1. Read **[Architecture Gallery](architecture-gallery.md)** (at least BaselineCNN, LeNet, ResNet18).
2. Train baseline/resnet with validation + confusion matrix:

   ```bash
   python convolutional/main.py baseline --epochs 3 --batch-size 64 --gpu \
     --val-split 0.1 --confusion-matrix

   python convolutional/main.py resnet18 --epochs 3 --batch-size 64 --gpu \
     --augment-cutmix-prob 0.3 --augment-randaug-layers 2 --confusion-matrix
   ```

3. Use `--save` / `--load` to practice checkpointing, then inspect misclassifications with `--show-misclassified`.

## Stage 6 – Tools & Utilities

1. Dive into **[Tools Quick Notes](tools/index.md)** (NumPy, matplotlib, tqdm, CuPy) to ensure you understand the ecosystem.
2. Explore `scripts/`:
   - `scripts/quickstart_scalar.py --scenario basic`
   - `scripts/quickstart_vectorized.py --scenario optimizer-compare`
   - `scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead`
3. Consider building a new script or preset augmentation config to automate future labs.

## Stage 7 – Reflection & Next Steps

1. Summarize key learnings in your experiment log (top 3 takeaways, biggest pain points).
2. Plan follow-up ideas:
   - Swap datasets (e.g., Fashion-MNIST using `DataUtility.load_data(...)` overrides).
   - Add new architectures or augmentations.
   - Automate metric logging (CSV/TensorBoard) or benchmarking.
