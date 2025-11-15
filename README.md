# Neural Networks Optimization and Training Tutorial (NNOTT)

This repository is an educational sandbox for implementing neural networks from first principles. You’ll find:

- **Scalar MLPs** built entirely with Python loops for maximum transparency.
- **Vectorized MLPs** that showcase how NumPy/CuPy acceleration changes performance.
- **Convolutional networks** ranging from LeNet to ConvNeXt-Tiny, all sharing a unified trainer and backend.
- **Shared utilities** for dataset loading, backend switching, model I/O, and reproducible experiments.

## Tutorial Series

The detailed learning materials now live under [`tutorial/`](tutorial/README.md). Start there to explore:

1. Project layout and supporting tools.
2. Scalar → vectorized → GPU implementations.
3. Step-by-step experiment guides and checkpoint workflows.
4. Deep-learning concept notes (losses, activations, optimizers, normalization, etc.).
5. Architectural deep-dives for every CNN in the repo plus dataset information.

## Quick Start

```bash
# install dependencies (inside your env)
pip install numpy matplotlib tqdm cupy  # cupy optional but recommended

# verify GPU health
python scripts/test_cupy.py --stress-seconds 5 --stress-size 2048

# run a CNN (GPU optional)
python convolutional/main.py resnet18 --epochs 2 --batch-size 64 --gpu
```

Need more context? Jump to the [tutorial hub](tutorial/README.md) for the full walkthrough.

---

## MkDocs Documentation

A new documentation portal is being built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

1. Ensure **mkdocs** and **mkdocs-material** are installed in your environment:

   ```bash
   pip install mkdocs mkdocs-material
   ```

2. Serve locally from the repository root:

   ```bash
   mkdocs serve
   ```

   Open the listed localhost URL to preview the docs.

3. Build the static site:

   ```bash
   mkdocs build
   ```

   The generated HTML will be in the `site/` directory (ignored by git).

The original markdown content is still available under `old_tutorial/` until the migration is complete.
