# Neural Networks Optimization and Training Tutorial (NNOTT)

NNOTT is a hands-on lab for understanding neural networks across multiple implementation tiers: **scalar loops**, **vectorized NumPy/CuPy**, and **production-style convolutional architectures**. Each tier has matching tutorials, scripts, and checkpoints so you can see how the same ideas scale from notebook math to GPU-ready pipelines. Browse the full tutorial portal at **https://polla-fattah.github.io/nnott** (or read the Markdown directly under `tutorial/`).

## What’s Inside

- **Scalar MLP stack (`scalar/`)** – loop-based neurons, layers, and trainers. Perfect for tracing how gradients propagate, seeing accumulation order, and teaching the math line-by-line.
- **Vectorized MLP stack (`vectorized/`)** – the same networks rewritten with matrix ops (NumPy on CPU, CuPy when `--gpu` is passed). Includes batch norm, dropout, advanced LR schedules, Lookahead, gradient clipping, and augmentation knobs.
- **CNN suite (`convolutional/architectures/`)** – a gallery ranging from compact LeNet and BaselineCNN to AlexNet, VGG16, ResNet18, EfficientNet-Lite0, ConvNeXt-Tiny, plus the scaffolding required to run them (custom Conv2D/DepthwiseConv, BatchNorm, SiLU/GELU activations, SE blocks, etc.).
- **Backends & utilities (`common/`, `scripts/`)** – dataset management, `xp` proxy for CPU↔GPU switching, checkpoint I/O, quickstart scripts that launch structured experiments (scalar/vectorized/CNN) with a single command.
- **Tutorial portal (`tutorial/`)** – full MkDocs site with concept primers, experiment guides, augmentation labs, optimizer walkthroughs, architecture references, and performance benchmarks that mirror the code.

## Repository Map

| Path | Highlights |
| --- | --- |
| `scalar/` | Neuron, Layer, Trainer classes built with pure Python loops; ideal for pedagogy, debugging, and tracing gradients. |
| `vectorized/` | Linear/Conv modules, optimizers, and trainers implemented with matrix ops; supports batch norm, dropout, Lookahead, LR schedules, augmentation hooks. |
| `convolutional/` | Custom CNN layers (Conv2D, DepthwiseConv2D, SE, LayerNorm2D, GlobalAvgPool), plus architecture definitions for BaselineCNN, LeNet, AlexNet, VGG16, ResNet18, EfficientNet-Lite0, ConvNeXt-Tiny, etc. |
| `common/` | Backend selector (`common/backend.py`), dataset utilities, checkpoint IO, shared math helpers. |
| `scripts/` | Scenario-driven launchers (quickstart_scalar/vectorized/convolutional), GPU stress tests, profiling helpers. |
| `tutorial/` | MkDocs-ready curriculum: concept notes (activations, losses, optimizers, normalization), experiment playbooks, architecture deep dives, augmentation labs, debug playbooks. |

## Three Ways to Build a Network

1. **Scalar-first (loop-based)**
   - Every neuron, activation, and layer is expressed as explicit Python loops.
   - Prints intermediate tensors to the console so you can trace forward/backward passes exactly.
   - Great for labs where you derive gradients by hand and then validate them step-by-step.

2. **Vectorized (NumPy/CuPy)**
   - Replaces loops with matrix operations (`xp` proxy points to NumPy or CuPy).
   - Adds modern training staples: BatchNorm, Dropout, LR schedules (multistep, cosine, ReduceLROnPlateau), Lookahead, gradient clipping, augmentation toggles.
   - Demonstrates how simple architectural changes translate from scalar prototypes to high-throughput implementations.

3. **Full CNN stack**
   - Custom Conv2D/DepthwiseConv2D, pooling, normalization, activation, SE blocks, Residual/MBConv/ConvNeXt blocks.
   - Unified trainer handles datasets, augmentation pipelines, CLI flags, checkpoint save/resume, GPU vs CPU switching.
   - Mirrors reference architectures so you can compare BaselineCNN ↔ LeNet ↔ AlexNet ↔ VGG16 ↔ ResNet18 ↔ EfficientNet-Lite0 ↔ ConvNeXt-Tiny without changing tooling.

Each tier includes matching notebooks/tutorial chapters plus quick-start scripts so you can hop in at the fidelity you need.

## Architecture Lineup

| Model | Highlights | Where to start |
| --- | --- | --- |
| **BaselineCNN** | Lightweight conv net tuned for MNIST, perfect for first GPU demos. | `convolutional/architectures/baseline.py` |
| **LeNet-5** | Classic 1998 architecture with subsampling, demonstrates early CNN design. | `convolutional/architectures/lenet.py` |
| **AlexNet** | Large kernels + dropout + data augmentation, illustrates the ImageNet 2012 breakthrough. | `convolutional/architectures/alexnet.py` |
| **VGG16** | Uniform 3×3 conv stacks, massive parameter counts—great for memory profiling. | `convolutional/architectures/vgg.py` |
| **ResNet18** | Residual blocks and skip connections, shows gradient-preserving design. | `convolutional/architectures/resnet.py` |
| **EfficientNet-Lite0** | MBConv + SE attention, compound scaling ideas in a mobile-friendly package. | `convolutional/architectures/efficientnet.py` |
| **ConvNeXt-Tiny** | Post-ViT CNN with depthwise convs, LayerNorm2D, GELU; reveals modern conv trends. | `convolutional/architectures/convnext.py` |

Every architecture ships with:

- Matching tutorial page (history, diagrams, experiments, references).
- CLI instructions for training/testing, checkpointing, and augmentation toggles.
- Guidance for CPU vs GPU runs, gradient clipping, LR schedules, misclassification visualization, etc.

## Getting Started

```bash
# inside your virtualenv/conda env
pip install numpy matplotlib tqdm
# optional but recommended for GPU acceleration
pip install cupy-cuda12x  # pick the wheel that matches your CUDA version

# verify GPU + CuPy
python scripts/test_cupy.py --stress-seconds 5 --stress-size 2048
```

## Running the Examples

### Scalar MLP (teaching path)

```bash
python scalar/main.py --epochs 1 --batch-size 64 \
    --hidden-sizes 256,128,64 --hidden-activations relu,gelu,tanh \
    --hidden-dropout 0.3,0.2,0.1 --plot
```

### Vectorized MLP (NumPy/CuPy)

```bash
python vectorized/main.py --epochs 3 --batch-size 128 --gpu \
    --hidden-sizes 512,256,128 --batchnorm --dropout 0.2 \
    --lr-schedule cosine --val-split 0.1 --early-stopping
```

### CNN entry point

```bash
python convolutional/main.py resnet18 --epochs 2 --batch-size 64 --gpu \
    --grad-clip 5 --lookahead --lookahead-k 5 --lookahead-alpha 0.5
```

### Scenario-driven quickstarts

```bash
python scripts/quickstart_scalar.py --scenario basic --plot
python scripts/quickstart_vectorized.py --scenario hidden-sweep --batchnorm --plot
python scripts/quickstart_convolutional.py --scenario gpu-fast --lookahead --plot
```

Each quickstart script narrates what it’s doing, prompts before long plotting passes, and exposes the same CLI flags as the main entry points so you can experiment quickly.

## Learning Materials

The tutorial hub (`tutorial/index.md` plus the rest of `tutorial/`) mirrors the codebase:

- **Concept primers** on activations, loss functions, optimizers, normalization, regularization, augmentation, etc.
- **Experiment guides** for scalar/vectorized/CNN workflows, checkpointing, performance profiling, and troubleshooting.
- **Architecture notes** with diagrams, historical context, and per-model references.
- **Labs** for augmentation, optimizer tinkering, and debugging playbooks.

Serve the tutorial locally through MkDocs (instructions below) or read the Markdown directly inside this repo.

## Documentation (MkDocs Material)

MkDocs is simply the publishing tool for the tutorial content. To preview the same site locally:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve  # visit the printed localhost URL
```

To build the static site:

```bash
mkdocs build
```

The generated HTML lands in `site/` (ignored by git). The MkDocs configuration lives in `mkdocs.yml`, and overrides/assets sit under `overrides/` and `tutorial/`.
