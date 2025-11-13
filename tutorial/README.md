---
**NNOTT Tutorial Series**
---

# Neural Networks From Scratch - Tutorial Hub

Welcome to the learning guide for the **Neural Networks From Scratch (NNFS)** sandbox. This tutorial series walks through the entire project—why it exists, how it is organized, and how to run meaningful experiments. Use the navigation list below to jump to the relevant module.

## How to Use This Tutorial

1. **Start with the Project Tour** to understand the repository layout and supporting tools.
2. **Study the Implementation Progression** (scalar to vectorized to GPU) to see how performance grows from simple loops to accelerated kernels.
3. **Follow the Experiment Playbook** to run the MLP and CNN scripts, save checkpoints, and compare results.
4. **Review the Concept Notes** whenever you need a refresher on activations, losses, optimizers, or regularization tricks.
5. **Explore the Architecture Gallery** to learn how classic and modern CNNs (LeNet through ConvNeXt) are assembled from the provided building blocks.

## Tutorial Modules

| Module | Description |
| --- | --- |
| [Project Tour](project-tour.md) | Repository structure, shared utilities, datasets, and helper scripts. |
| [Implementations & Hardware](implementations-and-hardware.md) | Scalar vs vectorized code paths, backend switching, and GPU diagnostics. |
| [Running Experiments](running-experiments.md) | Training commands, checkpoint workflows, quick-start scripts, and lab prompts. |
| [Core Concepts](core-concepts.md) | Notes on neurons, activations, normalization, losses, optimizers, and convolution tricks. |
| [Architecture Gallery](architecture-gallery.md) | Background on each CNN (LeNet, AlexNet, VGG16, ResNet18, EfficientNet-Lite0, ConvNeXt-Tiny) plus dataset considerations. |
| [Debug Playbook](debug-playbook.md) | Troubleshooting recipes for environment, GPU, data, and training issues. |
| [Optimization Lab](optimization-lab.md) | Jupyter walkthrough for tuning learning rate, gradient clipping, Lookahead, and batch sizes. |
| [Activation Zoo](activation-zoo.md) | Side-by-side comparison of ReLU, LeakyReLU, SiLU, and GELU with sample plots. |
| [Experiment Log Template](experiment-log-template.md) | Markdown template for recording dataset, hyperparameters, results, and observations. |
| [Performance Profiles](performance-profiles.md) | Typical training times and accuracies (CPU vs GPU) for each architecture. |
| [Augmentation Playground](augmentation-playground.md) | Guide for extending `_augment_batch` with rotations, flips, noise, and cutout. |

## Scenario Reference Table

| Goal | Fastest Way to Try |
| --- | --- |
| Learn optimizer behavior | `python scripts/quickstart_vectorized.py --scenario optimizer-compare --plot` |
| Compare CPU vs GPU throughput | `python scripts/quickstart_convolutional.py --scenario gpu-fast --epochs 1 --lookahead --plot` (run once with `--gpu`, once without) |
| Practice checkpoint save/resume | `python scripts/quickstart_convolutional.py --scenario resume-demo --save-path checkpoints/demo.npz --plot` |
| Swap in a new dataset | `python scripts/quickstart_scalar.py --scenario dataset-swap --plot --alt-train-images fashion_train_images.npy ...` |
| Test gradient clipping or Lookahead | `python convolutional/main.py resnet18 --epochs 1 --batch-size 64 --gpu --grad-clip 5 --lookahead --lookahead-k 5 --lookahead-alpha 0.5` |
| Visualize misclassifications | `python vectorized/main.py --epochs 2 --batch-size 64 --plot` (answer "y" when prompted) **or** `python convolutional/main.py baseline --plot --show-misclassified` (confirm the extra pass) |
| Stress-test CuPy/GPU | `python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096` |
| Scalar loop walkthrough | `python scripts/quickstart_scalar.py --scenario basic --plot` |

> Tip: Each module is self-contained. Read them in order or jump directly to the section that matches your current work.

---
Return to [Tutorial Hub](README.md)
