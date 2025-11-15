---
title: LeNet-5 (1998)
---


![LeNet-5 diagram](https://en.wikipedia.org/wiki/LeNet#/media/File:LeNet-5_architecture.svg)
Figure credit: Yann LeCun et al., via Wikipedia (CC BY-SA 3.0).

## Historical Context

LeNet-5, introduced by Yann LeCun and colleagues, is one of the first successful convolutional neural networks. It was designed for handwritten digit recognition on bank checks and showed that local receptive fields plus pooling dramatically outperform dense networks on images.

## Architecture Overview

1. **Conv Layer 1:** 6 filters of size 5×5, followed by tanh/ReLU and 2×2 subsampling.
2. **Conv Layer 2:** 16 filters, again followed by subsampling.
3. **Fully Connected Stack:** Two dense layers leading to a 10-way output with softmax.

In this project, the implementation mirrors the original layout but uses ReLU activations for better gradient flow.

## Implementation Notes

- **File:** `convolutional/architectures/lenet.py`
- **Layers:** All building blocks come from `convolutional/modules.py`.
- Padding is used to maintain spatial dimensions before pooling, keeping the math close to the canonical LeNet.

## Teaching Angles

- Demonstrates how convolutions reduce parameter counts compared to dense layers.
- Illustrates early use of pooling (subsampling) for translational invariance.
- Perfect baseline for MNIST—students can read the entire architecture in a few minutes.

## Suggested Experiments

- Train for 1–2 epochs with and without augmentation to see robustness changes.
- Compare training time vs BaselineCNN or modern nets to emphasize efficiency.

## References

- [LeNet on Wikipedia](https://en.wikipedia.org/wiki/LeNet)

