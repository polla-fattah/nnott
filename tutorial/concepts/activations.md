# Activation Functions

**Breadcrumb:** [Home](../README.md) / [Core Concepts](../core-concepts.md) / Activation Functions


Activations introduce non-linearity so networks can model complex decision boundaries. This project implements several classics and modern favorites. All activation modules live in `convolutional/modules.py`, and simpler scalar versions appear in `scalar/neuron.py`.

## ReLU (Rectified Linear Unit)

- **Definition:** `ReLU(x) = max(0, x)`
- **Derivative:** 1 for `x > 0`, 0 otherwise.
- **Use in project:** Default activation in most conv/dense layers. Found in BaselineCNN, LeNet, AlexNet, VGG, and ResNet blocks.
- **Why it matters:** Keeps gradients alive for positive activations, combats vanishing gradients, cheap to compute.

## Leaky ReLU

- **Definition:** `max(αx, x)` with a small α (e.g., 0.01).
- **Use:** Provided for experimentation; prevents neurons from dying by allowing a tiny gradient for `x < 0`.

## SiLU (Swish)

- **Definition:** `x · sigmoid(x)`
- **Where used:** EfficientNet-Lite0 MBConv blocks.
- **Benefit:** Smooth, non-monotonic activation that empirically improves performance in lightweight models.

## GELU (Gaussian Error Linear Unit)

- **Definition:** `0.5 x (1 + tanh(√(2/π)(x + 0.044715x^3)))`
- **Where used:** ConvNeXt blocks, mirroring transformer activations.
- **Reason:** Provides smoother transitions than ReLU and works well with LayerNorm-based architectures.

## Softmax

- **File:** `common/softmax.py`
- **Purpose:** Converts logits into probabilities for multi-class classifiers.
- **Usage:** Final layer in MLPs and CNNs before cross-entropy loss.

## Study Checklist

1. Trace how each activation is implemented in `convolutional/modules.py`.
2. Replace ReLU with another activation in a small model and observe training behavior.
3. Compare gradient distributions (using prints or histograms) to see why smooth activations can aid optimization.