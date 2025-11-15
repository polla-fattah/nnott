---
title: Activation Functions
---


This page compares the activation functions implemented across the project so you can see how they shape neuron outputs and gradients. Each section includes the definition, derivative, usage context, and a quick snippet for plotting in a notebook.

!!! tip "Interactive plots"
    Copy the sample code cells into a Jupyter notebook (see the Optimization Lab) for interactive plots. The equations below match what you’ll find in `convolutional/modules.py`.

---

## 1. ReLU (Rectified Linear Unit)

- **Definition:** \( \mathrm{ReLU}(x) = \max(0, x) \)
- **Derivative:** \( \frac{d}{dx} = 1 \) for \( x > 0 \), \( 0 \) otherwise.
- **Usage:** Default activation throughout BaselineCNN, LeNet, AlexNet, VGG16, ResNet18, and many MLP layers.
- **Intuition:** Fast, sparse activations; prevents vanishing gradients for positive inputs.

```python
import numpy as np, matplotlib.pyplot as plt
x = np.linspace(-3, 3, 400)
y = np.maximum(0, x)
dy = (x > 0).astype(float)
plt.plot(x, y, label="ReLU")
plt.plot(x, dy, label="Derivative", linestyle="--")
plt.legend(); plt.grid(True); plt.show()
```

---

## 2. Leaky ReLU

- **Definition:** \( \max(\alpha x, x) \) with small \( \alpha \) (e.g., 0.01).
- **Derivative:** \( 1 \) for \( x > 0 \), \( \alpha \) otherwise.
- **Usage:** Available for experimentation in the MLP modules; useful when you worry about “dead” ReLUs.
- **Intuition:** Allows a trickle of gradient on negative inputs so neurons stay active.

```python
alpha = 0.1
y = np.where(x > 0, x, alpha * x)
dy = np.where(x > 0, 1.0, alpha)
```

---

## 3. SiLU / Swish

- **Definition:** \( \mathrm{SiLU}(x) = x \cdot \sigma(x) \), where \( \sigma(x) \) is the sigmoid.
- **Derivative:** \( \sigma(x) + x \cdot \sigma(x) \cdot (1 - \sigma(x)) \)
- **Usage:** EfficientNet-Lite0 MBConv blocks (`convolutional/modules.py` supplies a `SiLU` class).
- **Intuition:** Smooth, non-monotonic activation; often improves accuracy in lightweight networks.

```python
sig = 1 / (1 + np.exp(-x))
y = x * sig
dy = sig + x * sig * (1 - sig)
```

---

## 4. GELU (Gaussian Error Linear Unit)

- **Definition:** \( \mathrm{GELU}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right) \) (approximation used in practice).
- **Derivative:** More complex; frameworks typically compute it automatically (see `GELU` class for the exact expression).
- **Usage:** ConvNeXt blocks in `convolutional/architectures/convnext.py`.
- **Intuition:** Smooths the activation transition and behaves well with LayerNorm-heavy architectures.

```python
import numpy as np
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
y = gelu(x)
```

## Quick Comparison Table

| Activation | Pros | Cons | Typical Use |
| --- | --- | --- | --- |
| ReLU | Simple, fast, prevents vanishing gradients | “Dead” neurons for large negative weights | Default in most CNN/MLP layers |
| Leaky ReLU | Keeps a gradient on negative side | Extra hyperparameter \( \alpha \) | Experimental variants of CNN/MLP |
| SiLU / Swish | Smooth, empirically strong in small nets | Slightly more compute (sigmoid) | EfficientNet-style blocks |
| GELU | Smooth, transformer-friendly | More expensive to compute, derivative complicated | ConvNeXt, transformer-inspired CNNs |

## Suggested Notebook Exercise

1. Copy the plotting snippets into a single Jupyter cell.
2. Plot all four activations and their derivatives on the same axes.
3. Feed a random batch through each activation and measure the mean/std of the outputs—note how ReLU truncates negatives while SiLU/GELU keep them but dampen their magnitude.

Document your observations in the notebook so you can refer back when choosing activations for custom architectures.
