# 04 · Core Concepts in Practice

This module connects classroom theory to the exact places in the codebase where each concept lives. Use it as a reference when you want to see how a mathematical idea is spelled out in Python.

## Neurons, Layers, and Networks

- **Files:** `scalar/neuron.py`, `scalar/layer.py`, `scalar/network.py`
- **Concept:** A neuron computes a weighted sum plus bias, then applies an activation. Layers are sequences of neurons; networks are compositions of layers.
- **In code:** The scalar implementation stores intermediate values for both forward and backward passes, letting you inspect how gradients propagate through each parameter.

## Activation Functions

| Activation | Files | Highlights |
| --- | --- | --- |
| ReLU / LeakyReLU | `convolutional/modules.py` | Piecewise-linear functions that keep gradients alive on positive inputs (ReLU) or allow a small negative slope (LeakyReLU). |
| SiLU (Swish) | `convolutional/modules.py` | Smooth, non-monotonic activation defined as `x * sigmoid(x)`. Used in EfficientNet-style blocks. |
| GELU | `convolutional/modules.py` | Gaussian Error Linear Unit, popular in transformers/ConvNeXt. Provides smooth saturation and has become a default in modern CNNs. |
| Softmax | `common/softmax.py` | Converts logits to probabilities—used in output layers for classification. |

## Loss Function: Cross-Entropy from Logits

- **File:** `common/cross_entropy.py`
- **Key idea:** Compute cross-entropy directly from logits using the log-sum-exp trick to avoid overflow/underflow.
- **Implementation notes:**
  - Converts target labels (indices or one-hot vectors) into integer class IDs on the active backend.
  - Uses `logsumexp` to evaluate `-z_y + log(sum(exp(z)))` safely.
  - Provides both value and gradient routines so trainers can call `forward`/`backward` cleanly.

## Optimizers

| Optimizer | File | Features |
| --- | --- | --- |
| SGD | `vectorized/optim.py` | Supports momentum, optional Nesterov acceleration, weight decay, and batch-size-aware gradients. |
| Adam | `vectorized/optim.py` | Maintains first/second-moment estimates (`m`, `v`), applies bias correction, and works seamlessly on CPU or GPU via the backend proxy. |

Both optimizers integrate with the trainer’s learning-rate schedule: see `ConvTrainer._default_multistep_schedule`, which drops the LR at ~50% and ~75% of the total epochs.

## Regularization Techniques

- **Dropout (`convolutional/modules.py`):** During training, multiplies activations by a Bernoulli mask and scales by the keep probability; disabled during evaluation.
- **Weight Decay:** Implemented inside optimizers by adding `weight_decay * param` to the gradient before the update.
- **Data Augmentation:** `ConvTrainer._augment_batch` performs random pixel shifts (±2) via `xp.roll` to teach translational robustness.

## Normalization Layers

| Layer | Purpose |
| --- | --- |
| `BatchNorm2D` | Normalizes activations across the batch/channel axes, tracks running mean/variance for inference, and has learnable scale/shift parameters. |
| `LayerNorm2D` | Normalizes across feature channels within each sample, independent of batch size—used in ConvNeXt blocks. |

Both layers stabilize training by keeping activations in a manageable range, which supports higher learning rates and faster convergence.

## Convolution Mechanics

- **Conv2D (`convolutional/modules.py`):** Implements convolution via the `im2col` + matrix-multiply trick. Steps:
  1. Pad the input tensor (if needed).
  2. Use `sliding_window_view` (or manual striding) to extract receptive fields.
  3. Reshape those patches into a 2D matrix (`cols`) and perform a dense matmul with flattened filters.
  4. Reshape the result back to `(batch, channels, height, width)`.
- **col2im:** Reverses the process during backprop to map gradients back onto the original input tensor.
- **DepthwiseConv2D / SqueezeExcite / GlobalAvgPool2D:** Provide modern building blocks used by EfficientNet-Lite0 and ConvNeXt—depthwise separable convolutions, channel attention, and spatial pooling.

## Backend Helpers

- `backend.to_device`: Moves NumPy arrays onto the active backend (no-op if already there).
- `backend.to_cpu`: Brings data back to NumPy, used for logging/plotting.
- `backend.get_array_module`: Returns `np` or `cp` depending on a tensor’s type—handy when writing generic utilities.

Understanding these helpers clarifies how the same trainer code can run unchanged on CPU or GPU.

---

Keep this sheet open while you explore the code. When you hear terms like “cross-entropy”, “BatchNorm”, or “depthwise convolution” in lecture, you now know exactly where to find them in the repo.
