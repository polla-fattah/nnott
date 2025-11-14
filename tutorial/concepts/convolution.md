---
title: Convolution Mechanics & Advanced Blocks
---

Convolutional layers are the backbone of the CNN architectures in this sandbox. This note explains how they’re implemented via `im2col`, and how advanced blocks build on top of them.

## Conv2D via im2col

- **File:** `convolutional/modules.py` (`class Conv2D`)
- **Key functions:** `_im2col` and `_col2im`
- **Process:**
  1. Pad the input tensor (if needed).
  2. Extract sliding windows (patches) using `sliding_window_view` or manual strides.
  3. Reshape patches into a 2D matrix (`cols`) where each row corresponds to one receptive field.
  4. Flatten filters into a matrix (`W_col`) and perform a dense matrix multiply.
  5. Reshape the result back to `(N, out_channels, H_out, W_out)`.
- **Backward pass:** Uses `_col2im` to scatter gradients from the matrix form back into the original spatial layout.

## Depthwise Convolution

- **File:** `convolutional/modules.py` (`class DepthwiseConv2D`)
- Processes each channel independently, drastically reducing compute.
- Used in EfficientNet-Lite0 and ConvNeXt blocks as part of MBConv/inverted bottlenecks.

## Squeeze-and-Excitation (SE)

- **File:** `convolutional/modules.py` (`class SqueezeExcite`)
- Steps: global average pool → small MLP (reduce-then-expand) → sigmoid gating per channel.
- Purpose: Reweight feature maps so important channels are amplified.

## Global Average Pooling

- **File:** `convolutional/modules.py` (`class GlobalAvgPool2D`)
- Replaces large fully connected heads by averaging each channel over spatial dimensions.
- Used in ResNet, EfficientNet, and ConvNeXt classifiers.

## Putting It Together

- **MBConv (EfficientNet):** 1×1 expansion → depthwise conv → SE → 1×1 projection, plus residual if shapes match.
- **ConvNeXt block:** 7×7 depthwise conv → LayerNorm → pointwise expansion (ratio 4) → GELU → pointwise projection → residual add.

## Exercises

1. Inspect `_im2col` and `_col2im` to understand how memory layout tricks speed up convolution.
2. Modify kernel size or stride in `Conv2D` and verify the output dimensions using `_compute_output_dim`.
3. Visualize SE weights to see which channels the network deems important for specific digits.

[Previous (Backend & Device Utilities)](backend.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Fundamentals: Neurons, Layers, Networks)](fundamentals.md)

[Back to Core Concepts](../core-concepts.md)
