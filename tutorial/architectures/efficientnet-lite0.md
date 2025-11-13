---
**NNOTT Tutorial Series**
---

# EfficientNet-Lite0 (2019)

**Breadcrumb:** [Home](../README.md) / [Architecture Gallery](../architecture-gallery.md) / EfficientNet-Lite0 (2019)


![EfficientNet MBConv + SE](https://upload.wikimedia.org/wikipedia/commons/e/e9/EfficientNet_block.png)
<sub>Figure credit: Tan & Le, via Wikipedia (CC BY 4.0).</sub>

## Historical Context

EfficientNet (Tan & Le) introduced a family of models that scale depth, width, and input resolution in a balanced way. EfficientNet-Lite0 adapts those ideas for mobile/CPU targets by removing swish activations and float16, yet still delivering strong accuracy per parameter.

## Core Building Blocks

- **MBConv (Mobile Inverted Bottleneck):** 1×1 pointwise expansion → depthwise 3×3 convolution → squeeze-and-excitation (SE) block → 1×1 projection.
- **Squeeze-and-Excitation:** Channel attention mechanism that globally pools each channel, passes through a small MLP, and scales the feature map.
- **Depthwise convolutions:** Provide most of the spatial processing at a fraction of the cost of full convolutions.

## Implementation Notes

- **File:** `convolutional/architectures/efficientnet.py`
- Includes helper classes for MBConv and SE directly within the architecture file.
- Uses `SiLU` activations for non-linearity and `Dropout` toward the end, mirroring the original paper’s regularization.

## Teaching Angles

- Shows how careful block design can slash parameter counts without losing accuracy.
- Highlights compound scaling: you can imagine varying block repeats/channel widths to create the rest of the EfficientNet family.
- Introduces SE attention as a practical example of channel recalibration.

## Suggested Experiments

- Compare accuracy vs VGG16 or ResNet18 for the same epoch budget to highlight efficiency.
- Inspect per-layer parameter counts using `sum(p[0].size for p in model.parameters())`.
- Demonstrate the impact of SE by temporarily disabling the squeeze-excite block (for exploratory learning).

## References

- [EfficientNet-Lite overview (Luxonis)](https://models.luxonis.com/luxonis/efficientnet-lite/fdacd30d-97f4-4c55-843f-8b7e872d8acb)
- [“EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks”](https://arxiv.org/abs/1905.11946)


[Previous (ConvNeXt-Tiny (2022))](convnext-tiny.md) | [Back to Gallery](../architecture-gallery.md) | [Next (LeNet-5 (1998))](lenet.md)

[Back to Architecture Gallery](../architecture-gallery.md)

**Navigation:**
[Previous (ConvNeXt-Tiny (2022))](convnext-tiny.md) | [Back to Gallery](../architecture-gallery.md) | [Next (LeNet-5 (1998))](lenet.md)

---
Return to [Tutorial Hub](../README.md)
