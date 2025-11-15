---
title: ConvNeXt-Tiny (2022)

---

![ConvNeXt block](https://upload.wikimedia.org/wikipedia/commons/0/03/ConvNeXt_block.png)
Figure credit: Liu et al., via Wikipedia (CC BY 4.0).

## Historical Context

Facebook AI’s ConvNeXt revisits CNN design in the age of Vision Transformers. By adopting transformer-inspired choices—large patchification, LayerNorm, GELU, depthwise separable convs—the authors produced pure CNNs that rival transformer accuracy on ImageNet.

## Block Structure

1. **Patchify Stem:** A large-stride convolution that acts like ViT’s patch embedding.
2. **ConvNeXt Blocks:** Each block applies a 7×7 depthwise convolution, LayerNorm, an MLP-like pointwise expansion (ratio 4), GELU activation, and a residual connection.
3. **Downsampling between stages** via stride-2 convolutions, similar to hierarchical transformers.
4. **Global average pooling + linear classifier** at the end.

## Implementation Notes

- **File:** `convolutional/architectures/convnext.py`
- Uses `LayerNorm2D`, `GELU`, `DepthwiseConv2D`, and `GlobalAvgPool2D` from `convolutional/modules.py`.
- Keeps the Tiny variant’s stage depths (3-3-9-3 blocks) adapted to MNIST input size.

## Teaching Angles

- Illustrates how modern CNNs borrow normalization and activation choices from transformers.
- Highlights the impact of large kernel depthwise convolutions for greater receptive fields.
- Shows the evolution from BatchNorm/Residual combos to LayerNorm/inverted bottlenecks.

## Suggested Experiments

- Compare convergence speed and final accuracy vs ResNet18 to see how the newer design fares on MNIST.
- Visualize misclassifications: does ConvNeXt make different mistakes compared to classic nets?
- Try running just one stage (trim the architecture) to observe how depth affects performance.

## References

- [ConvNeXt summary (Emergent Mind)](https://www.emergentmind.com/topics/convnext-backbone)
- [“A ConvNet for the 2020s” paper](https://arxiv.org/abs/2201.03545)


