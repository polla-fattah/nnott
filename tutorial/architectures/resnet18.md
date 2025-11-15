---
title: ResNet-18 (2015)

---


![ResNet residual block](https://upload.wikimedia.org/wikipedia/commons/b/b3/ResNet_block.svg)
Figure credit: He et al., via Wikipedia (CC BY 4.0).

## Historical Context

Residual Networks (ResNets) by He et al. solved the “degradation” problem: deeper models were underperforming shallower ones because gradients struggled to flow through many layers. ResNets introduced identity skip connections that let layers learn residual functions, enabling networks with 100+ layers.

## Architecture Structure

- **Stem:** 7×7 Conv + MaxPool to quickly reduce spatial size (adapted for MNIST in this repo).
- **Residual Stages:** Four stages, each containing multiple residual blocks. Every block has two 3×3 convolutions plus a skip path (identity or 1×1 projection when dimensions change).
- **Head:** Global average pooling → fully connected classifier.

## Implementation Notes

- **File:** `convolutional/architectures/resnet.py`
- Residual blocks live alongside helper modules in the same file; they rely on `BatchNorm2D`, `ReLU`, and `Conv2D` from `convolutional/modules.py`.
- Downsampling blocks use stride 2 plus a 1×1 conv on the skip path to keep tensor shapes compatible.

## Teaching Angles

- Demonstrates how skip connections preserve gradients and encourage feature reuse.
- Highlights the difference between “basic blocks” (used here) and “bottleneck blocks” in deeper ResNets.
- Provides a concrete example of identity vs projection shortcuts.

## Suggested Experiments

- Comment out the skip addition (for a single block) to see training degrade—great for illustrating why residuals matter.
- Compare training curves to VGG16; ResNet should converge faster/stabler despite similar depth.

## References

- [“Deep Residual Learning for Image Recognition”](https://arxiv.org/abs/1512.03385)
- [Analytics Vidhya summary of skip connections](https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/)
