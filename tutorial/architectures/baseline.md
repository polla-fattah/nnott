**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](../architecture-gallery.md)

# Baseline CNN

**Breadcrumb:** [Home](../README.md) / [Architecture Gallery](../architecture-gallery.md) / Baseline CNN


## Why It Exists

The BaselineCNN is a compact starter architecture built specifically for the sandbox. It bridges the gap between fully connected MNIST models and the more advanced historical networks. Its small size keeps iteration time low while still demonstrating convolution, pooling, and dropout in practice.

## Block Diagram

1. **Conv → ReLU → Pool** repeated three times with gradually increasing channel counts.
2. **Flatten → Dense → Dropout → Dense** to map the extracted features to digit logits.
3. Optional shift augmentation (handled in the trainer) to make the tiny network more robust.

## Implementation Notes

- **File:** `convolutional/architectures/baseline.py`
- **Layers defined in:** `convolutional/modules.py`
- Uses standard 3×3 kernels, stride 1, and 2×2 max pooling to reduce spatial dimensions.
- Keeps parameter count low enough that CPU training is still quick, making it ideal for sanity checks or teaching sessions.

## Experiment Ideas

- Compare BaselineCNN accuracy vs the vectorized MLP to highlight the benefits of convolutions on image data.
- Toggle dropout to illustrate overfitting on MNIST.
- Use it as a control when demonstrating CPU vs GPU speedups (short runs minimize waiting).

## Further Reading

- Review the scalar/vectorized tutorials first so you can inspect how convolutions extend the same ideas to 2D data.


[Previous (AlexNet (2012))](alexnet.md) | [Back to Gallery](../architecture-gallery.md) | [Next (ConvNeXt-Tiny (2022))](convnext-tiny.md)

[Back to Architecture Gallery](../architecture-gallery.md)

**Navigation:**
[Previous (AlexNet (2012))](alexnet.md) | [Back to Gallery](../architecture-gallery.md) | [Next (ConvNeXt-Tiny (2022))](convnext-tiny.md)

---

MIT License | [About](../about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
