# VGG-16 (2014)

**Breadcrumb:** [Home](../README.md) / [Architecture Gallery](../architecture-gallery.md) / VGG-16 (2014)


![VGG16 block diagram](https://upload.wikimedia.org/wikipedia/commons/4/41/VGG16.png)
<sub>Figure credit: Simonyan & Zisserman, via Wikipedia (CC BY 4.0).</sub>

## Historical Context

K. Simonyan and A. Zisserman’s VGG-16 demonstrated that a deep network built from simple, uniform 3×3 convolutions could reach top-tier accuracy on ImageNet. Its clean design made it a favorite starting point for transfer learning and architectural experiments.

## Architecture Structure

- **Convolutional Blocks:** Five stages, each stacking 2–3 Conv2D layers with 3×3 kernels, stride 1, and padding 1, followed by max pooling.
- **Fully Connected Head:** Two 4096-unit dense layers with ReLU + Dropout, then a final classifier layer.
- **Depth:** 13 convolutional layers + 3 dense layers = 16 learnable layers.

## Implementation Notes

- **File:** `convolutional/architectures/vgg.py`
- The sandbox version adapts pooling strides to ensure the flattened size matches the dense layers (see recent fix for stride-1 pools near the end).
- ReLU activations and dropout mirror the original training recipe.

## Teaching Angles

- Emphasizes the impact of depth and uniform design patterns.
- Highlights how quickly parameter counts explode (great example for discussing memory constraints).
- Serves as a gateway to discussions on why later models introduce residual connections or depthwise separable convs.

## Suggested Experiments

- Monitor GPU memory usage vs smaller architectures.
- Train with and without dropout to see overfitting on MNIST.
- Time a single epoch on CPU vs GPU to quantify acceleration benefits.

## References

- [“Very Deep Convolutional Networks for Large-Scale Image Recognition”](https://arxiv.org/abs/1409.1556)
- [VGG16 overview (Great Learning blog)](https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918)


[Previous (ResNet-18 (2015))](resnet18.md) | [Back to Gallery](../architecture-gallery.md) | [Next (AlexNet (2012))](alexnet.md)

[Back to Architecture Gallery](../architecture-gallery.md)

**Navigation:**
[Previous (ResNet-18 (2015))](resnet18.md) | [Back to Gallery](../architecture-gallery.md) | [Next (AlexNet (2012))](alexnet.md)
