# 05 · Architecture Gallery & Dataset Notes

This page is the entry point to detailed write-ups for every convolutional network in the sandbox. Each model now has its own tutorial with block diagrams, teaching angles, experiment prompts, and references.

## Dataset: MNIST (Handwritten Digits)

- **Files:** `data/train_images.npy`, `data/train_labels.npy`, `data/test_images.npy`, `data/test_labels.npy`
- **Shape:** 28×28 grayscale digits (0–9); 60k training + 10k test samples.
- **Format:** Stored as NumPy arrays for direct loading by `common/data_utils.DataUtility`, which also handles float32 conversion, normalization, and reshaping to `(N, 1, 28, 28)`.
- **Reference:** Read `data/readme.md` for provenance and tips on creating validation splits.

MNIST is intentionally simple so you can compare architectures rapidly without long training cycles.

---

## Architecture Index

| Model | Era / Theme | Tutorial |
| --- | --- | --- |
| BaselineCNN | Custom lightweight starter | [Read notes](architectures/baseline.md) |
| LeNet-5 | 1998 classic digit recognizer | [Read notes](architectures/lenet.md) |
| AlexNet | 2012 ImageNet breakthrough | [Read notes](architectures/alexnet.md) |
| VGG-16 | 2014 deep uniform conv blocks | [Read notes](architectures/vgg16.md) |
| ResNet-18 | 2015 residual learning | [Read notes](architectures/resnet18.md) |
| EfficientNet-Lite0 | 2019 MBConv + SE efficiency | [Read notes](architectures/efficientnet-lite0.md) |
| ConvNeXt-Tiny | 2022 transformer-inspired CNN | [Read notes](architectures/convnext-tiny.md) |

---

## Comparing Architectures

Use this quick matrix to guide experiments:

| Model | Key innovations | Suggested experiment |
| --- | --- | --- |
| LeNet | Mean-pooling, shallow conv stacks | Train 1–2 epochs; observe fast saturation on MNIST. |
| AlexNet | Deep conv stack, dropout, ReLU | Time GPU vs CPU; discuss speed gains. |
| VGG-16 | Uniform 3×3 conv blocks, depth | Monitor memory usage vs accuracy payoff. |
| ResNet-18 | Skip connections for gradient flow | Disable one skip (for learning) to see training degrade. |
| EfficientNet-Lite0 | Depthwise + SE, compound scaling | Compare accuracy-per-parameter with VGG16. |
| ConvNeXt-Tiny | LayerNorm, GELU, large depthwise kernels | Inspect misclassification patterns vs ResNet. |

Document observations—accuracy curves, loss plots, and timing tables make excellent lab artifacts.

---

Pick any architecture above to dive into its dedicated tutorial and explore how the theoretical concepts translate into actual code.

### Visual Lineage

To see how ideas evolved, skim these external diagrams (each hosted on Wikipedia and licensed under Creative Commons):

| Architecture | Diagram |
| --- | --- |
| LeNet-5 | ![LeNet-5 architecture](https://upload.wikimedia.org/wikipedia/commons/6/6a/LeNet.png) |
| AlexNet | ![AlexNet schematic](https://upload.wikimedia.org/wikipedia/commons/2/29/AlexNet_layers.png) |
| VGG16 | ![VGG16 block diagram](https://upload.wikimedia.org/wikipedia/commons/4/41/VGG16.png) |
| ResNet (bottleneck) | ![ResNet residual block](https://upload.wikimedia.org/wikipedia/commons/5/5f/ResNet_module.svg) |
| EfficientNet block | ![EfficientNet MBConv + SE](https://upload.wikimedia.org/wikipedia/commons/e/e9/EfficientNet_block.png) |
| ConvNeXt block | ![ConvNeXt block](https://upload.wikimedia.org/wikipedia/commons/0/03/ConvNeXt_block.png) |

These figures are credited to their original Wikipedia contributors and are provided under the licenses noted on each image page. Use them as a visual guide while reading the corresponding architecture tutorials.

### Lab Challenges

1. **Historical timeline:** Choose three architectures from different eras (e.g., LeNet, VGG16, ConvNeXt) and create a comparison table that lists their parameter counts, training tricks, and MNIST accuracy after 2 epochs. Highlight one design idea that evolved between each pair.
2. **Residual vs non-residual:** Train ResNet-18 for one epoch, then (for learning purposes only) comment out the skip connections in a single block and retrain. Document how loss and accuracy change, and explain why the skip path matters.
