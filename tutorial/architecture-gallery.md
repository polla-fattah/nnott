# 05 · Architecture Gallery & Dataset Notes

This section spotlights every convolutional network implemented in the sandbox, explains the design ideas behind each one, and summarizes the dataset used for training.

## Dataset: MNIST (Handwritten Digits)

- **Files:** `data/train_images.npy`, `data/train_labels.npy`, `data/test_images.npy`, `data/test_labels.npy`
- **Shape:** 28×28 grayscale digits (0–9); 60k training + 10k test samples.
- **Format:** Stored as NumPy arrays for direct loading by `common/data_utils.DataUtility`, which also handles float32 conversion, normalization, and reshaping to `(N, 1, 28, 28)`.
- **Reference:** `data/readme.md` explains provenance and how to split out validation sets if needed.

Even though MNIST is “easy” by modern standards, it’s perfect for demonstrating architectural differences without long training times.

---

## Classic-to-Modern CNN Lineup

### BaselineCNN (Custom Starter)

- **File:** `convolutional/architectures/baseline.py`
- **Idea:** Minimal three-block CNN with Conv → ReLU → Pool stacks followed by a small classifier head. Serves as a bridge between MLPs and deeper networks.

### LeNet-5 (1998)

- **File:** `convolutional/architectures/lenet.py`
- **Highlights:** Two conv layers with mean pooling, followed by fully connected layers. Demonstrates how early CNNs leveraged local receptive fields and downsampling to outperform dense networks on digit recognition.

### AlexNet (2012)

- **File:** `convolutional/architectures/alexnet.py`
- **Highlights:** Five conv layers, heavy use of ReLU, max-pooling, and dropout. Pioneered large-scale CNNs on ImageNet and showcased the benefit of training on GPUs.
- **Teaching angle:** Compare its depth and parameter count to LeNet to see how scaling up capacity boosts accuracy on more complex datasets.

### VGG16 (2014)

- **File:** `convolutional/architectures/vgg.py`
- **Highlights:** Deep stack of 3×3 convolutions arranged in uniform blocks. Simple design but very deep (13 conv + 3 dense layers). Emphasizes the power of depth and consistent filter sizes.
- **Code note:** The tutorial patches ensure fully connected layers receive the correct flattened size even with stride-1 pooling near the end.

### ResNet18 (2015)

- **File:** `convolutional/architectures/resnet.py`
- **Highlights:** Residual blocks with identity skip connections allow gradients to flow through very deep nets. Each block contains two 3×3 conv layers plus the direct addition of the input.
- **Teaching angle:** Observe how the skip connections stabilize training compared to stacking more layers without them.

### EfficientNet-Lite0 (2019)

- **File:** `convolutional/architectures/efficientnet.py`
- **Highlights:** Mobile Inverted Bottleneck (MBConv) blocks with depthwise convolutions, squeeze-and-excitation (SE) attention, and a compound-scaling philosophy that balances width/depth/resolution.
- **Why it matters:** Shows how modern models achieve high accuracy with far fewer parameters by carefully designing each block.

### ConvNeXt-Tiny (2022)

- **File:** `convolutional/architectures/convnext.py`
- **Highlights:** Modernized CNN inspired by Vision Transformers. Uses patchify stems, large-kernel depthwise conv, LayerNorm instead of BatchNorm, GELU activations, and inverted bottlenecks.
- **Teaching angle:** Demonstrates how CNNs evolved by borrowing design cues from transformer models to stay competitive.

---

## Comparing Architectures

| Model | Key Innovations | Suggested experiment |
| --- | --- | --- |
| LeNet | First successful CNN for digits; mean-pooling and small filters. | Train for 1–2 epochs to see how quickly it saturates MNIST. |
| AlexNet | Deep conv stack, ReLU, dropout, heavy augmentation. | Measure training time vs LeNet; note improved accuracy. |
| VGG16 | Uniform 3×3 conv blocks, very deep network. | Observe memory usage and training time compared to AlexNet. |
| ResNet18 | Residual connections for deep training. | Comment out skips to see optimization degrade (for experimentation only). |
| EfficientNet-Lite0 | Depthwise + SE blocks, compound scaling. | Compare accuracy-per-parameter against VGG16. |
| ConvNeXt-Tiny | Large depthwise kernels, LayerNorm, GELU. | Test how quickly it converges and inspect misclassification plots. |

Document your findings—accuracy curves or inference timing tables provide excellent discussion material in class or lab reports.

---

By exploring each architecture in this gallery, you not only learn the historical milestones of CNN design but also see exactly how those ideas translate into code. Use the linked files to dive deeper whenever you encounter a new block or layer in the tutorial.
