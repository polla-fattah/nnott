---
title: Core Concepts
---

[MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# 04 · Core Concepts Hub



This hub links to detailed tutorials covering every foundational concept used throughout the sandbox. Read them sequentially for a full theory-to-code walkthrough or jump directly to the topic you need.

## Concept Index

| Topic | What you’ll learn | Link |
| --- | --- | --- |
| Fundamentals | Neurons, layers, and network composition in scalar vs vectorized code | [Study fundamentals](concepts/fundamentals.md) |
| Activation functions | ReLU, LeakyReLU, SiLU, GELU, and softmax implementations | [Explore activations](concepts/activations.md) |
| Loss & softmax | Numerically stable cross-entropy from logits and its gradients | [Understand loss](concepts/loss-and-softmax.md) |
| Optimizers | Hub for SGD, Adam, Lookahead, gradient clipping, and schedules | [Review optimizers](concepts/optimizers.md) |
| Regularization | Dropout, weight decay, data augmentation, and batch-size effects | [Apply regularization](concepts/regularization.md) |
| Normalization | BatchNorm vs LayerNorm, when to use each, and why | [Compare normalization](concepts/normalization.md) |
| Convolution mechanics | im2col/col2im, depthwise conv, squeeze-excite, global pooling | [Decode convolution](concepts/convolution.md) |
| Backend utilities | NumPy ↔ CuPy switching, device helpers, diagnostics | [Manage backends](concepts/backend.md) |

---

**How to use this hub**

1. Pick the concept you’re studying in lecture.
2. Read the linked tutorial to see the theory mapped onto exact files/functions.
3. Modify the referenced code and run small experiments to solidify your understanding.

With these guides, you can build a solid foundation before diving into the larger architectures and experiments.

### Lab Challenges

1. **Gradient trace:** Follow the scalar implementation through a single training example (set batch size to 1). Write down the value of each neuron’s activation and gradient, then verify they match the equations in the Fundamentals and Loss tutorials.
2. **Optimizer swap experiment:** Using the Optimizer hub, implement both Adam and SGD+momentum for the same network, run each for 3 epochs, and plot/compare their loss histories. Summarize the behavioral differences you observed.

---

