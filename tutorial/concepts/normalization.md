---
title: Normalization
---

# Normalization Layers



Normalization stabilizes activations, speeds up training, and often improves generalization. This project provides both BatchNorm and LayerNorm variants tailored for CNNs.

## Batch Normalization (BatchNorm2D)

- **File:** `convolutional/modules.py` (`class BatchNorm2D`)
- **Behavior:** For each feature channel, computes batch mean/variance, normalizes inputs, then applies learnable scale (`gamma`) and shift (`beta`).
- **Running stats:** Maintains moving averages of mean/variance for inference mode.
- **Where used:** BaselineCNN, AlexNet, VGG16, ResNet18, EfficientNet blocks.
- **Benefits:** Allows higher learning rates, mitigates internal covariate shift, provides mild regularization.

## Layer Normalization (LayerNorm2D)

- **File:** `convolutional/modules.py` (`class LayerNorm2D`)
- **Behavior:** Normalizes across feature channels within each sample, independent of batch size.
- **Where used:** ConvNeXt blocks, where transformer-style norms replace BatchNorm.
- **Benefits:** Stable even with small batch sizes and fits architectures inspired by transformers.

## BatchNorm for Fully Connected Layers

- **File:** `vectorized/modules.py` (`class BatchNorm1D`)
- **Behavior:** Mirrors the 2D version but operates on `[batch, features]` tensors, making it ideal for the fully connected MLP.
- **How to enable:** Run `python vectorized/main.py --batchnorm ...` (or add `--batchnorm` to the quick-start script). The flag inserts `BatchNorm1D` between the Linear layer and its activation for every hidden block. Adjust `--bn-momentum` to control how quickly running statistics adapt.
- **Experiment:** Train once with `--batchnorm` and once without, keeping `--dropout` constant. Compare convergence speed and test accuracy, especially with larger hidden sizes.

## Implementation Details

- Both layers store intermediate state (`_cache`) for backward passes.
- During evaluation, they skip batch statistics and rely on stored parameters to keep inference deterministic.
- The trainer calls `model.train()` / `model.eval()` to switch behavior appropriately.

## Experiments

1. Disable BatchNorm in VGG16 (temporarily) to observe slower convergence or instability.
2. Swap BatchNorm for LayerNorm in a block to see how it affects performance on small batches.
3. Log running means/vars for BatchNorm to understand how they converge over time.

[Previous (Loss & Softmax: Cross-Entropy from Logits)](loss-and-softmax.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Optimizer Hub)](optimizers.md)

[Back to Core Concepts](../core-concepts.md)

---

