**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](../architecture-gallery.md)

# Regularization & Augmentation

**Breadcrumb:** [Home](../README.md) / [Core Concepts](../core-concepts.md) / Regularization & Augmentation


Regularization techniques keep models from overfitting and improve generalization. This sandbox implements several classic methods that you can toggle or modify.

## Dropout

- **File:** `convolutional/modules.py` (`class Dropout`)
- **Mechanism:** During training, randomly zeroes activations with probability `p` and scales by `1 / (1 - p)` to keep expectations consistent.
- **Usage:** Applied in BaselineCNN, AlexNet, VGG16, and EfficientNet heads.
- **Experiment:** Train VGG16 with and without dropout to see how quickly it overfits MNIST.

## Weight Decay (L2 Regularization)

- Implemented inside optimizers by adding `weight_decay * param` to gradients before updates.
- Shrinks weights toward zero and discourages overly complex solutions.

## Data Augmentation

- **Function:** `ConvTrainer._augment_batch` (in `convolutional/trainer.py`)
- **Current augmentation:** Random shifts up to ±2 pixels using `xp.roll`.
- **Teaching point:** Even simple translations can make MNIST models more robust; try increasing `max_shift` or adding random noise.

## Early Stopping (DIY)

- Not built-in, but you can monitor validation accuracy and halt training when it plateaus. Combine with checkpoints (`--save`) to resume later if needed.

## Batch Size Effects

- Smaller batches introduce noise in gradient estimates—this acts as a mild regularizer.
- Experiment by halving/doubling `--batch-size` and observing test accuracy variance.

## Checklist for Labs

1. Run baseline without augmentation to measure overfitting.
2. Enable dropout and augmentation; quantify the improvement.
3. Plot weight histograms before and after applying weight decay to see shrinkage.

[Previous (Optimizer Hub)](optimizers.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Activation Functions)](activations.md)

**Navigation:**
[Back to Core Concepts](../core-concepts.md)

---

MIT License | [About](../about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
