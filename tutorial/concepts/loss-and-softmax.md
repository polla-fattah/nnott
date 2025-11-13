**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](../architecture-gallery.md)

# Loss & Softmax: Cross-Entropy from Logits

**Breadcrumb:** [Home](../README.md) / [Core Concepts](../core-concepts.md) / Loss & Softmax: Cross-Entropy from Logits


Classification networks in this repo use the cross-entropy loss applied directly to raw logits. This ensures numerical stability and pairs cleanly with the softmax function.

## Where to Look

- **Primary file:** `common/cross_entropy.py`
- **Helper:** `common/softmax.py`
- **Trainer usage:** `common/cross_entropy.CrossEntropyLoss` is instantiated inside `scalar/trainer.py`, `vectorized/trainer.py`, and `convolutional/trainer.py`.

## Concept Recap

- **Softmax:** `softmax(z)_i = exp(z_i) / Σ_j exp(z_j)` converts logits `z` into probabilities.
- **Cross-Entropy:** For target class `y`, `CE = -log(softmax(z)_y)`. It penalizes low probability assigned to the correct class.

## Numerically Stable Implementation

Directly computing softmax can overflow when logits are large. The code uses the **log-sum-exp trick**:

1. Subtract the max logit from all logits to keep numbers small.
2. Compute `lse = log(sum(exp(z - z_max)))`.
3. Loss becomes `-z_y + z_max + lse`.

### Targets Handling

- Accepts either integer class indices or one-hot vectors.
- Converts everything to the active backend (`xp`) so CPU and GPU follow the same path.

## Gradients

- The gradient of cross-entropy with softmax simplifies to `softmax(z) - one_hot(target)`.
- `cross_entropy_grad_logits` implements this efficiently, which is why trainers call `criterion.backward(logits, targets)` immediately after the forward pass.

## Exercises

1. Print both logits and loss for a small batch to see how cross-entropy behaves as predictions improve.
2. Manually compute the loss for a few samples and compare with the function output to build intuition.
3. Experiment with label smoothing (modify targets slightly) to understand its regularization effect.

[Previous (Fundamentals: Neurons, Layers, Networks)](fundamentals.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Normalization Layers)](normalization.md)

**Navigation:**
[Back to Core Concepts](../core-concepts.md)

---

MIT License | [About](../about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
