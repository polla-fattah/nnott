---
title: Fundamentals
---

# Fundamentals: Neurons, Layers, Networks



Understanding neural networks starts with the simplest building block: a neuron that performs a weighted sum followed by a non-linearity. This section connects the theory to the code in the scalar and vectorized implementations.

## Neuron Mechanics

- **File to inspect:** `scalar/neuron.py`
- Each neuron stores its weights, bias, and intermediate values (`z`, `activation`).
- **Forward pass:** `z = w · x + b`, followed by an activation function (`sigmoid`, `tanh`, or ReLU in variants).
- **Backward pass:** Uses the chain rule to compute `∂L/∂w`, `∂L/∂b`, and `∂L/∂x`.

### Practice Tips

1. Run `python scalar/main.py` and print neuron activations to see how values change.
2. Modify the activation function to confirm how gradients are affected.

## Layers and Modules

- **Scalar version:** `scalar/layer.py` groups neurons and handles per-layer forward/backward logic.
- **Vectorized version:** `vectorized/modules.py` replaces explicit loops with matrix operations, but retains the same conceptual flow.
- Each layer maintains its own parameters and gradients, exposing them via `parameters()` so optimizers can update them.

## Assembling Networks

- **Files:** `scalar/network.py`, `vectorized/modules.py` (Sequential class), `convolutional/modules.py`.
- Networks are simply chains of modules. During a forward pass, tensors flow through each module in order; during backward, gradients traverse in reverse.
- This mirrors the mathematical composition of functions: `f(x) = f_n(...f_2(f_1(x)))`.

## Why It Matters

- A solid grasp of neurons/layers demystifies backpropagation. Once you understand the scalar implementation, every other architecture becomes a composition of the same primitives.
- Debugging tip: if a deep model misbehaves, temporarily swap in a smaller Sequential stack to isolate which module behaves unexpectedly.

[Previous (Convolution Mechanics & Advanced Blocks)](convolution.md) | [Back to Core Concepts](../core-concepts.md) | [Next (Loss & Softmax: Cross-Entropy from Logits)](loss-and-softmax.md)

[Back to Core Concepts](../core-concepts.md)

---

