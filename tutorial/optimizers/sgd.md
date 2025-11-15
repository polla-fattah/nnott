---
title: Stochastic Gradient Descent (SGD) & Momentum
---

SGD is the workhorse of neural-network training. This note covers the math behind plain SGD, Polyak momentum, and Nesterov accelerated gradient (NAG), shows how they’re implemented in this repo, and explains how to wire them into the CLI.

## Core Update Rule

Given parameters \( w \) and gradient estimate \( g \) (averaged over a mini-batch), vanilla SGD performs:

\[
w_{t+1} = w_t - \eta \, g_t
\]

where \( \eta \) is the learning rate. This idea was formalized by Robbins & Monro (1951) in “A Stochastic Approximation Method.”

## Momentum (Polyak, 1964)

Momentum accumulates an exponential moving average of past gradients:

\[
v_{t+1} = \mu v_t + g_t, \qquad w_{t+1} = w_t - \eta v_{t+1}
\]

with momentum coefficient \( \mu \in [0,1) \). This “heavy-ball” method smooths updates and accelerates convergence along ravines.

## Nesterov Accelerated Gradient (NAG)

Nesterov (1983) proposed looking ahead before computing the gradient:

1. \( v_{t+1} = \mu v_t + g(w_t - \eta \mu v_t) \)
2. \( w_{t+1} = w_t - \eta v_{t+1} \)

The code implements this by adding an extra term when `nesterov=True`.

## Implementation in the Sandbox

- **File:** `vectorized/optim.py`, class `SGD`.
- Key features:
  - Optional `momentum` and `nesterov` flags.
  - Weight decay (`weight_decay * p`) added to gradients before applying updates.
  - Uses `xp.zeros_like` so buffers live on CPU or GPU according to the backend.

### How to Use

```python
from vectorized.optim import SGD

optim = SGD(lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
trainer = ConvTrainer(model, optim, num_classes=10, grad_clip_norm=args.grad_clip)
```

Currently, the CLI defaults to Adam. To switch to SGD globally, edit `convolutional/main.py` and replace the optimizer construction with the snippet above (or make it conditional on a new CLI flag if desired).

### Vectorized MLP Example

```python
trainer = VTrainer(model, SGD(lr=0.05, momentum=0.9), grad_clip_norm=1.0)
trainer.train(X_train, y_train, epochs=20, batch_size=128)
```

## Practical Tips

- Start with `lr=0.01` and `momentum=0.9` for CNNs on MNIST; adjust by factors of 3 if training diverges or converges too slowly.
- Combine with `--grad-clip` to keep updates stable at higher learning rates.
- Use the default learning-rate schedule (`ConvTrainer._default_multistep_schedule`) to drop the LR mid-training, mimicking classic ImageNet training recipes.

## References

- Robbins, H., & Monro, S. (1951). “A Stochastic Approximation Method.” *Annals of Mathematical Statistics*. <https://projecteuclid.org/journals/annals-of-mathematics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full>
- Polyak, B. T. (1964). “Some methods of speeding up the convergence of iteration methods.” *USSR Computational Mathematics and Mathematical Physics*. <https://doi.org/10.1016/0041-5553(64)90137-5>
- Nesterov, Y. (1983). “A method for solving the convex programming problem with convergence rate O(1/k^2).” *Soviet Mathematics Doklady*.
