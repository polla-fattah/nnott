[MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](../architecture-gallery.md)

# Lookahead Optimizer

**Breadcrumb:** [Home](../README.md) / [Optimizers Hub](../concepts/optimizers.md) / Lookahead Optimizer


Lookahead (Zhang et al., 2019) is a meta-optimizer that wraps any base optimizer (SGD, Adam, etc.) with a slow/fast weight coupling. It improves stability and often yields better generalization with little overhead.

## Idea in a Nutshell

1. Maintain two sets of weights: fast weights (updated every step by the base optimizer) and slow weights (periodically synchronized).
2. After every \( k \) updates, interpolate:
   \[
   w_{\text{slow}} = w_{\text{slow}} + \alpha (w_{\text{fast}} - w_{\text{slow}})
   \]
   Then copy `w_slow` back into the fast weights.
3. Fast weights continue training from the interpolated position.

Parameters:
- \( k \): number of inner steps before synchronization (default 5).
- \( \alpha \): interpolation factor (default 0.5).

## Implementation

- **File:** `vectorized/optim.py`, class `Lookahead`.
- Wraps any optimizer inheriting from `Optimizer`.
- Stores a per-parameter “slow” copy in `_slow`.
- Calls `base_optimizer.step(...)` every iteration; applies synchronization when `step % k == 0`.

### Example Construction

```python
from vectorized.optim import Adam, Lookahead

base = Adam(lr=5e-4, weight_decay=1e-4)
optim = Lookahead(base, k=5, alpha=0.5)
trainer = ConvTrainer(model, optim, num_classes=10, grad_clip_norm=args.grad_clip)
```

## CLI Support

`convolutional/main.py` exposes Lookahead through flags:

```bash
python convolutional/main.py resnet18 --epochs 5 --batch-size 128 --gpu \
    --lookahead --lookahead-k 5 --lookahead-alpha 0.5
```

- `--lookahead`: enables the wrapper.
- `--lookahead-k`: number of inner updates.
- `--lookahead-alpha`: interpolation factor.

Vectorized MLPs currently require a code edit (wrap the optimizer manually as in the snippet above).

## When to Use

- Training oscillates or diverges with the base optimizer alone.
- You want Adam’s rapid progress but closer-to-SGD generalization.
- You’re experimenting with higher learning rates and need extra stability.

## Reference

- Zhang, M., Lucas, J., Hinton, G., & Ba, J. (2019). “Lookahead Optimizer: k steps forward, 1 step back.” arXiv:1907.08610. <https://arxiv.org/abs/1907.08610>

[Previous (Gradient Clipping)](gradient-clipping.md) | [Back to Optimizers Hub](../concepts/optimizers.md) | [Next (Stochastic Gradient Descent (SGD) & Momentum)](sgd.md)

**Navigation:**
[Back to Optimizers Hub](../concepts/optimizers.md)

---

MIT License | [About](../about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
