---
title: Gradient Clipping
---

Gradient clipping limits the magnitude of backpropagated gradients to prevent exploding updates. Originally popularized in recurrent networks (Pascanu et al., 2013), it’s also useful for CNNs and MLPs when experimenting with aggressive learning rates or new architectures.

## Global Norm Clipping

The sandbox implements *global-norm* clipping:

1. Compute \( \|g\| = \sqrt{\sum_i \|g_i\|^2} \) across all parameter gradients.
2. If \( \|g\| > c \) (threshold), scale every gradient tensor by \( c / (\|g\| + \varepsilon) \).
3. Proceed with the optimizer step.

This preserves gradient direction while shrinking its magnitude, keeping updates stable.

## Implementation Details

- **Vectorized trainer:** `_clip_gradients` method in `vectorized/trainer.py`.
- **Convolutional trainer:** `_clip_gradients` method in `convolutional/trainer.py`.
- Both take `grad_clip_norm` (float) at construction. When `None` or ≤0, clipping is disabled.

## CLI Usage

`convolutional/main.py` exposes the feature via `--grad-clip`:

```bash
python convolutional/main.py resnet18 --epochs 5 --batch-size 128 --gpu \
    --grad-clip 5.0
```

- Set `--grad-clip` to the desired norm (e.g., `1.0`, `5.0`). Leave unset to disable.
- The value is passed to `ConvTrainer(..., grad_clip_norm=args.grad_clip)`.

For vectorized runs, instantiate the trainer directly:

```python
trainer = VTrainer(model, Adam(lr=1e-3), grad_clip_norm=1.0)
```

## When to Enable

- Loss spikes or diverges after a few batches.
- Training with high learning rates or on architectures newly added to the sandbox.
- Using optimizers without adaptive behavior (plain SGD) on deeper networks.

## Tips

- Start with a moderate threshold (e.g., 5.0 for CNNs). Lower the value if you still observe instability.
- Combine with Lookahead to further stabilize training.
- Monitor gradient norms before/after clipping (add temporary prints) to understand how often clipping triggers.

## Reference

- Pascanu, R., Mikolov, T., & Bengio, Y. (2013). “On the difficulty of training recurrent neural networks.” arXiv:1211.5063. <https://arxiv.org/abs/1211.5063>

[Previous (Adam (Adaptive Moment Estimation))](adam.md) | [Back to Optimizers Hub](../concepts/optimizers.md) | [Next (Lookahead Optimizer)](lookahead.md)

[Back to Optimizers Hub](../concepts/optimizers.md)
