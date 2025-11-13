# Adam (Adaptive Moment Estimation)

Adam combines momentum and adaptive learning rates by keeping first and second moment estimates of gradients. Introduced by Kingma & Ba (2014), it became the default optimizer for many deep-learning tasks thanks to its robustness and minimal tuning.

## Math Recap

For gradient \( g_t \) at step \( t \):

1. First moment (mean):
   \[
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
   \]
2. Second moment (uncentered variance):
   \[
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
   \]
3. Bias correction:
   \[
   \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
   \]
4. Parameter update:
   \[
   w_{t+1} = w_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
   \]

Default hyperparameters: \( \beta_1 = 0.9 \), \( \beta_2 = 0.999 \), \( \epsilon = 10^{-8} \).

## Implementation in the Sandbox

- **File:** `vectorized/optim.py`, class `Adam`.
- Stores per-parameter dictionaries `_m`, `_v`, and `_t`.
- Uses the backend proxy (`xp`) so it works on NumPy or CuPy tensors.
- Applies weight decay (if specified) before moment calculations.

### Usage Examples

**Convolutional CLI (default)**:
```python
optim = Adam(lr=5e-4, weight_decay=1e-4)
trainer = ConvTrainer(model, optim, num_classes=10, grad_clip_norm=args.grad_clip)
```

**Vectorized MLP**:
```python
trainer = VTrainer(model, Adam(lr=1e-3), grad_clip_norm=1.0)
trainer.train(X_train, y_train, epochs=10, batch_size=128)
```

**With Lookahead wrapper**:
```python
from vectorized.optim import Adam, Lookahead
optim = Lookahead(Adam(lr=5e-4, weight_decay=1e-4), k=5, alpha=0.5)
```

## Practical Guidance

- Adam is forgiving on learning rate; start with `1e-3` for MLPs and `5e-4` for CNNs, then fine-tune.
- Combine with `--grad-clip` if you observe spikes in loss (e.g., `--grad-clip 5`).
- When resuming from checkpoints, optimizer state (`m`, `v`, `t`) is restored via `common/model_io.py`, so you can continue training seamlessly.

## Reference

- Kingma, D. P., & Ba, J. (2014). “Adam: A Method for Stochastic Optimization.” arXiv:1412.6980. <https://arxiv.org/abs/1412.6980>
