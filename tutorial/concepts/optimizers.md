# Optimizers: SGD & Adam

Once gradients are computed, optimizers update parameters to minimize the loss. This sandbox implements Stochastic Gradient Descent (SGD) and Adam, both located in `vectorized/optim.py` and reused across scalar/vectorized/convolutional trainers.

## Stochastic Gradient Descent (SGD)

- **Update rule:** `w ← w - η * g`, where `g` is the gradient averaged over the batch.
- **Momentum:** Adds an exponential moving average of past gradients to accelerate convergence: `v ← μv + g`, `w ← w - ηv`.
- **Nesterov momentum:** Looks ahead by applying the velocity before computing the gradient, then corrects the update—often improves stability.
- **Weight decay:** Adds `λw` to the gradient, implementing L2 regularization to prevent weights from growing too large.

### Where to explore

- `class SGD(Optimizer)` in `vectorized/optim.py`.
- Check how `self._vel` caches velocities per parameter (`id(p)` as key).

## Adam

- **Idea:** Maintain first (`m`) and second (`v`) moment estimates of gradients to adapt learning rates per parameter.
- **Equations:**
  - `m ← β1 m + (1 - β1) g`
  - `v ← β2 v + (1 - β2) g^2`
  - Bias-corrected: `m_hat = m / (1 - β1^t)`, `v_hat = v / (1 - β2^t)`
  - Update: `w ← w - η * m_hat / (sqrt(v_hat) + ε)`
- **Why use it:** Faster convergence on many problems, less sensitive to initial learning-rate choice.

### Where to explore

- `class Adam(Optimizer)` in `vectorized/optim.py`.
- Note how state dictionaries `_m`, `_v`, `_t` store per-parameter statistics and how arrays are allocated on the active backend (`xp.zeros_like`).

## Learning-Rate Scheduling

- `ConvTrainer._default_multistep_schedule` (in `convolutional/trainer.py`) demonstrates a simple schedule that halves the LR mid-training and again near the end.
- You can supply your own schedule by passing `lr_schedule` to the trainer.

## Practical Tips

- Start with Adam when exploring new architectures; switch to SGD+momentum when you want tighter control or mimic classic training regimes.
- Monitor gradient norms—if they explode or vanish, tweak learning rates or consider gradient clipping.
- When resuming from checkpoints, ensure optimizer state (`m`, `v`, `velocities`) is restored; `common/model_io.py` handles this automatically.
