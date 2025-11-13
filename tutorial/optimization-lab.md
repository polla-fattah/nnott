**Links:** [MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](..) | [Architectures](architecture-gallery.md)


# Optimization Lab (Jupyter Walkthrough)

**Breadcrumb:** [Home](README.md) / Optimization Lab (Jupyter Walkthrough)


This lab is designed to be run inside a Jupyter notebook so you can tweak hyperparameters interactively. The notebook uses the vectorized MLP (NumPy-based) so the code stays concise while still reflecting all the optimizer features available in the main project.

## Prerequisites

1. Install Jupyter (e.g., `pip install notebook`).
2. In the repository root, launch `jupyter notebook`.
3. Create a new notebook and copy increasingly complex cells from the sections below.

> Tip: Keep the repo’s virtual environment activated before launching Jupyter so it can import project modules.

---

## 1. Setup & Data Loading

```python
import os, sys, numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # adjust if notebook lives elsewhere

from common.data_utils import DataUtility
from vectorized.modules import Sequential, Linear, ReLU
from vectorized.optim import Adam, SGD, Lookahead
from vectorized.trainer import VTrainer

data = DataUtility("data")
X_train, y_train, X_test, y_test = data.load_data()
X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
X_test = X_test.reshape(len(X_test), -1).astype(np.float32)
```

- **Why:** Loads MNIST, standardizes it, and flattens the images for the MLP.

---

## 2. Baseline Model

```python
def build_mlp(hidden_sizes=(256, 128)):
    layers = []
    in_dim = 28 * 28
    for h in hidden_sizes:
        layers.append(Linear(in_dim, h, activation_hint="relu"))
        layers.append(ReLU())
        in_dim = h
    layers.append(Linear(in_dim, 10, activation_hint=None))
    return Sequential(*layers)

baseline_model = build_mlp()
baseline_optim = Adam(lr=1e-3, weight_decay=1e-4)
trainer = VTrainer(baseline_model, baseline_optim, num_classes=10, grad_clip_norm=None)
trainer.train(X_train, y_train, epochs=3, batch_size=64, verbose=True)
acc = trainer.evaluate(X_test, y_test)
print(f"Baseline accuracy: {acc*100:.2f}%")
```

- **Experiment:** Vary `hidden_sizes`, `epochs`, or `batch_size` to see how capacity and training time change.

---

## 3. Learning Rate Sweep

Use a small loop to try several learning rates quickly:

```python
def train_with_lr(lr):
    model = build_mlp()
    optim = Adam(lr=lr, weight_decay=1e-4)
    trainer = VTrainer(model, optim, num_classes=10)
    trainer.train(X_train, y_train, epochs=2, batch_size=64, verbose=False)
    return trainer.loss_history, trainer.evaluate(X_test, y_test)

for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
    loss_history, acc = train_with_lr(lr)
    print(f"lr={lr:.4g} -> final loss={loss_history[-1]:.4f}, acc={acc*100:.2f}%")
```

- **Observation:** A learning rate that’s too high may explode; too low converges slowly. Plot `loss_history` to visualize this.

---

## 4. Gradient Clipping Demo

```python
model = build_mlp()
optim = Adam(lr=1e-3)
trainer = VTrainer(model, optim, num_classes=10, grad_clip_norm=1.0)
trainer.train(X_train, y_train, epochs=3, batch_size=64, verbose=True)
```

- **Challenge:** Try `grad_clip_norm=None` vs `grad_clip_norm=1.0` and note whether loss spikes disappear.

---

## 5. Lookahead Wrapper

```python
base_opt = Adam(lr=1e-3, weight_decay=1e-4)
lookahead_opt = Lookahead(base_opt, k=5, alpha=0.5)
model = build_mlp()
trainer = VTrainer(model, lookahead_opt, num_classes=10, grad_clip_norm=5.0)
trainer.train(X_train, y_train, epochs=3, batch_size=64, verbose=True)
trainer.evaluate(X_test, y_test)
```

- **Activity:** Adjust `k` and `alpha`, then compare loss curves to plain Adam. Does Lookahead smooth convergence?

---

## 6. Visualization (Optional)

```python
import matplotlib.pyplot as plt

plt.plot(range(1, len(trainer.loss_history)+1), trainer.loss_history, marker="o")
plt.title("Loss curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

imgs, preds, trues, total = trainer.misclassification_data(X_test, y_test, max_images=25)
```

- **Warning:** Collecting misclassifications requires another forward pass; only do this when necessary.

---

## Suggested Notebook Challenges

1. **Optimizer grid:** For learning rates `[1e-4, 1e-3, 5e-3]` and optimizers `{SGD+momentum, Adam, Lookahead(Adam)}`, fill in an accuracy table and highlight the best combo.
2. **Clip vs No Clip:** Run the same model with `grad_clip_norm=None` and `grad_clip_norm=1`. Plot both loss curves on the same axes and describe the difference.
3. **Lookahead intuition:** Fix the base optimizer and vary `k` (e.g., 3, 5, 10) to see how frequently syncing slow weights affects convergence speed.
4. **Batch size stress-test:** Keep epochs constant but sweep batch sizes `[32, 64, 128, 256]`. Track runtime and accuracy to discuss the trade-offs.

Save the notebook as `notebooks/optimization_lab.ipynb` (or similar) so future students can open it, rerun the cells, and add their own observations.

**Navigation:**
[Back to Running Experiments](../running-experiments.md)

---
Return to [Tutorial Hub](README.md)
