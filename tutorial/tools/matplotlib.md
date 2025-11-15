---
title: Matplotlib
---


## Why You Need Matplotlib

Training is more insightful when you can visualize learning curves and misclassified samples. Matplotlib powers all figures produced by the trainers.

## Where It Appears in the Project

- **Loss curves:** `scalar/main.py`, `vectorized/main.py`, and `convolutional/main.py` each plot `trainer.loss_history` after training.
- **Misclassification grids:** The main scripts gather misclassification data from the trainers and render grids directly.
- **Sample visualization:** `scalar/main.py` calls `DataUtility.sample_images(...)` and handles plotting itself.

## Tips

- Use a backend that supports interactive windows (or run inside environments like Jupyter) if you want to inspect plots during training.
- If you’re on a headless server, set `matplotlib.use("Agg")` or save figures to disk instead of showing them.

Official docs: <https://matplotlib.org/stable/contents.html>

