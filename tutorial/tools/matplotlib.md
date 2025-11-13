# Matplotlib Quick Notes

## Why You Need Matplotlib

Training is more insightful when you can visualize learning curves and misclassified samples. Matplotlib powers all figures produced by the trainers.

## Where It Appears in the Project

- **Loss curves:** `ConvTrainer.plot_loss()` and the vectorized trainer both call Matplotlib to chart epoch losses.
- **Misclassification grids:** `ConvTrainer.show_misclassifications()` renders a tiled gallery of incorrectly predicted digits, labeling predictions vs ground truth.
- **Sample visualization:** `common/data_utils.DataUtility.show_samples()` uses Matplotlib to preview the dataset before training.

## Tips

- Use a backend that supports interactive windows (or run inside environments like Jupyter) if you want to inspect plots during training.
- If you’re on a headless server, set `matplotlib.use("Agg")` or save figures to disk instead of showing them.

Official docs: <https://matplotlib.org/stable/contents.html>
