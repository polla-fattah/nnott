[MyHome](https://polla.dev) | [Tutorial Hub](README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](architecture-gallery.md)

# Experiment Log Template

**Breadcrumb:** [Home](README.md) / Experiment Log Template


Use this template to record every meaningful run. Consistent notes help you reproduce results, compare hyperparameters, and communicate findings to classmates or your instructor.

```markdown
# Experiment Title

- **Date:** YYYY-MM-DD
- **Author:** Your name
- **Goal:** Describe the purpose (e.g., “Compare SGD vs Adam on vectorized MLP”, “Test gradient clipping on ResNet18”).

## Setup

- **Dataset:** MNIST / Fashion-MNIST / CIFAR-10 (include file names if custom)
- **Architecture:** e.g., scalar MLP (128,64), vectorized MLP (256,128), ResNet18, etc.
- **Script/Command:** `python convolutional/main.py ...` or quick-start script invocation
- **Hardware:** CPU model, GPU model (if any), environment (Conda env name, Python version)

## Hyperparameters

| Parameter | Value |
| --- | --- |
| Epochs |  |
| Batch size |  |
| Optimizer |  |
| Learning rate |  |
| Weight decay |  |
| Gradient clipping |  |
| Lookahead |  |
| Augmentation |  |
| Other |  |

## Results

- **Training loss (per epoch):** `[ ... ]`
- **Validation/Test accuracy:** `... %`
- **Runtime:** `... min`
- **Notes on logs:** (e.g., loss spikes, warnings, CUDA messages)

## Observations

- What went well?
- What didn’t?
- Any anomalies or bugs?
- Next steps (follow-up experiments, parameters to try next)

## Attachments (optional)

- Paste charts (loss curves, misclassification grids) or link to screenshots.
- If using Jupyter, note the notebook name and cell numbers.
```

Fill in one template per experiment and store them under `experiments/` or a personal `notes/` folder. Reviewing these logs before exams or project milestones will save you hours of guesswork.

**Navigation:**
[Back to Running Experiments](../running-experiments.md)

---

MIT License | [About](about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
