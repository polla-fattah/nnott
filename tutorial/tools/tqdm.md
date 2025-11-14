---
title: Tqdm
---

# tqdm Quick Notes

## Why You Need tqdm

Training large datasets is more manageable when you can see progress. `tqdm` wraps iterables with a live progress bar so you can monitor batch throughput, ETA, and overall health of a run.

## Where It Appears in the Project

- **Trainers:** Both `vectorized/trainer.py` and `convolutional/trainer.py` wrap batch loops inside `tqdm(range(...))`, yielding a progress bar labeled by epoch.
- **Scripts:** Any custom experiment you write can adopt the same pattern—`for batch in tqdm(loader, desc="Epoch 1")`.

## Handy Patterns

```python
from tqdm import tqdm
for idx in tqdm(range(0, len(data), batch_size), desc="Epoch 3", unit="batch"):
    ...
```

- Use `unit` (`"batch"`, `"step"`, etc.) to keep terminology consistent with training logs.
- Combine with `leave=False` if you want the bar to disappear after completion in notebooks.

Official docs: <https://tqdm.github.io/>
