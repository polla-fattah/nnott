# NumPy Quick Notes

**Breadcrumb:** [Home](../README.md) / [Project Tour](../project-tour.md) / NumPy Quick Notes


## Why You Need NumPy

NumPy is the numerical backbone of this sandbox. Even when you run on GPU, the API surface mirrors NumPy semantics, so understanding it is essential.

## Where It Appears in the Project

- **Scalar/vectorized MLPs:** All tensor math in `scalar/` and `vectorized/` ultimately uses NumPy arrays (or compatible wrappers).
- **Common utilities:** `common/data_utils.py`, `common/cross_entropy.py`, and `common/softmax.py` operate on `xp` objects, which default to NumPy unless you request GPU mode.
- **Training scripts:** When you run without `--gpu`, every forward/backward pass is powered by NumPy.

## Skills to Practice

- Creating arrays: `np.array`, `np.zeros`, `np.random.randn`.
- Vectorized ops: broadcasting, `np.dot`, `np.matmul`.
- Reshaping: `reshape`, `transpose`, `swapaxes`.
- Reductions: `np.sum`, `np.mean`, `np.max`, `np.argmax`.

Brush up on NumPy and you’ll understand 100% of the CPU path in this repo. The official docs are excellent: <https://numpy.org/doc/stable/>