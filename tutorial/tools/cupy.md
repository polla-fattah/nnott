# CuPy Quick Notes

**Breadcrumb:** [Home](../README.md) / [Project Tour](../project-tour.md) / CuPy Quick Notes


## Why You Need CuPy

CuPy mirrors the NumPy API but executes on NVIDIA GPUs. By swapping the backend (`backend.use_gpu()` or `--gpu` flag), the project instantly accelerates heavy tensor ops without changing core logic.

## Where It Appears in the Project

- **Backend proxy:** `common/backend.py` sets `xp = cupy` when you call `backend.use_gpu()` or pass `--gpu` to `convolutional/main.py`.
- **Vectorized & CNN modules:** Every layer/optimizer referencing `xp` automatically runs on CuPy when available.
- **Diagnostics:** `scripts/test_cupy.py` ensures your drivers, CUDA runtime, and CuPy installation are healthy before long training jobs.

## Setup Checklist

1. Install a CuPy build compatible with your CUDA version: <https://docs.cupy.dev/en/stable/install.html>
2. Verify with `python -c "import cupy; print(cupy.__version__)"`.
3. Run `python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096` to confirm kernels, memory pools, and custom ops behave.

Once configured, add `--gpu` to any training command to feel the difference in throughput.

[Previous (tqdm Quick Notes)](tqdm.md) | [Back to Project Tour](../project-tour.md) | [Next (Matplotlib Quick Notes)](matplotlib.md)

**Navigation:**
[Back to Project Tour](../project-tour.md)
