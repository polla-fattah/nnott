[MyHome](https://polla.dev) | [Tutorial Hub](../README.md) | [Code Base](https://github.com/polla-fattah/nnott/) | [Architectures](../architecture-gallery.md)

# CuPy Quick Notes

**Breadcrumb:** [Home](../README.md) / [Project Tour](../project-tour.md) / CuPy Quick Notes


## Why You Need CuPy

CuPy mirrors the NumPy API but executes on NVIDIA GPUs. By swapping the backend (`backend.use_gpu()` or `--gpu` flag), the project instantly accelerates heavy tensor ops without changing core logic.

## Where It Appears in the Project

- **Backend proxy:** `common/backend.py` sets `xp = cupy` when you call `backend.use_gpu()` or pass `--gpu` to `convolutional/main.py`.
- **Vectorized & CNN modules:** Every layer/optimizer referencing `xp` automatically runs on CuPy when available.
- **Diagnostics:** `scripts/test_cupy.py` ensures your drivers, CUDA runtime, and CuPy installation are healthy before long training jobs.

## Setup Checklist

1. Confirm that NVIDIA drivers + CUDA toolkit are installed (`nvidia-smi` should list your GPU).
2. Install a CuPy build that matches your CUDA version (see below for commands).
3. Verify with `python -c "import cupy; print(cupy.__version__)"`.
4. Run `python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096` to confirm kernels, memory pools, and custom ops behave.

Once configured, add `--gpu` to any training command to feel the difference in throughput.

## Installation Notes

| CUDA version | Pip command | Conda (conda-forge) |
| --- | --- | --- |
| CUDA 12.x | `pip install cupy-cuda12x` | `conda install -c conda-forge cupy cudatoolkit=12.1` |
| CUDA 11.x | `pip install cupy-cuda11x` | `conda install -c conda-forge cupy cudatoolkit=11.8` |
| CUDA 10.x | `pip install cupy-cuda102` | `conda install -c conda-forge cupy cudatoolkit=10.2` |

If you do not have CUDA installed (e.g., WSL without GPU passthrough), install `pip install cupy-cuda12x` inside a CUDA-enabled environment or fall back to CPU-only runs (NumPy).

### Quick install script (Bash/WSL)

```bash
#!/usr/bin/env bash
set -euo pipefail
CUDA_VER=${1:-12x}          # pass 11x or 12x depending on driver
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install cupy-cuda${CUDA_VER} numpy tqdm matplotlib
python - <<'PY'
import cupy as cp
info = cp.cuda.runtime.getDeviceProperties(0)
print("CuPy OK on:", info["name"].decode())
PY
```

Save as `scripts/install_cupy.sh`, run `bash scripts/install_cupy.sh 12x`, and you will end up with a virtual environment that already validates CuPy + GPU visibility.

## Smoke Tests

1. **Driver check:** `nvidia-smi` should report utilization and driver versions.
2. **Minimal CuPy test:**
   ```bash
   python - <<'PY'
   import cupy as cp
   a = cp.arange(1_000_000, dtype=cp.float32)
   b = cp.sin(a)
   print("Device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
   print("Result checksum:", float(b.sum()))
   PY
   ```
   If this succeeds, the CUDA runtime and CuPy wheel match.
3. **Project stress test:** `python scripts/test_cupy.py --stress-seconds 10 --stress-size 4096` pounds on GEMMs, reductions, and random kernels. Increase `--stress-seconds` if you suspect thermal throttling or unstable overclocks.

## Using the Backend Flags

- Scalar/vectorized MLPs remain CPU-only; the flag only affects vectorized/CNN entry points.
- `python convolutional/main.py resnet18 --epochs 1 --batch-size 64 --gpu` calls `backend.use_gpu()` and every module switches to CuPy.
- Turn GPUs off by simply omitting `--gpu` or running `python -c "from common import backend; backend.use_cpu()"` before importing trainers.

If `--gpu` raises `ModuleNotFoundError: cupy`, follow the installation steps above, then rerun the training command.

[Previous (tqdm Quick Notes)](tqdm.md) | [Back to Project Tour](../project-tour.md) | [Next (Matplotlib Quick Notes)](matplotlib.md)

**Navigation:**
[Back to Project Tour](../project-tour.md)

---

MIT License | [About](../about.md) | [Code Base](https://github.com/polla-fattah/nnott/)
