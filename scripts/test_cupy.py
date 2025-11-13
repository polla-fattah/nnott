"""
Quick CuPy self-test to verify the CUDA stack is operational.

Usage
-----
Run `python scripts/test_cupy.py [--stress-seconds N] [--stress-size M]`. The
script exits with status 0 only if all checks succeed, and prints diagnostic
information otherwise. Increase `--stress-seconds` or `--stress-size` to apply
heavier GPU load.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Callable, List, Tuple

import numpy as np


def require_cupy():
    try:
        import cupy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"[FAIL] Unable to import CuPy: {exc}")
        sys.exit(1)
    return cp


def format_bytes(num_bytes: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for suffix in suffixes:
        if value < 1024.0:
            return f"{value:.1f} {suffix}"
        value /= 1024.0
    return f"{value:.1f} PB"


def check_device(cp) -> Tuple[bool, str]:
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if count == 0:
            return False, "No CUDA devices visible to CuPy"
        name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        return True, f"{name} (device 0) | {count} device(s) | free {format_bytes(free_mem)} / total {format_bytes(total_mem)}"
    except cp.cuda.runtime.CUDARuntimeError as exc:
        return False, f"CUDA runtime error: {exc}"


def check_basic_ops(cp) -> Tuple[bool, str]:
    try:
        x = cp.arange(1024, dtype=cp.float32)
        y = cp.sqrt(x) + cp.sin(x)
        cp.cuda.runtime.deviceSynchronize()
        return True, f"basic ops OK (sum={float(cp.sum(y)):.3f})"
    except Exception as exc:
        return False, f"basic ops failed: {exc}"


def check_matmul(cp) -> Tuple[bool, str]:
    try:
        rng = cp.random.default_rng(42)
        a = rng.standard_normal((256, 128), dtype=cp.float32)
        b = rng.standard_normal((128, 64), dtype=cp.float32)
        c = a @ b
        cp.cuda.runtime.deviceSynchronize()
        return True, f"matmul OK (norm={float(cp.linalg.norm(c)):.3f})"
    except Exception as exc:
        return False, f"matmul failed: {exc}"


def check_numpy_agreement(cp) -> Tuple[bool, str]:
    try:
        host = np.linspace(-3, 3, num=2048, dtype=np.float32)
        gpu = cp.asarray(host)
        gpu_out = cp.tanh(gpu * 1.5 + 0.25)
        cpu_out = np.tanh(host * 1.5 + 0.25)
        diff = cp.abs(gpu_out - cp.asarray(cpu_out))
        max_err = float(cp.max(diff))
        cp.cuda.runtime.deviceSynchronize()
        passed = max_err < 1e-6
        detail = f"max abs error {max_err:.2e}"
        return passed, detail if passed else f"numpy mismatch: {detail}"
    except Exception as exc:
        return False, f"numpy agreement failed: {exc}"


def check_custom_kernel(cp) -> Tuple[bool, str]:
    try:
        kernel = cp.ElementwiseKernel(
            "float32 x, float32 y",
            "float32 z",
            "z = x * y + 0.1f",
            "cupy_self_test_axpy",
        )
        x = cp.linspace(0, 1, 1024, dtype=cp.float32)
        y = cp.linspace(1, 2, 1024, dtype=cp.float32)
        out = kernel(x, y)
        cp.cuda.runtime.deviceSynchronize()
        return True, f"custom kernel OK (mean={float(cp.mean(out)):.3f})"
    except Exception as exc:
        return False, f"custom kernel failed: {exc}"


def _describe_gemm(size: int) -> str:
    flops = 2 * (size**3)
    return f"{size}x{size} GEMM (~{flops/1e12:.2f} TFLOPs)"


def run_stress_test(cp, seconds: float, size: int) -> Tuple[bool, str]:
    if seconds <= 0:
        return True, "stress skipped (0 seconds requested)"

    if size <= 0:
        return False, "stress size must be positive"

    try:
        rng = cp.random.default_rng(7)
        start = time.perf_counter()
        matmul_count = 0
        while time.perf_counter() - start < seconds:
            a = rng.standard_normal((size, size), dtype=cp.float32)
            b = rng.standard_normal((size, size), dtype=cp.float32)
            _ = a @ b
            cp.cuda.runtime.deviceSynchronize()
            matmul_count += 1

        elapsed = time.perf_counter() - start
        detail = f"{matmul_count}x {_describe_gemm(size)} in {elapsed:.1f}s"
        return True, f"stress OK ({detail})"
    except Exception as exc:
        return False, f"stress test failed: {exc}"


def parse_args():
    parser = argparse.ArgumentParser(description="CuPy self-test / stress utility")
    parser.add_argument(
        "--stress-seconds",
        type=float,
        default=10.0,
        help="minimum time to keep GPU busy with GEMMs (default: 10s)",
    )
    parser.add_argument(
        "--stress-size",
        type=int,
        default=4096,
        help="square matrix size used during stress (default: 4096)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cp = require_cupy()

    checks: List[Tuple[str, Callable]] = [
        ("Device visibility", check_device),
        ("Elementwise math", check_basic_ops),
        ("Matrix multiply", check_matmul),
        ("NumPy parity", check_numpy_agreement),
        ("Custom kernel", check_custom_kernel),
        (
            f"Stress ({args.stress_seconds:.1f}s @ {args.stress_size})",
            lambda cp: run_stress_test(cp, args.stress_seconds, args.stress_size),
        ),
    ]

    failures = 0
    print(f"CuPy version: {cp.__version__}")
    for name, fn in checks:
        ok, detail = fn(cp)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {detail}")
        if not ok:
            failures += 1

    if failures:
        print(f"CuPy self-test failed ({failures} check(s) failed).")
        sys.exit(1)

    print("CuPy self-test succeeded. CUDA stack looks healthy.")


if __name__ == "__main__":
    main()
