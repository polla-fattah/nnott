import torch
import cupy as cp

print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Torch GPU:", torch.cuda.get_device_name(0))

print("CuPy version:", cp.__version__)
print("CuPy device count:", cp.cuda.runtime.getDeviceCount())

x = cp.arange(5)
print("CuPy x:", x)
print("CuPy x^2 back on CPU:", (x**2).get())
