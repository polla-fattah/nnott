# Performance Profiles

**Breadcrumb:** [Home](README.md) / Performance Profiles


Use this page as a reference for typical training times and accuracies across architectures. The numbers below were collected on a representative development machine (Intel i7-11700K CPU, NVIDIA RTX 3060 GPU) using the default settings from the main scripts. Your hardware may differ, but these baselines help you spot configurations that are way off.

> **Note:** Times include only the training loop (not plotting or misclassification collection). GPU runs use `--gpu` with CuPy installed.

---

## Scalar & Vectorized MLPs

| Model | Epochs | Batch Size | CPU Time (approx) | GPU Time | Test Accuracy |
| --- | --- | --- | --- | --- | --- |
| Scalar MLP `(128,64)` | 1 | 64 | ~5 min | N/A | 85–90% |
| Scalar MLP `(128,64)` | 5 | 64 | ~25 min | N/A | 93–95% |
| Vectorized MLP `(256,128)` | 1 | 128 | ~40 sec | ~8 sec | 95–96% |
| Vectorized MLP `(256,128)` | 5 | 128 | ~3 min | ~40 sec | 97–98% |

**Checklist:** If your vectorized run takes 10× longer than the table, double-check that NumPy is built for your CPU, or switch to `--gpu`.

---

## Convolutional Architectures (CPU vs GPU)

| Architecture | Epochs | Batch Size | CPU Time | GPU Time | Test Accuracy |
| --- | --- | --- | --- | --- | --- |
| BaselineCNN | 1 | 64 | ~2 min | ~30 sec | 98% |
| LeNet-5 | 1 | 64 | ~3 min | ~45 sec | 98% |
| AlexNet | 1 | 64 | ~10 min | ~1.5 min | 99% |
| VGG16 | 1 | 64 | ~25 min | ~3.5 min | 99% |
| ResNet18 | 1 | 64 | ~18 min | ~2.5 min | 99% |
| EfficientNet-Lite0 | 1 | 64 | ~20 min | ~3 min | 99% |
| ConvNeXt-Tiny | 1 | 64 | ~30 min | ~4 min | 99% |

**Notes:**
- Times scale roughly linearly with epochs. Multiply by 5 for a 5-epoch estimate.
- GPU speedup varies with architecture depth; bigger models benefit more.
- For ResNet/ConvNeXt, consider enabling gradient clipping (`--grad-clip 5`) to avoid rare divergence.

---

## Quick Guidance

- **CPU vs GPU delta:** Expect 4–8× speedups for CNNs. If the GPU run is slower than CPU, check that CuPy is actually active (look for “GPU backend enabled via CuPy” message).
- **Accuracy sanity check:** All networks should exceed 98% on MNIST after 1–2 epochs. Significantly lower accuracy usually indicates missing normalization, mislabeled data, or an incorrect architecture configuration.
- **Batch size impact:** Doubling the batch size roughly halves the number of steps per epoch but may hurt generalization slightly. Monitor accuracy when changing it.

---

## Lab Exercise

1. Pick two architectures (one shallow, one deep) and reproduce the 1-epoch CPU and GPU timings on your own hardware. Record the numbers in your [Experiment Log](experiment-log-template.md).
2. Explain any deviations from the table (e.g., slower GPU due to integrated graphics, faster CPU due to more cores).
3. Share your findings with classmates to build a classroom-wide performance reference.

**Navigation:**
[Back to Project Tour](../project-tour.md)
