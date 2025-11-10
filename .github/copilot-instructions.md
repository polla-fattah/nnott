## How to be productive in this repo

This project is a small educational neural-net implemented in plain NumPy. The goal of this file is to give an AI coding assistant the exact, project-specific facts and examples needed to make correct, low-risk changes.

Key facts
- Entrypoint: `main.py` — it loads data, constructs `Network`, runs `Trainer.train`, and visualizes results.
- Data: `data/` contains NumPy arrays: `train_images.npy`, `train_labels.npy`, `test_images.npy`, `test_labels.npy`. `DataUtility.load_data()` loads and normalizes images to float32 in [0,1].
- Shapes: images are expected as 28x28 (or flattened 784). Example shapes printed in `main.py`: train (60000,28,28), test (10000,28,28).

Architecture & core files (quick map)
- `data_utils.py` — DataUtility: loads .npy files, normalizes images, converts labels to ints, helpers `show_samples` and `_to_image` used by trainer for plotting.
- `network.py` — Network: a small feed-forward stack of `Layer`s. `forward` flattens inputs (calls .flatten()) and passes through each layer. `predict` returns argmax of raw outputs.
- `layer.py` — Layer: contains list of `Neuron` objects. `forward` computes outputs list-wise. `backward` computes per-neuron delta, updates neuron.weights and neuron.bias in-place (gradient descent), and returns input-gradient for previous layer.
- `neuron.py` — Neuron: stores `weights`, `bias`, `activation` and `last_input/last_output`. Activations: 'sigmoid' (default) and 'relu' supported. Weights initialized as randn * 0.01.
- `trainer.py` — Trainer: single-sample SGD loop (no batching). Uses MSE loss and one-hot encoding for labels. Uses `tqdm` for progress and `matplotlib` to plot loss and predictions.

Important behavioral details to preserve
- Training is single-sample SGD; the loop performs forward -> compute mse loss -> compute gradient -> call `network.backward(grad)`. There is no optimizer abstraction; weight updates happen inside `Layer.backward`.
- Labels are converted to one-hot vectors in `Trainer._one_hot()` and MSE is used as loss. The output layer uses the same activation as hidden layers (often 'sigmoid') and `predict()` uses argmax on raw outputs.
- `Network.forward()` flattens inputs to 1D. Any change to input handling must keep compatibility with `DataUtility._to_image` and `main.py` expectations.
- Plotting functions call `plt.show()` (blocking GUI). For headless CI, consider switching to a non-interactive backend before plotting.

Dev workflows & commands (Windows PowerShell)
- Create a venv and install dependencies (recommended):

  python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1; pip install --upgrade pip; pip install numpy matplotlib tqdm

- Run the example training flow:

  .\\.venv\\Scripts\\Activate.ps1; python main.py

- Quick syntax check (no tests in repo):

  .\\.venv\\Scripts\\Activate.ps1; python -m py_compile main.py network.py layer.py neuron.py trainer.py data_utils.py

Project-specific conventions
- Use small random init: `weights` seeded as `np.random.randn(...) * 0.01`. When changing initialization, keep the small-magnitude convention to avoid exploding activations.
- Learning rate is passed into `Layer`/`Neuron` constructors; many experiments set it in `Network(...)` and propagate it to layers. Changing LR semantics should be done consistently across layers.
- Always use `tqdm` for training loops (see `trainer.py`). Avoid removing `tqdm` unless intentionally changing UX.

Integration points and cross-cutting concerns
- Data files in `data/` are required at runtime. New dataset code should match the same normalization scheme (float32, /255.0) and shape expectations.
- Plotting and GUI: training and evaluation invoke `matplotlib.pyplot.show()` which is interactive — in automated runs prefer non-interactive backend (`matplotlib.use('Agg')`) or saving figures to disk.

Good examples to copy from
- One-hot encoding for labels: `Trainer._one_hot()` (trainer.py)
- Flattening and forward pass: `Network.forward()` (network.py)
- Per-neuron activation and numerical-safety for sigmoid: `Neuron.activate()` (neuron.py)

When making changes, prioritize these safety checks
- Preserve input normalization and flattening compatibility.
- Keep in-place weight updates semantics unless implementing a full optimizer replacement; update tests/examples accordingly.
- If changing activation functions or output-layer semantics, update `Trainer` (loss/one-hot) and `predict()` behavior to match (e.g., switch to cross-entropy + softmax if moving away from MSE).

If anything above is unclear or you'd like extra examples (unit tests, CI-friendly plotting, or a requirements.txt), tell me which area to expand and I'll update this file.
