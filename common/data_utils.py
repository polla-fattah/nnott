import numpy as np
from pathlib import Path
 # moved to common package

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_EPS = 1e-6


class DataUtility:
    def __init__(self, data_dir="data"):
        data_path = Path(data_dir)

        # allow callers to pass relative paths regardless of current working directory
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path

        self.data_dir = data_path

    def load_data(
        self,
        train_images_file="train_images.npy",
        train_labels_file="train_labels.npy",
        test_images_file="test_images.npy",
        test_labels_file="test_labels.npy",
    ):
        X_train = np.load(self.data_dir / train_images_file)
        y_train = np.load(self.data_dir / train_labels_file)
        X_test = np.load(self.data_dir / test_images_file)
        y_test = np.load(self.data_dir / test_labels_file)

        # IMPORTANT: convert images to float32 and normalize
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0

        # Standardize using training statistics: zero-center and scale by std
        train_mean = float(np.mean(X_train))
        train_std = float(np.std(X_train) + 1e-7)
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std

        # labels stay as ints
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

        return X_train, y_train, X_test, y_test

    @staticmethod
    def _to_image(img):
        arr = np.array(img)
        if arr.ndim == 1:
            side = int(np.sqrt(arr.size))
            return arr.reshape(side, side)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            return arr.squeeze(-1)
        return arr

    @classmethod
    def sample_images(cls, images, labels, num_samples=10):
        """Return a list of (image, label) pairs for visualization."""
        num_samples = min(num_samples, len(images))
        if num_samples == 0:
            return []
        idxs = np.random.choice(len(images), size=num_samples, replace=False)
        samples = []
        for idx in idxs:
            samples.append((cls._to_image(images[idx]), labels[idx]))
        return samples

    # Backward-compatible alias
    @classmethod
    def show_samples(cls, images, labels, num_samples=10, title=None):
        """Deprecated: returns sample data instead of plotting."""
        return cls.sample_images(images, labels, num_samples=num_samples)

    @staticmethod
    def train_val_split(X, y, val_fraction=0.1, seed=None):
        """Split arrays into train/validation subsets."""
        val_fraction = float(max(0.0, min(0.9, val_fraction)))
        if val_fraction <= 0.0 or len(X) == 0:
            return X, y, None, None
        n_total = len(X)
        n_val = max(1, int(n_total * val_fraction))
        rng = np.random.default_rng(seed)
        indices = np.arange(n_total)
        rng.shuffle(indices)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        X_val = X[val_idx]
        y_val = y[val_idx]
        X_train = X[train_idx]
        y_train = y[train_idx]
        return X_train, y_train, X_val, y_val


def ensure_label_format(labels, num_classes):
    """
    Validate and normalize labels to int64 class indices.
    Supports either 1D int arrays or 2D one-hot matrices.
    """
    arr = np.asarray(labels)
    if arr.ndim == 2:
        if arr.shape[1] != num_classes:
            raise ValueError(
                f"One-hot label matrix has {arr.shape[1]} columns but expected {num_classes}."
            )
        row_sums = arr.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=LABEL_EPS):
            raise ValueError("Each one-hot label row must sum to 1.")
        arr = np.argmax(arr, axis=1)
    elif arr.ndim != 1:
        raise ValueError("Labels must be 1D indices or 2D one-hot matrices.")

    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)

    if arr.size == 0:
        return arr
    min_label = int(arr.min())
    max_label = int(arr.max())
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            f"Label values must be in [0, {num_classes-1}] but range [{min_label}, {max_label}] was found."
        )
    return arr
