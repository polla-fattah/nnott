import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math


class DataUtility:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)

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

        # Zero-center using training mean for better ReLU dynamics
        train_mean = float(np.mean(X_train))
        X_train = X_train - train_mean
        X_test = X_test - train_mean

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
    def show_samples(cls, images, labels, num_samples=10, title="Sample images"):
        num_samples = min(num_samples, len(images))
        idxs = np.random.choice(len(images), size=num_samples, replace=False)

        cols = min(num_samples, 5)
        rows = math.ceil(num_samples / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = np.array(axes).reshape(-1)

        for ax, idx in zip(axes, idxs):
            img = cls._to_image(images[idx])
            ax.imshow(img, cmap="gray")
            ax.set_title(str(labels[idx]))
            ax.axis("off")

        for ax in axes[num_samples:]:
            ax.axis("off")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
