import numpy as np
import matplotlib.pyplot as plt
from data_utils import DataUtility
from tqdm import tqdm  # <- ALWAYS use tqdm now
import time  # at the top of trainer.py


def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)


def mse_loss_grad(pred, target):
    return 2 * (pred - target) / pred.size


class Trainer:
    def __init__(self, network, num_classes=10):
        self.network = network
        self.num_classes = num_classes
        self.loss_history = []

    def _one_hot(self, label):
        v = np.zeros(self.num_classes, dtype=np.float32)
        v[int(label)] = 1.0
        return v

    def train(self, X_train, y_train, epochs=5, verbose=True):
        n = len(X_train)
        self.loss_history = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"\n=== Epoch {epoch}/{epochs} ===")

            # Shuffle data each epoch
            indices = np.random.permutation(n)
            X_shuf = X_train[indices]
            y_shuf = y_train[indices]

            total_loss = 0.0

            # tqdm over indices so we’re 100% sure it runs
            for i in tqdm(range(n), desc=f"Epoch {epoch}", unit="sample"):
                x = X_shuf[i]
                y = y_shuf[i]

                target = self._one_hot(y)
                pred = self.network.forward(x)
                loss = mse_loss(pred, target)
                total_loss += loss

                grad = mse_loss_grad(pred, target)
                self.network.backward(grad)

            avg_loss = total_loss / n
            self.loss_history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        # After all epochs, show loss plot
        self._plot_loss()

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n⏱ Total training time: {elapsed:.2f} seconds "
              f"({elapsed / epochs:.2f} sec/epoch)")

    def _plot_loss(self):
        if not self.loss_history:
            return

        epochs = range(1, len(self.loss_history) + 1)
        plt.figure()
        plt.plot(epochs, self.loss_history, marker='o')
        plt.title("Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def evaluate(self, X_test, y_test):
        correct = 0
        total = len(X_test)

        for x, y in zip(X_test, y_test):
            pred_class = self.network.predict(x)
            if pred_class == int(y):
                correct += 1

        acc = correct / total
        print(f"Test accuracy: {acc * 100:.2f}%")
        return acc

    def show_random_predictions(self, X_test, y_test, num_samples=10):
        num_samples = min(num_samples, len(X_test))
        idxs = np.random.choice(len(X_test), size=num_samples, replace=False)

        cols = min(num_samples, 5)
        rows = int(np.ceil(num_samples / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
        axes = axes.flatten()

        for ax, idx in zip(axes, idxs):
            img = DataUtility._to_image(X_test[idx])
            true_label = int(y_test[idx])
            pred_label = self.network.predict(X_test[idx])

            ax.imshow(img, cmap="gray")
            ax.set_title(f"T: {true_label}  P: {pred_label}")
            ax.axis("off")

        for ax in axes[num_samples:]:
            ax.axis("off")

        fig.suptitle("Random Test Predictions")
        plt.tight_layout()
        plt.show()
