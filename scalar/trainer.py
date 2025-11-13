import numpy as np
from common.data_utils import DataUtility
from tqdm import tqdm  # <- ALWAYS use tqdm now
import time  # at the top of trainer.py
from common.cross_entropy import CrossEntropyLoss
from scalar.optimizer import SGD, Adam


class Trainer:
    def __init__(self, network, num_classes=10, optimizer="adam", lr=0.001, weight_decay=0.0):
        self.network = network
        self.num_classes = num_classes
        self.loss_history = []
        self.criterion = CrossEntropyLoss(reduction="mean")
        if isinstance(optimizer, str):
            if optimizer.lower() == "sgd":
                self.optimizer = SGD(lr=lr, momentum=0.0, weight_decay=weight_decay)
            elif optimizer.lower() == "adam":
                self.optimizer = Adam(lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError("Unknown optimizer; use 'sgd' or 'adam' or pass an instance")
        else:
            self.optimizer = optimizer

    def _one_hot(self, label):
        v = np.zeros(self.num_classes, dtype=np.float32)
        v[int(label)] = 1.0
        return v

    def train(self, X_train, y_train, epochs=5, batch_size=64, verbose=True):
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
            batches = (n + batch_size - 1) // batch_size
            if verbose:
                print(f"Batches/epoch: {batches} | Batch size: {batch_size}")

            # iterate over mini-batches
            for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}", unit="batch"):
                end = min(start + batch_size, n)
                bs = end - start

                # zero accumulators
                for layer in self.network.layers:
                    layer.zero_grad()

                # accumulate over batch
                for i in range(start, end):
                    x = X_shuf[i]
                    y = y_shuf[i]

                    logits = self.network.forward(x)
                    loss = self.criterion.forward(logits, int(y))
                    total_loss += loss

                    grad_logits = self.criterion.backward(logits, int(y))
                    self.network.backward(grad_logits)

                # optimizer step with average gradients
                self.optimizer.step(self.network.layers, batch_size=bs)

            avg_loss = total_loss / n
            self.loss_history.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg_loss:.6f}")

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nTotal training time: {elapsed:.2f} seconds "
              f"({elapsed / epochs:.2f} sec/epoch)")

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

    def get_random_predictions(self, X_test, y_test, num_samples=10):
        num_samples = min(num_samples, len(X_test))
        if num_samples == 0:
            return []
        idxs = np.random.choice(len(X_test), size=num_samples, replace=False)
        samples = []
        for idx in idxs:
            img = DataUtility._to_image(X_test[idx])
            true_label = int(y_test[idx])
            pred_label = self.network.predict(X_test[idx])
            samples.append((img, true_label, pred_label))
        return samples
