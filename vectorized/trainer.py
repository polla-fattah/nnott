import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from common.cross_entropy import CrossEntropyLoss


class VTrainer:
    def __init__(self, model, optimizer, num_classes=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.num_classes = num_classes
        self.loss_history = []

    def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=True):
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.int64)
        n = len(X)
        self.loss_history = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]
            total = 0.0
            batches = (n + batch_size - 1) // batch_size
            if verbose:
                print(f"\n=== Epoch {epoch}/{epochs} ===")
                print(f"Batches/epoch: {batches} | Batch size: {batch_size}")

            for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}", unit="batch"):
                end = min(start + batch_size, n)
                Xb = Xs[start:end]
                yb = ys[start:end]

                self.model.zero_grad()
                logits = self.model.forward(Xb)
                loss = self.criterion.forward(logits, yb)
                total += float(loss) * len(Xb)
                grad_logits = self.criterion.backward(logits, yb)
                self.model.backward(grad_logits)
                self.optimizer.step(self.model.parameters(), batch_size=len(Xb))

            avg = total / n
            self.loss_history.append(avg)
            if verbose:
                print(f"Epoch {epoch}/{epochs} - Avg Loss: {avg:.6f}")

        if verbose:
            print(f"\n⏱ Total training time: {time.time()-t0:.2f}s")

    def evaluate(self, X_test, y_test):
        X = np.asarray(X_test, dtype=np.float32)
        y = np.asarray(y_test, dtype=np.int64)
        logits = self.model.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = float((preds == y).mean())
        print(f"Test accuracy: {acc*100:.2f}%")
        return acc

    def plot_loss(self):
        if not self.loss_history:
            return
        plt.figure()
        plt.plot(range(1, len(self.loss_history)+1), self.loss_history, marker='o')
        plt.title("Vectorized Training Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
