import time
import numpy as _np
import matplotlib.pyplot as plt
from tqdm import tqdm

import common.backend as backend
from common.cross_entropy import CrossEntropyLoss
from common.model_io import save_model, load_model as load_model_state

xp = backend.xp


class ConvTrainer:
    def __init__(self, model, optimizer, num_classes=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.num_classes = num_classes
        self.loss_history = []

    def train(
        self,
        X_train,
        y_train,
        epochs=10,
        batch_size=64,
        verbose=True,
        augment=True,
        lr_schedule=None,
    ):
        X = backend.to_device(X_train, dtype=xp.float32)
        y = backend.to_device(y_train, dtype=xp.int64)
        n = len(X)
        self.loss_history = []
        t0 = time.time()
        base_lr = getattr(self.optimizer, "lr", None)
        schedule = lr_schedule or self._default_multistep_schedule(epochs, base_lr)

        for epoch in range(1, epochs + 1):
            if schedule is not None and epoch in schedule:
                if hasattr(self.optimizer, "lr"):
                    self.optimizer.lr = schedule[epoch]
                    if verbose:
                        print(f"[LR] -> {self.optimizer.lr:.6f}")

            if hasattr(self.model, "train"):
                self.model.train()

            idx = xp.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]
            running = 0.0
            batches = (n + batch_size - 1) // batch_size
            if verbose:
                print(f"\n=== Epoch {epoch}/{epochs} ===")
                print(f"Batches: {batches} | Batch size: {batch_size}")

            for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}", unit="batch"):
                end = min(start + batch_size, n)
                xb = Xs[start:end]
                yb = ys[start:end]
                if augment:
                    xb = self._augment_batch(xb)
                self.model.zero_grad()
                logits = self.model.forward(xb)
                loss = self.criterion.forward(logits, yb)
                running += float(loss) * len(xb)
                grad = self.criterion.backward(logits, yb)
                self.model.backward(grad)
                self.optimizer.step(self.model.parameters(), batch_size=len(xb))

            avg = running / n
            self.loss_history.append(float(avg))
            if verbose:
                print(f"Epoch {epoch}: avg loss {avg:.5f}")

        if verbose:
            print(f"\n⏱ Total training time: {time.time() - t0:.2f}s")

    def evaluate(self, X_test, y_test, batch_size=256):
        if hasattr(self.model, "eval"):
            self.model.eval()
        X = backend.to_device(X_test, dtype=xp.float32)
        y = backend.to_device(y_test, dtype=xp.int64)
        n = len(X)
        preds = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            logits = self.model.forward(X[start:end])
            preds.append(xp.argmax(logits, axis=1))
        preds = xp.concatenate(preds)
        acc = float(backend.to_cpu((preds == y).mean()))
        print(f"Test accuracy: {acc * 100:.2f}%")
        return acc

    def plot_loss(self):
        if not self.loss_history:
            return
        plt.figure()
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker="o")
        plt.title("Convolutional Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)


    def show_misclassifications(self, X_test, y_test, max_images=25, cols=5, batch_size=512):
        if hasattr(self.model, "eval"):
            self.model.eval()
        n = len(X_test)
        if n == 0:
            print("No samples provided for misclassification display.")
            return _np.array([], dtype=_np.int64)

        preds = xp.empty(n, dtype=xp.int64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = backend.to_device(X_test[start:end], dtype=xp.float32)
            logits = self.model.forward(xb)
            preds[start:end] = xp.argmax(logits, axis=1)

        y_all = backend.to_device(y_test, dtype=xp.int64)
        mis_idx = xp.where(preds != y_all)[0]
        mis_idx_cpu = backend.to_cpu(mis_idx).astype(_np.int64, copy=False)
        total = len(mis_idx_cpu)
        if total == 0:
            print("No misclassifications dYZ%")
            return mis_idx_cpu
        mis_idx_cpu = mis_idx_cpu[:max_images]
        imgs = _np.asarray(X_test)[mis_idx_cpu]
        preds_cpu_all = backend.to_cpu(preds)
        y_cpu_all = backend.to_cpu(y_all)
        preds_cpu = preds_cpu_all[mis_idx_cpu]
        y_cpu = y_cpu_all[mis_idx_cpu]
        rows = int(_np.ceil(len(mis_idx_cpu) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = _np.atleast_1d(axes).ravel()
        for ax, img, true, pred in zip(axes, imgs, y_cpu, preds_cpu):
            disp = img[0]
            ax.imshow(disp, cmap="gray")
            ax.set_title(f"T:{int(true)} P:{int(pred)}")
            ax.axis("off")
        for ax in axes[len(mis_idx_cpu):]:
            ax.axis("off")
        fig.suptitle(f"Misclassifications: showing {len(mis_idx_cpu)} of {total}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        return mis_idx_cpu

    def save_model(self, path, metadata=None):
        meta = dict(metadata or {})
        meta.setdefault("num_classes", self.num_classes)
        save_model(self.model, path, meta)
        return meta

    def load_model(self, path):
        return load_model_state(self.model, path)

    @staticmethod
    def _default_multistep_schedule(epochs, base_lr):
        if base_lr is None or epochs < 4:
            return None
        mid = max(2, int(epochs * 0.5))
        late = max(mid + 1, int(epochs * 0.75))
        return {mid: base_lr * 0.5, late: base_lr * 0.1}

    def _augment_batch(self, xb, max_shift=2):
        if xb.ndim != 4:
            return xb
        shifted = xp.empty_like(xb)
        for i in range(len(xb)):
            dx = int(xp.random.randint(-max_shift, max_shift + 1))
            dy = int(xp.random.randint(-max_shift, max_shift + 1))
            shifted[i] = xp.roll(xp.roll(xb[i], dy, axis=1), dx, axis=2)
        return shifted
