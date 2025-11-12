import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from common.cross_entropy import CrossEntropyLoss
from common.model_io import save_model, load_model as load_model_state


class VTrainer:
    def __init__(self, model, optimizer, num_classes=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.num_classes = num_classes
        self.loss_history = []

    def train(self, X_train, y_train, epochs=10, batch_size=32, verbose=True, lr_schedule=True, augment=True):
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.int64)
        n = len(X)
        self.loss_history = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            # optional LR schedule
            if lr_schedule:
                if epoch == 10:
                    if hasattr(self.optimizer, 'lr'):
                        self.optimizer.lr = self.optimizer.lr * 0.3
                        if verbose:
                            print(f"[LR] set to {self.optimizer.lr}")
                if epoch == 15:
                    if hasattr(self.optimizer, 'lr'):
                        self.optimizer.lr = self.optimizer.lr * (1.0/3.0)
                        if verbose:
                            print(f"[LR] set to {self.optimizer.lr}")

            # training mode
            if hasattr(self.model, 'train'):
                self.model.train()
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
                if augment:
                    logits = self.model.forward(self._augment_batch(Xb))
                else:
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
        if hasattr(self.model, 'eval'):
            self.model.eval()
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

    def show_misclassifications(self, X_test, y_test, max_images=None, cols=5):
        X = np.asarray(X_test, dtype=np.float32)
        y = np.asarray(y_test, dtype=np.int64)
        logits = self.model.forward(X)
        preds = np.argmax(logits, axis=1)
        mis_idx = np.where(preds != y)[0]

        total = len(mis_idx)
        if total == 0:
            print("No misclassifications found.")
            return mis_idx

        if max_images is not None:
            mis_idx = mis_idx[:max_images]

        n = len(mis_idx)
        cols = max(1, min(cols, n))
        rows = int(np.ceil(n / cols))

        # Infer image side length from feature size if flattened
        D = X.shape[1]
        side = int(np.sqrt(D))
        if side * side != D:
            side = None  # can't reshape to square cleanly

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
        axes = np.atleast_1d(axes).ravel()

        for ax, idx in zip(axes, mis_idx):
            img = X[idx]
            if side is not None:
                img = img.reshape(side, side)
            # rescale per-image for display
            imin, imax = float(img.min()), float(img.max())
            if imax > imin:
                disp = (img - imin) / (imax - imin)
            else:
                disp = img
            ax.imshow(disp, cmap="gray")
            ax.set_title(f"T:{int(y[idx])} P:{int(preds[idx])}")
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"Misclassifications: showing {n} of {total}")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
        return mis_idx

    def save_model(self, path, metadata=None):
        meta = dict(metadata or {})
        meta.setdefault("num_classes", self.num_classes)
        save_model(self.model, path, meta)
        return meta

    def load_model(self, path):
        return load_model_state(self.model, path)

    # --- augmentation utilities ---
    def _augment_batch(self, Xb_flat, side=28, max_shift=2, max_rotate_deg=10):
        B, D = Xb_flat.shape
        if side * side != D:
            return Xb_flat
        Xb = Xb_flat.reshape(B, side, side)
        out = np.empty_like(Xb)
        for i in range(B):
            img = Xb[i]
            dx = np.random.randint(-max_shift, max_shift + 1)
            dy = np.random.randint(-max_shift, max_shift + 1)
            shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            # with 50% probability apply rotation
            if max_rotate_deg > 0 and np.random.rand() < 0.5:
                angle = np.deg2rad(np.random.uniform(-max_rotate_deg, max_rotate_deg))
                out[i] = self._rotate_nearest(shifted, angle)
            else:
                out[i] = shifted
        return out.reshape(B, D)

    @staticmethod
    def _rotate_nearest(img, angle_rad):
        h, w = img.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        x0 = xx - cx
        y0 = yy - cy
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        xr = c * x0 + s * y0 + cx
        yr = -s * x0 + c * y0 + cy
        xi = np.rint(xr).astype(int)
        yi = np.rint(yr).astype(int)
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)
        return img[yi, xi]
