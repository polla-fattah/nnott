import time
import numpy as _np
from tqdm import tqdm

import common.backend as backend
from common.cross_entropy import CrossEntropyLoss
from common.model_io import save_model, load_model as load_model_state
from common.data_utils import ensure_label_format
from common.augment import augment_image_batch, build_augment_config as shared_augment_config

xp = backend.xp


class ConvTrainer:
    def __init__(self, model, optimizer, num_classes=10, grad_clip_norm=None, augment_config=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.criterion_per_sample = CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes
        self.loss_history = []
        self.grad_clip_norm = grad_clip_norm
        self.augment_cfg = self._build_augment_cfg(augment_config)

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
        clean_y = ensure_label_format(y_train, self.num_classes)
        X = backend.to_device(X_train, dtype=xp.float32)
        y = backend.to_device(clean_y, dtype=xp.int64)
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
                mix_meta = None
                if augment:
                    xb, mix_meta = augment_image_batch(
                        xb,
                        self.augment_cfg,
                        xp_module=xp,
                        labels=yb,
                        allow_label_mix=True,
                    )
                self.model.zero_grad()
                logits = self.model.forward(xb)
                loss, grad = self._compute_loss_and_grad(logits, yb, mix_meta)
                running += float(loss) * len(xb)
                self.model.backward(grad)
                self._clip_gradients(self.model.parameters())
                self.optimizer.step(self.model.parameters(), batch_size=len(xb))

            avg = running / n
            self.loss_history.append(float(avg))
            if verbose:
                print(f"Epoch {epoch}: avg loss {avg:.5f}")

        if verbose:
            print(f"\nTotal training time: {time.time() - t0:.2f}s")

    def evaluate(self, X_test, y_test, batch_size=256):
        if hasattr(self.model, "eval"):
            self.model.eval()
        clean_y = ensure_label_format(y_test, self.num_classes)
        X = backend.to_device(X_test, dtype=xp.float32)
        y = backend.to_device(clean_y, dtype=xp.int64)
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

    def collect_misclassifications(self, X_test, y_test, max_images=25, batch_size=512):
        if hasattr(self.model, "eval"):
            self.model.eval()
        n = len(X_test)
        if n == 0:
            return _np.empty((0,)), _np.empty((0,)), _np.empty((0,)), 0

        preds = xp.empty(n, dtype=xp.int64)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = backend.to_device(X_test[start:end], dtype=xp.float32)
            logits = self.model.forward(xb)
            preds[start:end] = xp.argmax(logits, axis=1)

        y_all = backend.to_device(ensure_label_format(y_test, self.num_classes), dtype=xp.int64)
        mis_idx = xp.where(preds != y_all)[0]
        mis_idx_cpu = backend.to_cpu(mis_idx).astype(_np.int64, copy=False)
        total = len(mis_idx_cpu)
        if total == 0:
            return _np.empty((0,)), _np.empty((0,)), _np.empty((0,)), 0
        mis_idx_cpu = mis_idx_cpu[:max_images]
        imgs = _np.asarray(X_test)[mis_idx_cpu]
        preds_cpu_all = backend.to_cpu(preds)
        y_cpu_all = backend.to_cpu(y_all)
        preds_cpu = preds_cpu_all[mis_idx_cpu]
        y_cpu = y_cpu_all[mis_idx_cpu]
        return imgs, preds_cpu, y_cpu, total

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

    def _compute_loss_and_grad(self, logits, targets, mix_meta):
        if not mix_meta:
            loss = self.criterion.forward(logits, targets)
            grad = self.criterion.backward(logits, targets)
            return loss, grad
        lam = mix_meta["lam"].reshape(-1, 1)
        ta = mix_meta["targets_a"]
        tb = mix_meta["targets_b"]
        loss_a = self.criterion_per_sample.forward(logits, ta)
        loss_b = self.criterion_per_sample.forward(logits, tb)
        combined = lam[:, 0] * loss_a + (1.0 - lam[:, 0]) * loss_b
        loss = float(backend.to_cpu(combined.mean()))
        grad_a = self.criterion.backward(logits, ta)
        grad_b = self.criterion.backward(logits, tb)
        grad = lam * grad_a + (1.0 - lam) * grad_b
        return loss, grad

    def _clip_gradients(self, params):
        max_norm = self.grad_clip_norm
        if not max_norm or max_norm <= 0:
            return
        total = xp.asarray(0.0, dtype=xp.float32)
        for _, g in params:
            if g is None:
                continue
            total += xp.sum(g.astype(xp.float32) ** 2)
        norm = xp.sqrt(total)
        norm_val = float(backend.to_cpu(norm))
        if norm_val <= max_norm:
            return
        scale = max_norm / (norm_val + 1e-8)
        for _, g in params:
            if g is not None:
                g *= scale

    def _build_augment_cfg(self, config):
        if config:
            return shared_augment_config(config)
        return shared_augment_config(None)
