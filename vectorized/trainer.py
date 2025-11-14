import numpy as np
from tqdm import tqdm
import time
from math import cos, pi
from common.cross_entropy import CrossEntropyLoss
from common.model_io import save_model, load_model as load_model_state
from common.data_utils import ensure_label_format
from common.augment import augment_flat_batch, build_augment_config as shared_augment_config


class CosineScheduler:
    def __init__(self, initial_lr, min_lr, total_epochs):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = max(1, total_epochs)

    def step(self, epoch, *_):
        if self.initial_lr is None:
            return None
        progress = min(epoch, self.total_epochs) / self.total_epochs
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + cos(pi * progress))


class ReduceLROnPlateau:
    def __init__(self, initial_lr, factor=0.5, patience=3, min_lr=1e-5, min_delta=1e-4):
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = max(1, patience)
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.best = np.inf
        self.wait = 0
        self.current_lr = initial_lr

    def step(self, _, metric):
        if self.initial_lr is None or metric is None:
            return None
        if metric + self.min_delta < self.best:
            self.best = metric
            self.wait = 0
            return None
        self.wait += 1
        if self.wait >= self.patience:
            new_lr = max(self.min_lr, self.current_lr * self.factor)
            if new_lr < self.current_lr:
                self.current_lr = new_lr
                self.wait = 0
                return self.current_lr
        return None


class VTrainer:
    def __init__(
        self,
        model,
        optimizer,
        num_classes=10,
        grad_clip_norm=None,
        lr_scheduler_config=None,
        early_stopping_config=None,
        augment_config=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.criterion_per_sample = CrossEntropyLoss(reduction="none")
        self.num_classes = num_classes
        self.loss_history = []
        self.grad_clip_norm = grad_clip_norm
        self.scheduler = self._build_scheduler(lr_scheduler_config)
        self.early_cfg = early_stopping_config or {}
        self.early_best = np.inf
        self.early_wait = 0
        self.early_enabled = bool(self.early_cfg)
        self.augment_cfg = self._build_augment_cfg(augment_config)

    def train(
        self,
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=True,
        lr_schedule=True,
        augment=True,
        val_data=None,
    ):
        X = np.asarray(X_train, dtype=np.float32)
        y = self._prepare_labels(y_train)
        n = len(X)
        self.loss_history = []
        t0 = time.time()

        X_val, y_val = (None, None)
        if val_data:
            X_val, raw_val = val_data
            y_val = self._prepare_labels(raw_val)

        for epoch in range(1, epochs + 1):
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
                mix_meta = None
                if augment:
                    Xb, mix_meta = augment_flat_batch(
                        Xb,
                        self.augment_cfg,
                        xp_module=np,
                        labels=yb,
                        image_side=28,
                    )
                logits = self.model.forward(Xb)
                loss, grad_logits = self._compute_loss_and_grad(logits, yb, mix_meta)
                total += float(loss) * len(Xb)
                self.model.backward(grad_logits)
                self._clip_gradients(self.model.parameters())
                self.optimizer.step(self.model.parameters(), batch_size=len(Xb))

            avg = total / n
            self.loss_history.append(avg)
            val_loss = None
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate_loss(X_val, y_val, batch_size)
            self._maybe_adjust_lr(epoch, epochs, val_loss if val_loss is not None else avg)
            stopped = self._maybe_early_stop(val_loss)
            if verbose:
                msg = f"Epoch {epoch}/{epochs} - Avg Loss: {avg:.6f}"
                if val_loss is not None:
                    msg += f" | Val Loss: {val_loss:.6f}"
                if hasattr(self.optimizer, 'lr'):
                    msg += f" | LR: {self.optimizer.lr:.6g}"
                print(msg)
            if stopped:
                if verbose:
                    print(f"[EarlyStopping] Triggered after epoch {epoch}.")
                break

        if verbose:
            print(f"\nTotal training time: {time.time()-t0:.2f}s")

    def evaluate(self, X_test, y_test):
        if hasattr(self.model, 'eval'):
            self.model.eval()
        X = np.asarray(X_test, dtype=np.float32)
        y = self._prepare_labels(y_test)
        logits = self.model.forward(X)
        preds = np.argmax(logits, axis=1)
        acc = float((preds == y).mean())
        print(f"Test accuracy: {acc*100:.2f}%")
        return acc

    def misclassification_data(self, X_test, y_test, max_images=None):
        X = np.asarray(X_test, dtype=np.float32)
        y = self._prepare_labels(y_test)
        logits = self.model.forward(X)
        preds = np.argmax(logits, axis=1)
        mis_idx = np.where(preds != y)[0]
        total = len(mis_idx)
        if total == 0:
            return np.array([]), np.array([]), np.array([]), 0
        if max_images is not None:
            mis_idx = mis_idx[:max_images]
        return X[mis_idx], preds[mis_idx], y[mis_idx], total

    def save_model(self, path, metadata=None):
        meta = dict(metadata or {})
        meta.setdefault("num_classes", self.num_classes)
        save_model(self.model, path, meta)
        return meta

    def load_model(self, path):
        return load_model_state(self.model, path)

    # --- augmentation utilities ---
    def _compute_loss_and_grad(self, logits, targets, mix_meta):
        if not mix_meta:
            loss = self.criterion.forward(logits, targets)
            grad = self.criterion.backward(logits, targets)
            return loss, grad
        lam = np.asarray(mix_meta["lam"], dtype=np.float32).reshape(-1, 1)
        ta = mix_meta["targets_a"]
        tb = mix_meta["targets_b"]
        loss_a = self.criterion_per_sample.forward(logits, ta)
        loss_b = self.criterion_per_sample.forward(logits, tb)
        combined = lam[:, 0] * loss_a + (1.0 - lam[:, 0]) * loss_b
        loss = float(np.mean(combined))
        grad_a = self.criterion.backward(logits, ta)
        grad_b = self.criterion.backward(logits, tb)
        grad = lam * grad_a + (1.0 - lam) * grad_b
        return loss, grad

    def _clip_gradients(self, params):
        max_norm = self.grad_clip_norm
        if not max_norm or max_norm <= 0:
            return
        total = 0.0
        grads = []
        for _, g in params:
            if g is None:
                continue
            grads.append(g)
            total += float(np.sum(g.astype(np.float32) ** 2))
        norm = np.sqrt(total)
        if norm <= max_norm:
            return
        scale = max_norm / (norm + 1e-8)
        for g in grads:
            g *= scale

    # --- scheduler / early stopping helpers ---
    def _build_scheduler(self, config):
        if not config:
            return None
        sched_type = config.get("type")
        base_lr = getattr(self.optimizer, "lr", None)
        if sched_type == "cosine":
            return CosineScheduler(
                initial_lr=base_lr,
                min_lr=config.get("min_lr", 1e-5),
                total_epochs=config.get("total_epochs", 1),
            )
        if sched_type == "reduce_on_plateau":
            return ReduceLROnPlateau(
                initial_lr=base_lr,
                factor=config.get("factor", 0.5),
                patience=config.get("patience", 3),
                min_lr=config.get("min_lr", 1e-5),
                min_delta=config.get("min_delta", 1e-4),
            )
        return None

    def _maybe_adjust_lr(self, epoch, total_epochs, metric):
        if not self.scheduler or not hasattr(self.optimizer, "lr"):
            return
        if isinstance(self.scheduler, CosineScheduler):
            self.scheduler.total_epochs = total_epochs
        new_lr = self.scheduler.step(epoch, metric)
        if new_lr is not None:
            self.optimizer.lr = new_lr

    def _maybe_early_stop(self, val_loss):
        if not self.early_enabled or val_loss is None:
            return False
        min_delta = self.early_cfg.get("min_delta", 1e-4)
        patience = max(1, self.early_cfg.get("patience", 5))
        if val_loss + min_delta < self.early_best:
            self.early_best = val_loss
            self.early_wait = 0
            return False
        self.early_wait += 1
        return self.early_wait >= patience

    def _evaluate_loss(self, X, y, batch_size):
        if hasattr(self.model, 'eval'):
            self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        y = self._prepare_labels(y)
        n = len(X)
        total = 0.0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = X[start:end]
            yb = y[start:end]
            logits = self.model.forward(xb)
            loss = self.criterion.forward(logits, yb)
            total += float(loss) * len(xb)
        if hasattr(self.model, 'train'):
            self.model.train()
        return total / n

    def _build_augment_cfg(self, config):
        if config:
            return shared_augment_config(config)
        return shared_augment_config(None)

    def _prepare_labels(self, labels):
        return ensure_label_format(labels, self.num_classes)
