import numpy as np
from tqdm import tqdm
import time
from math import cos, pi
from common.cross_entropy import CrossEntropyLoss
from common.model_io import save_model, load_model as load_model_state
from common.data_utils import ensure_label_format


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
                if augment:
                    logits = self.model.forward(self._augment_batch(Xb))
                else:
                    logits = self.model.forward(Xb)
                loss = self.criterion.forward(logits, yb)
                total += float(loss) * len(Xb)
                grad_logits = self.criterion.backward(logits, yb)
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
    def _augment_batch(self, Xb_flat, side=28):
        cfg = self.augment_cfg
        B, D = Xb_flat.shape
        if side * side != D:
            return Xb_flat
        Xb = Xb_flat.reshape(B, side, side)
        out = np.empty_like(Xb)
        for i in range(B):
            img = Xb[i]
            if cfg["max_shift"] > 0:
                dx = np.random.randint(-cfg["max_shift"], cfg["max_shift"] + 1)
                dy = np.random.randint(-cfg["max_shift"], cfg["max_shift"] + 1)
                img = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
            if cfg["rotate_deg"] > 0 and np.random.rand() < cfg["rotate_prob"]:
                angle = np.deg2rad(np.random.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]))
                img = self._rotate_nearest(img, angle)
            if cfg["hflip_prob"] > 0 and np.random.rand() < cfg["hflip_prob"]:
                img = np.flip(img, axis=1)
            if cfg["vflip_prob"] > 0 and np.random.rand() < cfg["vflip_prob"]:
                img = np.flip(img, axis=0)
            if cfg["noise_std"] > 0 and np.random.rand() < cfg["noise_prob"]:
                noise = np.random.normal(0.0, cfg["noise_std"], size=img.shape).astype(np.float32)
                img = img + noise
            if cfg["noise_clip"] > 0:
                img = np.clip(img, -cfg["noise_clip"], cfg["noise_clip"])
            out[i] = img
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
        cfg = {
            "max_shift": 2,
            "rotate_deg": 10.0,
            "rotate_prob": 0.5,
            "hflip_prob": 0.5,
            "vflip_prob": 0.0,
            "noise_std": 0.02,
            "noise_prob": 0.3,
            "noise_clip": 3.0,
        }
        if config:
            for key in cfg:
                if key in config and config[key] is not None:
                    cfg[key] = config[key]
        cfg["max_shift"] = max(0, int(cfg["max_shift"]))
        cfg["rotate_deg"] = max(0.0, float(cfg["rotate_deg"]))
        cfg["rotate_prob"] = float(np.clip(cfg["rotate_prob"], 0.0, 1.0))
        cfg["hflip_prob"] = float(np.clip(cfg["hflip_prob"], 0.0, 1.0))
        cfg["vflip_prob"] = float(np.clip(cfg["vflip_prob"], 0.0, 1.0))
        cfg["noise_std"] = max(0.0, float(cfg["noise_std"]))
        cfg["noise_prob"] = float(np.clip(cfg["noise_prob"], 0.0, 1.0))
        cfg["noise_clip"] = max(0.0, float(cfg["noise_clip"]))
        return cfg

    def _prepare_labels(self, labels):
        return ensure_label_format(labels, self.num_classes)
