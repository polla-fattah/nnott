import numpy as np
from common.data_utils import DataUtility, ensure_label_format
from tqdm import tqdm  # <- ALWAYS use tqdm now
import time  # at the top of trainer.py
from common.cross_entropy import CrossEntropyLoss
from scalar.optimizer import SGD, Adam
from common.augment import build_augment_config, augment_image_batch


class Trainer:
    def __init__(
        self,
        network,
        num_classes=10,
        optimizer="adam",
        lr=0.001,
        weight_decay=0.0,
        augment_config=None,
    ):
        self.network = network
        self.num_classes = num_classes
        self.loss_history = []
        self.train_history = []
        self.val_history = []
        self.criterion = CrossEntropyLoss(reduction="mean")
        self.augment_cfg = build_augment_config(augment_config)
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

    def train(self, X_train, y_train, epochs=5, batch_size=64, verbose=True, val_data=None, augment=True):
        y_train = ensure_label_format(y_train, self.num_classes)
        n = len(X_train)
        self.loss_history = []
        self.train_history = []
        self.val_history = []
        start_time = time.time()
        if hasattr(self.network, "train"):
            self.network.train()

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"\n=== Epoch {epoch}/{epochs} ===")

            # Shuffle data each epoch
            indices = np.random.permutation(n)
            X_shuf = X_train[indices]
            y_shuf = y_train[indices]

            total_loss = 0.0
            correct = 0
            batches = (n + batch_size - 1) // batch_size
            if verbose:
                print(f"Batches/epoch: {batches} | Batch size: {batch_size}")

            # iterate over mini-batches
            for start in tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}", unit="batch"):
                end = min(start + batch_size, n)
                bs = end - start
                batch_inputs = X_shuf[start:end]
                batch_labels = y_shuf[start:end]
                flat_inputs = batch_inputs.reshape(bs, -1).astype(np.float32, copy=False)
                if augment and batch_inputs.ndim == 3:
                    imgs = batch_inputs[:, None, :, :].astype(np.float32, copy=False)
                    aug_imgs, _ = augment_image_batch(
                        imgs,
                        self.augment_cfg,
                        xp_module=np,
                        labels=None,
                        allow_label_mix=False,
                    )
                    flat_inputs = aug_imgs.reshape(bs, -1)

                # zero accumulators
                if hasattr(self.network, "zero_grad"):
                    self.network.zero_grad()
                else:
                    for layer in self.network.layers:
                        layer.zero_grad()

                # accumulate over batch
                for idx in range(bs):
                    x = flat_inputs[idx]
                    y = batch_labels[idx]

                    logits = self.network.forward(x)
                    loss = self.criterion.forward(logits, int(y))
                    total_loss += loss

                    pred_label = int(np.argmax(logits))
                    if pred_label == int(y):
                        correct += 1

                    grad_logits = self.criterion.backward(logits, int(y))
                    self.network.backward(grad_logits)

                # optimizer step with average gradients
                self.optimizer.step(self.network.layers, batch_size=bs)

            avg_loss = total_loss / n
            train_acc = correct / n
            self.loss_history.append(avg_loss)
            self.train_history.append({"loss": avg_loss, "acc": train_acc})

            if verbose:
                print(f"Epoch {epoch}/{epochs} - Train Loss: {avg_loss:.6f} | Train Acc: {train_acc*100:.2f}%")

            if val_data and val_data[0] is not None:
                val_loss, val_acc = self._evaluate_metrics(val_data[0], val_data[1])
                self.val_history.append({"loss": val_loss, "acc": val_acc})
                if verbose:
                    print(f"                 Val Loss: {val_loss:.6f} | Val Acc: {val_acc*100:.2f}%")

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\nTotal training time: {elapsed:.2f} seconds "
              f"({elapsed / epochs:.2f} sec/epoch)")

    def evaluate(self, X_test, y_test, return_preds=False):
        metrics = self._evaluate_metrics(X_test, y_test, collect_preds=return_preds)
        if return_preds:
            (loss, acc), preds, targets = metrics
        else:
            loss, acc = metrics
            preds = targets = None
        print(f"Test loss: {loss:.6f} | Test accuracy: {acc * 100:.2f}%")
        if return_preds:
            return acc, preds, targets
        return acc

    def get_random_predictions(self, X_test, y_test, num_samples=10):
        y_test = ensure_label_format(y_test, self.num_classes)
        num_samples = min(num_samples, len(X_test))
        if num_samples == 0:
            return []
        prev_mode = getattr(self.network, "training", True)
        if hasattr(self.network, "eval"):
            self.network.eval()
        idxs = np.random.choice(len(X_test), size=num_samples, replace=False)
        samples = []
        for idx in idxs:
            img = DataUtility._to_image(X_test[idx])
            true_label = int(y_test[idx])
            pred_label = self.network.predict(X_test[idx])
            samples.append((img, true_label, pred_label))
        if prev_mode:
            if hasattr(self.network, "train"):
                self.network.train()
        return samples

    def _evaluate_metrics(self, X, y, collect_preds=False):
        if X is None or y is None or len(X) == 0:
            return (0.0, 0.0) if not collect_preds else ((0.0, 0.0), np.array([]), np.array([]))
        y_clean = ensure_label_format(y, self.num_classes)
        prev_mode = getattr(self.network, "training", True)
        if hasattr(self.network, "eval"):
            self.network.eval()
        total = 0.0
        correct = 0
        preds = []
        for x, label in zip(X, y_clean):
            logits = self.network.forward(x)
            total += self.criterion.forward(logits, int(label))
            pred_label = int(np.argmax(logits))
            if pred_label == int(label):
                correct += 1
            if collect_preds:
                preds.append(pred_label)
        if prev_mode:
            if hasattr(self.network, "train"):
                self.network.train()
        loss = total / len(X)
        acc = correct / len(X)
        if collect_preds:
            return (loss, acc), np.asarray(preds, dtype=np.int64), np.asarray(y_clean, dtype=np.int64)
        return loss, acc
