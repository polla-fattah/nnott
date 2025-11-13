import os, sys, numpy as np
import matplotlib.pyplot as plt

# Allow running this file directly (adds project root to sys.path)
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from common.data_utils import DataUtility
from vectorized.modules import Sequential, Linear, ReLU
from vectorized.optim import Adam, SGD
from vectorized.trainer import VTrainer


def main():
    # Load and flatten
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)

    # Build vectorized model (logits output)
    model = Sequential(
        Linear(28*28, 256, activation_hint='relu'),
        ReLU(),
        Linear(256, 128, activation_hint='relu'),
        ReLU(),
        Linear(128, 10, activation_hint=None),
    )

    # Optimizer: Adam default
    optim = Adam(lr=1e-3, weight_decay=1e-4)
    trainer = VTrainer(model, optim, num_classes=10)

    # Train
    trainer.train(X_train, y_train, epochs=2, batch_size=32, verbose=True)
    plot_loss(trainer.loss_history)

    # Evaluate
    trainer.evaluate(X_test, y_test)
    imgs, preds, trues, total = trainer.misclassification_data(X_test, y_test, max_images=100)
    if total:
        plot_misclassifications(imgs, preds, trues, total)


def plot_loss(loss_history):
    if not loss_history:
        return
    epochs = range(1, len(loss_history) + 1)
    plt.figure()
    plt.plot(epochs, loss_history, marker="o")
    plt.title("Vectorized Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_misclassifications(imgs, preds, trues, total, cols=5):
    n = len(imgs)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    side = int(np.sqrt(imgs.shape[1])) if imgs.size else 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, true, pred in zip(axes, imgs, trues, preds):
        disp = img
        if side and side * side == img.size:
            disp = img.reshape(side, side)
        imin, imax = float(disp.min()), float(disp.max())
        if imax > imin:
            disp = (disp - imin) / (imax - imin)
        ax.imshow(disp, cmap="gray")
        ax.set_title(f"T:{int(true)} P:{int(pred)}")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Misclassifications: showing {n} of {total}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
