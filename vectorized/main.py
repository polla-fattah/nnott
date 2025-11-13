import os, sys, numpy as np
import argparse
import matplotlib.pyplot as plt

# Allow running this file directly (adds project root to sys.path)
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from common.data_utils import DataUtility
from vectorized.modules import Sequential, Linear, ReLU
from vectorized.optim import Adam, SGD
from vectorized.trainer import VTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the vectorized MLP.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,128",
        help="Comma-separated hidden layer sizes (e.g., 256,128,64).",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay value.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable matplotlib plots (disabled by default). Collecting misclassifications requires an extra full pass and can be slow.",
    )
    return parser.parse_args()


def main(opts=None):
    args = opts or parse_args()

    # Load and flatten
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)

    hidden_sizes = tuple(
        int(h) for h in args.hidden_sizes.split(",") if h.strip()
    ) or (256, 128)

    layers = []
    in_dim = 28 * 28
    for h in hidden_sizes:
        layers.append(Linear(in_dim, h, activation_hint="relu"))
        layers.append(ReLU())
        in_dim = h
    layers.append(Linear(in_dim, 10, activation_hint=None))
    model = Sequential(*layers)

    if args.optimizer == "sgd":
        optim = SGD(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
    trainer = VTrainer(model, optim, num_classes=10)

    trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    if args.plot:
        plot_loss(trainer.loss_history)

    trainer.evaluate(X_test, y_test)
    imgs, preds, trues, total = trainer.misclassification_data(X_test, y_test, max_images=100)
    if args.plot and total:
        if confirm_heavy_step("collect and plot misclassifications"):
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


def confirm_heavy_step(task):
    response = input(f"\nAbout to {task}, which requires an extra pass and may take time. Continue? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


if __name__ == "__main__":
    main()
