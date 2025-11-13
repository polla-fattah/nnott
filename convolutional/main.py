import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
import common.backend as backend
from convolutional.architectures import (
    LeNet,
    BaselineCNN,
    AlexNet,
    VGG16,
    ResNet18,
    EfficientNetLite0,
    ConvNeXtTiny,
)
from convolutional.trainer import ConvTrainer
from vectorized.optim import Adam, Lookahead


ARCH_REGISTRY = {
    "lenet": LeNet(),
    "baseline": BaselineCNN(),
    "alexnet": AlexNet(),
    "vgg16": VGG16(),
    "resnet18": ResNet18(),
    "efficientnet_lite0": EfficientNetLite0(),
    "convnext_tiny": ConvNeXtTiny(),
}

ARCH_CHOICES = tuple(sorted(ARCH_REGISTRY.keys()))
ARCH_HELP = "Architecture key ({0}).".format(", ".join(ARCH_CHOICES))


def build_cnn(name="baseline", num_classes=10):
    name = name.lower()
    if name not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}")
    builder = ARCH_REGISTRY[name]
    return builder.build(num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate convolutional architectures.")
    parser.add_argument(
        "arch",
        nargs="?",
        default="baseline",
        choices=ARCH_CHOICES,
        help=ARCH_HELP,
    )
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--save", type=str, help="Path to save trained weights.")
    parser.add_argument("--load", type=str, help="Load weights before training/evaluation.")
    parser.add_argument("--no-augment", action="store_true", help="Disable shift augmentation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training phase (use with --load).")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy GPU backend if available.")
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Clip global gradient norm to this value (disabled by default).",
    )
    parser.add_argument(
        "--lookahead",
        action="store_true",
        help="Wrap the base optimizer with Lookahead (k-step).",
    )
    parser.add_argument(
        "--lookahead-k",
        type=int,
        default=5,
        help="Lookahead sync interval (k steps).",
    )
    parser.add_argument(
        "--lookahead-alpha",
        type=float,
        default=0.5,
        help="Lookahead interpolation factor alpha.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable matplotlib plots (disabled by default). Visualization requires an extra inference sweep.",
    )
    parser.add_argument(
        "--show-misclassified",
        action="store_true",
        help="Collect misclassified samples after evaluation; adds a full pass over the test set.",
    )
    return parser.parse_args()


def main(opts=None):
    args = opts or parse_args()
    arch_name = args.arch.lower()
    if args.gpu:
        try:
            backend.use_gpu()
            print("GPU backend enabled via CuPy.")
        except RuntimeError as exc:
            print(f"[WARN] {exc} Falling back to CPU backend.")
            backend.use_cpu()
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    model = build_cnn(arch_name, num_classes=10)
    optim = Adam(lr=5e-4, weight_decay=1e-4)
    if args.lookahead:
        optim = Lookahead(optim, k=args.lookahead_k, alpha=args.lookahead_alpha)
    trainer = ConvTrainer(model, optim, num_classes=10, grad_clip_norm=args.grad_clip)

    if args.load:
        meta = trainer.load_model(args.load)
        msg = f"Loaded weights from {args.load}"
        if meta:
            msg += f" | metadata: {meta}"
        print(msg)

    if not args.skip_train:
        trainer.train(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
            augment=not args.no_augment,
        )
        if args.plot:
            plot_loss(trainer.loss_history)
    else:
        print("Skipping training as requested.")

    trainer.evaluate(X_test, y_test)
    if args.show_misclassified or args.plot:
        imgs, preds, trues, total = trainer.collect_misclassifications(X_test, y_test, max_images=25)
        if args.plot and total:
            if not confirm_heavy_step("collect and plot misclassifications"):
                return
            plot_misclassifications(imgs, preds, trues, total, cols=5)

    if args.save:
        metadata = {"arch": arch_name, "epochs": args.epochs}
        trainer.save_model(args.save, metadata=metadata)
        print(f"Saved weights to {args.save}")


def confirm_heavy_step(action):
    resp = input(f"\nAbout to {action}, which runs another full inference pass. Continue? [y/N]: ").strip().lower()
    return resp in {"y", "yes"}


def plot_loss(loss_history):
    if not loss_history:
        return
    epochs = range(1, len(loss_history) + 1)
    plt.figure()
    plt.plot(epochs, loss_history, marker="o")
    plt.title("Convolutional Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_misclassifications(imgs, preds, trues, total, cols=5):
    n = len(imgs)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, true, pred in zip(axes, imgs, trues, preds):
        disp = img
        if disp.ndim == 3 and disp.shape[0] == 1:
            disp = disp[0]
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
