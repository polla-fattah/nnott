import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# Allow running this file directly
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
from common.augment import build_augment_config
from common.seed import set_global_seed
from common.metrics import confusion_matrix, format_confusion_matrix
from scalar.network import Network
from scalar.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate scalar MLP on MNIST.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden sizes (e.g., 256,128,64).",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "leaky_relu", "sigmoid", "tanh", "gelu"],
        default="relu",
        help="Default hidden activation when --hidden-activations is omitted.",
    )
    parser.add_argument(
        "--hidden-activations",
        type=str,
        default=None,
        help="Comma-separated activation per hidden layer (e.g., relu,tanh,sigmoid).",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=str,
        default="0.2",
        help="Dropout probability per hidden layer (single value or comma list). Use 0 to disable.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of training data used for validation (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility.")
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print confusion matrix after test evaluation.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentations.",
    )
    parser.add_argument("--augment-max-shift", type=int, default=2, help="Pixel shift radius for jitter augmentation.")
    parser.add_argument("--augment-rotate-deg", type=float, default=10.0, help="Max rotation degrees (0 disables).")
    parser.add_argument("--augment-rotate-prob", type=float, default=0.5, help="Probability of applying rotation.")
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5, help="Probability of horizontal flip.")
    parser.add_argument("--augment-vflip-prob", type=float, default=0.0, help="Probability of vertical flip.")
    parser.add_argument("--augment-noise-std", type=float, default=0.02, help="Stddev for Gaussian noise (0 disables).")
    parser.add_argument("--augment-noise-prob", type=float, default=0.3, help="Probability of injecting noise.")
    parser.add_argument("--augment-noise-clip", type=float, default=3.0, help="Clamp magnitude after noise.")
    parser.add_argument("--augment-cutout-prob", type=float, default=0.0, help="Probability of applying cutout masks.")
    parser.add_argument("--augment-cutout-size", type=int, default=4, help="Cutout square size.")
    parser.add_argument("--augment-randaug-layers", type=int, default=0, help="RandAugment layers (0 disables).")
    parser.add_argument("--augment-randaug-magnitude", type=float, default=0.0, help="RandAugment magnitude (0-1).")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable plotting (sample grid, loss curve, prediction grid). These add extra passes and can slow runs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    # 1. Load the data
    data_util = DataUtility(data_dir="data")
    X_train, y_train, X_test, y_test = data_util.load_data()
    val_split = min(max(args.val_split, 0.0), 0.4)
    X_train, y_train, X_val, y_val = DataUtility.train_val_split(
        X_train, y_train, val_fraction=val_split, seed=args.seed
    )

    print("Train images:", X_train.shape)   # (60000, 28, 28)
    print("Train labels:", y_train.shape)   # (60000,)
    print("Test images:", X_test.shape)     # (10000, 28, 28)
    print("Test labels:", y_test.shape)     # (10000,)
    if X_val is not None:
        print("Validation images:", X_val.shape)

    # 1b. Show some of the training data visually
    sample_pairs = DataUtility.sample_images(X_train, y_train, num_samples=10)
    if sample_pairs and args.plot:
        plot_image_grid(sample_pairs, title="Training samples")

    # 2. Create network & trainer
    # for MNIST-like data: 28x28, 10 classes
    num_classes = 10  # labels 0-9

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes, default=(256, 128, 64))
    hidden_acts = parse_hidden_activations(args.hidden_activations, len(hidden_sizes), default=args.activation)
    dropout_values = parse_dropout(args.hidden_dropout, len(hidden_sizes))

    network = Network(
        input_size=28 * 28,
        num_classes=num_classes,
        hidden_sizes=hidden_sizes,
        learning_rate=0.01,
        activation=args.activation,
        hidden_activations=hidden_acts,
        hidden_dropout=dropout_values,
    )

    augment_config = build_augment_config(
        max_shift=args.augment_max_shift,
        rotate_deg=args.augment_rotate_deg,
        rotate_prob=args.augment_rotate_prob,
        hflip_prob=args.augment_hflip_prob,
        vflip_prob=args.augment_vflip_prob,
        noise_std=args.augment_noise_std,
        noise_prob=args.augment_noise_prob,
        noise_clip=args.augment_noise_clip,
        cutout_prob=args.augment_cutout_prob,
        cutout_size=args.augment_cutout_size,
        randaugment_layers=args.augment_randaug_layers,
        randaugment_magnitude=args.augment_randaug_magnitude,
        cutmix_prob=0.0,  # scalar trainer keeps labels single-hot
    )

    trainer = Trainer(network, num_classes=num_classes, augment_config=augment_config)

    # 2. Start training (tip: start with 1–2 epochs to make it faster)
    trainer.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        val_data=(X_val, y_val) if X_val is not None else None,
        augment=not args.no_augment,
    )
    if args.plot:
        plot_loss(trainer.loss_history)

    # 3. Test the result (evaluate)
    if args.confusion_matrix:
        acc, preds, targets = trainer.evaluate(X_test, y_test, return_preds=True)
        cm = confusion_matrix(preds, targets, num_classes)
        print("Confusion Matrix:\n" + format_confusion_matrix(cm))
    else:
        trainer.evaluate(X_test, y_test)

    # 3b. Show 10 random test images with true/pred labels
    pred_samples = trainer.get_random_predictions(X_test, y_test, num_samples=10)
    if pred_samples and args.plot:
        plot_prediction_grid(pred_samples, title="Random Test Predictions")


def plot_image_grid(samples, title, cols=5):
    cols = max(1, min(cols, len(samples)))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for ax, (img, label) in zip(axes, samples):
        ax.imshow(img, cmap="gray")
        ax.set_title(str(label))
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_loss(loss_history):
    if not loss_history:
        return
    epochs = range(1, len(loss_history) + 1)
    plt.figure()
    plt.plot(epochs, loss_history, marker="o")
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_prediction_grid(samples, title, cols=5):
    cols = max(1, min(cols, len(samples)))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()
    for ax, (img, true_label, pred_label) in zip(axes, samples):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"T:{true_label} P:{pred_label}")
        ax.axis("off")
    for ax in axes[len(samples):]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def parse_hidden_sizes(spec, default):
    values = tuple(int(h.strip()) for h in (spec or "").split(",") if h.strip())
    return values or tuple(default)


def parse_hidden_activations(spec, length, default):
    if spec:
        acts = [a.strip().lower() for a in spec.split(",") if a.strip()]
    else:
        acts = []
    if not acts:
        acts = [default] * length
    if len(acts) == 1 and length > 1:
        acts = acts * length
    if len(acts) != length:
        raise ValueError("hidden_activations must match hidden layer count.")
    return acts


def parse_dropout(spec, length):
    if not spec:
        return [0.0] * length
    vals = []
    for token in (s.strip().lower() for s in spec.split(",") if s.strip()):
        if token in {"none", "off"}:
            vals.append(0.0)
        else:
            vals.append(max(0.0, min(0.95, float(token))))
    if len(vals) == 1 and length > 1:
        vals = vals * length
    if len(vals) != length:
        raise ValueError("hidden-dropout must have one value or match hidden layer count.")
    return vals


if __name__ == "__main__":
    main()
