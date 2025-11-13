"""
Quick-start scenarios for the vectorized (NumPy) MLP implementation.

Examples:

    python scripts/quickstart_vectorized.py --scenario basic --plot
    python scripts/quickstart_vectorized.py --scenario optimizer-compare --epochs 3
    python scripts/quickstart_vectorized.py --scenario hidden-sweep --hidden-options "512,256;256,128;128,64"
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data_utils import DataUtility

from vectorized.modules import ACTIVATION_KINDS
from vectorized.optim import Adam, SGD
from vectorized.trainer import VTrainer
from vectorized.main import (
    plot_loss,
    plot_misclassifications,
    parse_hidden_sizes,
    parse_activation_list,
    parse_dropout_list,
    build_mlp,
    build_scheduler_config,
    build_early_config,
    build_augment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Vectorized MLP quick-start scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["basic", "optimizer-compare", "hidden-sweep"],
        default="basic",
        help="Select which walkthrough to run.",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden sizes for the basic scenario.",
    )
    parser.add_argument(
        "--activation",
        choices=sorted(k for k in ACTIVATION_KINDS if k != "linear"),
        default="relu",
        help="Default hidden-layer activation.",
    )
    parser.add_argument(
        "--hidden-activations",
        type=str,
        default=None,
        help="Comma-separated activation list per hidden layer.",
    )
    parser.add_argument(
        "--dropout",
        type=str,
        default="0.2",
        help="Dropout value(s) per hidden layer (single value or comma-separated).",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation during training.",
    )
    parser.add_argument("--augment-max-shift", type=int, default=2, help="Pixel shift radius for jitter augmentation.")
    parser.add_argument("--augment-rotate-deg", type=float, default=10.0, help="Max rotation degrees; 0 disables.")
    parser.add_argument("--augment-rotate-prob", type=float, default=0.5, help="Probability of applying rotation.")
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5, help="Probability of horizontal flip.")
    parser.add_argument("--augment-vflip-prob", type=float, default=0.0, help="Probability of vertical flip.")
    parser.add_argument("--augment-noise-std", type=float, default=0.02, help="Stddev for Gaussian noise (0 disables).")
    parser.add_argument("--augment-noise-prob", type=float, default=0.3, help="Probability of adding noise.")
    parser.add_argument("--augment-noise-clip", type=float, default=3.0, help="Clamp magnitude after noise/flips.")
    parser.add_argument(
        "--batchnorm",
        action="store_true",
        help="Insert BatchNorm1D layers before each activation.",
    )
    parser.add_argument("--bn-momentum", type=float, default=0.1, help="BatchNorm momentum.")
    parser.add_argument(
        "--leaky-negative-slope",
        type=float,
        default=0.01,
        help="Negative slope used when LeakyReLU appears in the activation list.",
    )
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of training data for validation.")
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine", "reduce_on_plateau"],
        default="none",
        help="Learning-rate schedule to apply.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate for schedulers.")
    parser.add_argument("--reduce-factor", type=float, default=0.5, help="LR factor for ReduceLROnPlateau.")
    parser.add_argument("--reduce-patience", type=int, default=3, help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--reduce-delta", type=float, default=1e-4, help="Min improvement for ReduceLROnPlateau.")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping on validation loss.",
    )
    parser.add_argument("--early-patience", type=int, default=5, help="Early stopping patience.")
    parser.add_argument("--early-delta", type=float, default=1e-4, help="Minimum validation improvement.")
    parser.add_argument(
        "--hidden-options",
        type=str,
        default="512,256;256,128;128,64",
        help="Semicolon-separated hidden-size options for the sweep scenario.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing .npy datasets.")
    parser.add_argument("--train-images", default="train_images.npy")
    parser.add_argument("--train-labels", default="train_labels.npy")
    parser.add_argument("--test-images", default="test_images.npy")
    parser.add_argument("--test-labels", default="test_labels.npy")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display Matplotlib plots. Collecting misclassifications requires an extra pass and can be slow.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_dataset(args)
    if args.scenario == "basic":
        run_basic(args, data)
    elif args.scenario == "optimizer-compare":
        run_optimizer_compare(args, data)
    elif args.scenario == "hidden-sweep":
        run_hidden_sweep(args, data)
    else:
        raise ValueError(f"Unknown scenario {args.scenario}")


def run_basic(args, data):
    print("== Basic vectorized MLP training ==")
    model = build_model(args)
    optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
    trainer = build_trainer(args, model, optim)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        val_data=data.val_tuple,
        augment=not args.no_augment,
    )
    if args.plot:
        plot_loss(trainer.loss_history)
    trainer.evaluate(data.X_test, data.y_test)
    imgs, preds, trues, total = trainer.misclassification_data(data.X_test, data.y_test, max_images=50)
    if args.plot and total:
        plot_misclassifications(imgs, preds, trues, total)


def run_optimizer_compare(args, data):
    print("== Optimizer comparison (SGD vs Adam) ==")
    results = {}
    for name in ("sgd", "adam"):
        print(f"\n--- Training with {name.upper()} ---")
        model = build_model(args)
        if name == "sgd":
            optim = SGD(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
        trainer = build_trainer(args, model, optim)
        trainer.train(
            data.X_train,
            data.y_train,
            epochs=max(1, args.epochs // 2),
            batch_size=args.batch_size,
            verbose=True,
            val_data=data.val_tuple,
            augment=not args.no_augment,
        )
        acc = trainer.evaluate(data.X_test, data.y_test)
        results[name] = acc
        if args.plot:
            plot_loss(trainer.loss_history)
    print("\nSummary:")
    for name, acc in results.items():
        print(f"{name.upper():>4}: {acc*100:.2f}% accuracy")


def run_hidden_sweep(args, data):
    print("== Hidden-layer sweep ==")
    configs = []
    for block in args.hidden_options.split(";"):
        clean = tuple(int(h.strip()) for h in block.split(",") if h.strip())
        if clean:
            configs.append(clean)
    if not configs:
        configs = [(512, 256), (256, 128), (128, 64)]
    for cfg in configs:
        print(f"\n--- Hidden sizes: {cfg} ---")
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.hidden_sizes = ",".join(str(c) for c in cfg)
        model = build_model(sweep_args)
        trainer = build_trainer(args, model, Adam(lr=args.lr, weight_decay=args.weight_decay))
        trainer.train(
            data.X_train,
            data.y_train,
            epochs=max(1, args.epochs),
            batch_size=args.batch_size,
            verbose=True,
            val_data=data.val_tuple,
            augment=not args.no_augment,
        )
        trainer.evaluate(data.X_test, data.y_test)
        if args.plot:
            plot_loss(trainer.loss_history)


def build_model(args) -> 'Sequential':
    hidden = parse_hidden_sizes(args.hidden_sizes, default=(256, 128, 64))
    hidden_acts = parse_activation_list(args.hidden_activations, len(hidden), default_act=args.activation)
    dropout_list = parse_dropout_list(args.dropout, len(hidden))
    return build_mlp(
        hidden_sizes=hidden,
        hidden_activations=hidden_acts,
        dropout_list=dropout_list,
        use_batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        negative_slope=args.leaky_negative_slope,
    )


class DatasetBundle:
    def __init__(self, X_train, y_train, X_test, y_test, val_tuple=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.val_tuple = val_tuple


def load_dataset(args):
    util = DataUtility(args.data_dir)
    X_train, y_train, X_test, y_test = util.load_data(
        train_images_file=args.train_images,
        train_labels_file=args.train_labels,
        test_images_file=args.test_images,
        test_labels_file=args.test_labels,
    )
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)
    val_split = min(max(args.val_split, 0.0), 0.4)
    val_tuple = None
    if val_split > 0.0:
        n_val = max(1, int(len(X_train) * val_split))
        val_tuple = (X_train[-n_val:], y_train[-n_val:])
        X_train = X_train[:-n_val]
        y_train = y_train[:-n_val]
        print(f"[Data] Validation split: {n_val} samples ({val_split*100:.1f}%).")
    return DatasetBundle(X_train, y_train, X_test, y_test, val_tuple=val_tuple)


def build_trainer(args, model, optim):
    scheduler = build_scheduler_config(args)
    early = build_early_config(args)
    augment = build_augment_config(args)
    return VTrainer(
        model,
        optim,
        num_classes=10,
        lr_scheduler_config=scheduler,
        early_stopping_config=early,
        augment_config=augment,
    )


if __name__ == "__main__":
    main()
