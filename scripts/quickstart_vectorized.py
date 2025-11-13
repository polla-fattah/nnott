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

from common.data_utils import DataUtility
from vectorized.modules import Sequential, Linear, ReLU
from vectorized.optim import Adam, SGD
from vectorized.trainer import VTrainer
from vectorized.main import plot_loss, plot_misclassifications


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
        default="256,128",
        help="Comma-separated hidden sizes for the basic scenario.",
    )
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
    model = build_model(args.hidden_sizes)
    optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
    trainer = VTrainer(model, optim, num_classes=10)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
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
        model = build_model(args.hidden_sizes)
        if name == "sgd":
            optim = SGD(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        else:
            optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
        trainer = VTrainer(model, optim, num_classes=10)
        trainer.train(
            data.X_train,
            data.y_train,
            epochs=max(1, args.epochs // 2),
            batch_size=args.batch_size,
            verbose=True,
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
        model = build_model(",".join(str(c) for c in cfg))
        trainer = VTrainer(model, Adam(lr=args.lr, weight_decay=args.weight_decay), num_classes=10)
        trainer.train(
            data.X_train,
            data.y_train,
            epochs=max(1, args.epochs // len(configs)),
            batch_size=args.batch_size,
            verbose=True,
        )
        trainer.evaluate(data.X_test, data.y_test)
        if args.plot:
            plot_loss(trainer.loss_history)


def build_model(hidden_sizes_str: str) -> Sequential:
    hidden = tuple(int(h.strip()) for h in hidden_sizes_str.split(",") if h.strip())
    if not hidden:
        hidden = (256, 128)
    layers = []
    in_dim = 28 * 28
    for h in hidden:
        layers.append(Linear(in_dim, h, activation_hint="relu"))
        layers.append(ReLU())
        in_dim = h
    layers.append(Linear(in_dim, 10, activation_hint=None))
    return Sequential(*layers)


class DatasetBundle:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


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
    return DatasetBundle(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
