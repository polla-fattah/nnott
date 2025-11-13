"""
Quick-start scenarios for the scalar (loop-based) MLP implementation.

Run any of the predefined scenarios without editing source files:

    python scripts/quickstart_scalar.py --scenario basic --plot
    python scripts/quickstart_scalar.py --scenario optimizer-compare
    python scripts/quickstart_scalar.py --scenario dataset-swap \
        --alt-train-images fashion_train_images.npy \
        --alt-train-labels fashion_train_labels.npy \
        --alt-test-images fashion_test_images.npy \
        --alt-test-labels fashion_test_labels.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.data_utils import DataUtility
from scalar.network import Network
from scalar.trainer import Trainer
from scalar.main import (
    plot_image_grid,
    plot_loss,
    plot_prediction_grid,
    parse_hidden_sizes,
    parse_hidden_activations,
    parse_dropout,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scalar MLP quick-start scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["basic", "optimizer-compare", "dataset-swap"],
        default="basic",
        help="Which walkthrough to run.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for the primary run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for the primary run.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden sizes (e.g., 256,64).",
    )
    parser.add_argument(
        "--activation",
        choices=["relu", "sigmoid", "tanh", "leaky_relu", "gelu"],
        default="relu",
        help="Hidden-layer activation.",
    )
    parser.add_argument(
        "--hidden-activations",
        type=str,
        default=None,
        help="Comma-separated activation list per hidden layer (e.g., relu,tanh).",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=str,
        default="0.2",
        help="Dropout value(s) per hidden layer (single value or comma list).",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing .npy datasets.")
    parser.add_argument("--train-images", default="train_images.npy")
    parser.add_argument("--train-labels", default="train_labels.npy")
    parser.add_argument("--test-images", default="test_images.npy")
    parser.add_argument("--test-labels", default="test_labels.npy")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show matplotlib plots. Note: prediction grids/misclassification checks run additional passes and can be slow.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of training samples to preview when plotting is enabled.",
    )
    # dataset swap overrides
    parser.add_argument("--alt-train-images", default="fashion_train_images.npy")
    parser.add_argument("--alt-train-labels", default="fashion_train_labels.npy")
    parser.add_argument("--alt-test-images", default="fashion_test_images.npy")
    parser.add_argument("--alt-test-labels", default="fashion_test_labels.npy")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scenario == "basic":
        run_basic(args)
    elif args.scenario == "optimizer-compare":
        run_optimizer_compare(args)
    elif args.scenario == "dataset-swap":
        run_dataset_swap(args)
    else:
        raise ValueError(f"Unknown scenario {args.scenario}")


def run_basic(args):
    print("== Basic scalar MLP training ==")
    data = load_dataset(
        args,
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels,
    )
    model = build_network(args)
    trainer = Trainer(model, num_classes=10, optimizer="adam", lr=args.lr)

    maybe_plot_samples(args, data)
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
    maybe_plot_predictions(args, trainer, data)


def run_optimizer_compare(args):
    print("== Optimizer comparison (SGD vs Adam) ==")
    data = load_dataset(
        args,
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels,
    )
    epochs = max(1, args.epochs // 2)
    results = {}
    for opt_name in ("sgd", "adam"):
        print(f"\n--- Training with {opt_name.upper()} ---")
        model = build_network(args)
        trainer = Trainer(model, num_classes=10, optimizer=opt_name, lr=args.lr)
        trainer.train(
            data.X_train,
            data.y_train,
            epochs=epochs,
            batch_size=args.batch_size,
            verbose=True,
        )
        acc = trainer.evaluate(data.X_test, data.y_test)
        results[opt_name] = acc
        if args.plot:
            plot_loss(trainer.loss_history)
    print("\nSummary:")
    for name, acc in results.items():
        print(f"{name.upper():>4}: {acc*100:.2f}% accuracy")


def run_dataset_swap(args):
    print("== Dataset swap demo ==")
    alt_files = (
        args.alt_train_images,
        args.alt_train_labels,
        args.alt_test_images,
        args.alt_test_labels,
    )
    for file in alt_files:
        full = Path(args.data_dir) / file
        if not full.exists():
            print(f"[WARN] Expected file {full} is missing. Convert your alternative dataset to .npy first.")
    data = load_dataset(args, *alt_files)
    model = build_network(args)
    trainer = Trainer(model, num_classes=10, optimizer="adam", lr=args.lr)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )
    trainer.evaluate(data.X_test, data.y_test)
    if args.plot:
        plot_loss(trainer.loss_history)
        maybe_plot_predictions(args, trainer, data)


def maybe_plot_samples(args, data):
    if not args.plot or args.sample_count <= 0:
        return
    samples = DataUtility.sample_images(data.X_train, data.y_train, num_samples=args.sample_count)
    if samples:
        plot_image_grid(samples, title="Training samples", cols=min(5, args.sample_count))


def maybe_plot_predictions(args, trainer, data):
    if not args.plot:
        return
    preds = trainer.get_random_predictions(data.X_test, data.y_test, num_samples=args.sample_count)
    if preds:
        plot_prediction_grid(preds, title="Random Test Predictions", cols=min(5, args.sample_count))


def build_network(args):
    hidden = parse_hidden_sizes(args.hidden_sizes, default=(256, 128, 64))
    activations = parse_hidden_activations(args.hidden_activations, len(hidden), default=args.activation)
    dropout_vals = parse_dropout(args.hidden_dropout, len(hidden))
    return Network(
        input_size=28 * 28,
        num_classes=10,
        hidden_sizes=hidden,
        learning_rate=args.lr,
        activation=args.activation,
        hidden_activations=activations,
        hidden_dropout=dropout_vals,
    )


class DatasetBundle:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def load_dataset(args, train_images, train_labels, test_images, test_labels):
    util = DataUtility(args.data_dir)
    X_train, y_train, X_test, y_test = util.load_data(
        train_images_file=train_images,
        train_labels_file=train_labels,
        test_images_file=test_images,
        test_labels_file=test_labels,
    )
    return DatasetBundle(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
