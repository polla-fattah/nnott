import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
# Allow running this file directly
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
from scalar.network import Network
from scalar.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate scalar MLP on MNIST.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plots for samples, loss, and predictions.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load the data
    data_util = DataUtility(data_dir="data")
    X_train, y_train, X_test, y_test = data_util.load_data()

    print("Train images:", X_train.shape)   # (60000, 28, 28)
    print("Train labels:", y_train.shape)   # (60000,)
    print("Test images:", X_test.shape)     # (10000, 28, 28)
    print("Test labels:", y_test.shape)     # (10000,)

    # 1b. Show some of the training data visually
    sample_pairs = DataUtility.sample_images(X_train, y_train, num_samples=10)
    if sample_pairs and not args.no_plot:
        plot_image_grid(sample_pairs, title="Training samples")

    # 2. Create network & trainer
    # for MNIST-like data: 28x28, 10 classes
    num_classes = 10  # labels 0–9

    network = Network(
        input_size=28 * 28,       # explicit, matches (28,28)
        num_classes=num_classes,
        hidden_sizes=(128, 64),
        learning_rate=0.01,
        activation='relu' #sigmoid
    )

    trainer = Trainer(network, num_classes=num_classes)

    # 2. Start training (tip: start with 1–2 epochs to make it faster)
    trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    if not args.no_plot:
        plot_loss(trainer.loss_history)

    # 3. Test the result (evaluate)
    trainer.evaluate(X_test, y_test)

    # 3b. Show 10 random test images with true/pred labels
    pred_samples = trainer.get_random_predictions(X_test, y_test, num_samples=10)
    if pred_samples and not args.no_plot:
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


if __name__ == "__main__":
    main()
