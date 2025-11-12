import os
import sys
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
from convolutional.modules import (
    Sequential,
    Conv2D,
    BatchNorm2D,
    ReLU,
    LeakyReLU,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
)
from convolutional.trainer import ConvTrainer
from vectorized.optim import Adam


def build_cnn():
    return Sequential(
        Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
        BatchNorm2D(32),
        ReLU(),
        Conv2D(32, 64, kernel_size=3, stride=1, padding=1),
        BatchNorm2D(64),
        ReLU(),
        MaxPool2D(kernel_size=2, stride=2),
        Dropout(p=0.15),
        Conv2D(64, 128, kernel_size=3, stride=1, padding=1),
        BatchNorm2D(128),
        LeakyReLU(negative_slope=0.1),
        MaxPool2D(kernel_size=2, stride=2),
        Dropout(p=0.25),
        Flatten(),
        Dense(128 * 7 * 7, 256, activation_hint="relu"),
        ReLU(),
        Dropout(p=0.5),
        Dense(256, 10),
    )


def main():
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    model = build_cnn()
    optim = Adam(lr=5e-4, weight_decay=1e-4)
    trainer = ConvTrainer(model, optim, num_classes=10)
    trainer.train(X_train, y_train, epochs=8, batch_size=64, verbose=True, augment=True)
    trainer.plot_loss()
    trainer.evaluate(X_test, y_test)
    trainer.show_misclassifications(X_test, y_test, max_images=25, cols=5)


if __name__ == "__main__":
    main()
