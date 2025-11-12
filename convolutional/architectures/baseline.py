from ..modules import (
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
from .base import Architecture


class BaselineCNN(Architecture):
    name = "baseline"

    def build(self, num_classes=10):
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
            Dense(256, num_classes),
        )


def deeper_baseline(num_classes=10, **kwargs):
    return BaselineCNN().build(num_classes=num_classes, **kwargs)


__all__ = ["BaselineCNN", "deeper_baseline"]
