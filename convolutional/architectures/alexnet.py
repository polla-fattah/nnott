from ..modules import (
    Sequential,
    Conv2D,
    BatchNorm2D,
    ReLU,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
)
from .base import Architecture


class AlexNet(Architecture):
    name = "alexnet"

    def build(self, num_classes=10):
        return Sequential(
            Conv2D(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(64),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(64, 192, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(192),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(192, 384, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(384),
            ReLU(),
            Conv2D(384, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(256),
            ReLU(),
            Conv2D(256, 256, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(256),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Dense(256 * 3 * 3, 1024, activation_hint="relu"),
            ReLU(),
            Dropout(p=0.5),
            Dense(1024, 512, activation_hint="relu"),
            ReLU(),
            Dropout(p=0.5),
            Dense(512, num_classes),
        )


def alexnet(num_classes=10, **kwargs):
    return AlexNet().build(num_classes=num_classes, **kwargs)


__all__ = ["AlexNet", "alexnet"]
