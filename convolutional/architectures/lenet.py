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


class LeNet(Architecture):
    name = "lenet"

    def build(self, num_classes=10, use_batchnorm=True, dropout_p=0.0):
        blocks = [Conv2D(1, 6, kernel_size=5, stride=1, padding=2)]
        if use_batchnorm:
            blocks.append(BatchNorm2D(6))
        blocks.extend(
            [
                ReLU(),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(6, 16, kernel_size=5, stride=1, padding=0),
            ]
        )
        if use_batchnorm:
            blocks.append(BatchNorm2D(16))
        blocks.extend(
            [
                ReLU(),
                MaxPool2D(kernel_size=2, stride=2),
                Conv2D(16, 120, kernel_size=5, stride=1, padding=0),
                ReLU(),
                Flatten(),
                Dense(120, 84, activation_hint="relu"),
                ReLU(),
            ]
        )
        if dropout_p > 0.0:
            blocks.append(Dropout(p=dropout_p))
        blocks.append(Dense(84, num_classes))
        return Sequential(*blocks)


def lenet(num_classes=10, **kwargs):
    return LeNet().build(num_classes=num_classes, **kwargs)


__all__ = ["LeNet", "lenet"]
