"""Predefined convolutional model blueprints (e.g., LeNet variants)."""

from .modules import (
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


def lenet(num_classes=10, use_batchnorm=True, dropout_p=0.0):
    """Classic LeNet-5 style stack for grayscale 28x28 inputs."""
    blocks = [
        Conv2D(1, 6, kernel_size=5, stride=1, padding=2),
    ]
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


def deeper_baseline(num_classes=10):
    """Slightly deeper CNN used as the default baseline."""
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


def alexnet(num_classes=10):
    """AlexNet-inspired architecture adapted for 28x28 grayscale images."""
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


def vgg16(num_classes=10, dropout_p=0.5):
    """VGG16-style network adapted to 28x28 grayscale inputs."""
    cfg = [
        (64, 2),
        (128, 2),
        (256, 3),
        (512, 3),
        (512, 3),
    ]
    blocks = []
    in_ch = 1
    spatial = 28
    downsampled = 0
    for out_ch, reps in cfg:
        for _ in range(reps):
            blocks.extend(
                [
                    Conv2D(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                    BatchNorm2D(out_ch),
                    ReLU(),
                ]
            )
            in_ch = out_ch
        stride = 2 if downsampled < 3 else 1
        blocks.append(MaxPool2D(kernel_size=2, stride=stride))
        if stride == 2:
            spatial = max(1, spatial // 2)
            downsampled += 1
    fc_in = in_ch * spatial * spatial
    blocks.extend(
        [
            Flatten(),
            Dense(fc_in, 4096, activation_hint="relu"),
            ReLU(),
            Dropout(p=dropout_p),
            Dense(4096, 4096, activation_hint="relu"),
            ReLU(),
            Dropout(p=dropout_p),
            Dense(4096, num_classes),
        ]
    )
    return Sequential(*blocks)


__all__ = ["lenet", "deeper_baseline", "alexnet", "vgg16"]
