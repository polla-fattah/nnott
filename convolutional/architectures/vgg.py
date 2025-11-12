from ..modules import (
    Sequential,
    Conv2D,
    BatchNorm2D,
    ReLU,
    MaxPool2D,
    Flatten,
    Dense,
    Dropout,
)
from .base import Architecture


class VGG16(Architecture):
    name = "vgg16"

    def build(self, num_classes=10, dropout_p=0.5):
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


def vgg16(num_classes=10, **kwargs):
    return VGG16().build(num_classes=num_classes, **kwargs)


__all__ = ["VGG16", "vgg16"]
