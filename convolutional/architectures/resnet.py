from ..modules import (
    Module,
    Sequential,
    Conv2D,
    BatchNorm2D,
    ReLU,
    Flatten,
    Dense,
    Dropout,
)
from .base import Architecture


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        self.conv1 = Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2D(out_channels)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2D(out_channels)
        self.relu2 = ReLU()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                BatchNorm2D(out_channels),
            )
        else:
            self.shortcut = None
        self.training = True

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        residual = x if self.shortcut is None else self.shortcut.forward(x)
        out = out + residual
        out = self.relu2.forward(out)
        return out

    def backward(self, grad_output):
        grad = self.relu2.backward(grad_output)
        grad_main = self.bn2.backward(grad)
        grad_main = self.conv2.backward(grad_main)
        grad_main = self.relu1.backward(grad_main)
        grad_main = self.bn1.backward(grad_main)
        grad_main = self.conv1.backward(grad_main)
        grad_skip = grad if self.shortcut is None else self.shortcut.backward(grad)
        return grad_main + grad_skip

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        if self.shortcut is not None:
            params.extend(self.shortcut.parameters())
        return params

    def zero_grad(self):
        self.conv1.zero_grad()
        self.bn1.zero_grad()
        self.conv2.zero_grad()
        self.bn2.zero_grad()
        if self.shortcut is not None:
            self.shortcut.zero_grad()

    def train(self):
        self.conv1.train()
        self.bn1.train()
        self.relu1.train()
        self.conv2.train()
        self.bn2.train()
        self.relu2.train()
        if self.shortcut is not None:
            self.shortcut.train()
        return self

    def eval(self):
        self.conv1.eval()
        self.bn1.eval()
        self.relu1.eval()
        self.conv2.eval()
        self.bn2.eval()
        self.relu2.eval()
        if self.shortcut is not None:
            self.shortcut.eval()
        return self


def _make_resnet_layer(in_channels, out_channels, blocks, stride):
    layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return layers


class ResNet18(Architecture):
    name = "resnet18"

    def build(self, num_classes=10):
        layers = [
            Conv2D(1, 64, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(64),
            ReLU(),
        ]
        in_channels = 64
        stage_cfg = [
            (64, 2, 1),
            (128, 2, 2),
            (256, 2, 2),
            (512, 2, 2),
        ]
        for out_channels, blocks, stride in stage_cfg:
            layers.extend(_make_resnet_layer(in_channels, out_channels, blocks, stride))
            in_channels = out_channels
        layers.extend(
            [
                Flatten(),
                Dense(512 * 4 * 4, 512, activation_hint="relu"),
                ReLU(),
                Dropout(p=0.5),
                Dense(512, num_classes),
            ]
        )
        return Sequential(*layers)


def resnet18(num_classes=10, **kwargs):
    return ResNet18().build(num_classes=num_classes, **kwargs)


__all__ = ["ResidualBlock", "ResNet18", "resnet18"]
