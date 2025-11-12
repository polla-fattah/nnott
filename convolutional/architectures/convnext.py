from ..modules import (
    Module,
    Sequential,
    Conv2D,
    LayerNorm2D,
    DepthwiseConv2D,
    GELU,
    GlobalAvgPool2D,
    Dense,
)
from .base import Architecture


class ConvNeXtBlock(Module):
    def __init__(self, channels, expansion=4):
        self.dw = DepthwiseConv2D(channels, kernel_size=7, stride=1, padding=3)
        self.norm = LayerNorm2D(channels)
        self.pw1 = Conv2D(channels, channels * expansion, kernel_size=1, stride=1, padding=0)
        self.act = GELU()
        self.pw2 = Conv2D(channels * expansion, channels, kernel_size=1, stride=1, padding=0)
        self.training = True

    def forward(self, x):
        out = self.dw.forward(x)
        out = self.norm.forward(out)
        out = self.pw1.forward(out)
        out = self.act.forward(out)
        out = self.pw2.forward(out)
        return out + x

    def backward(self, grad_output):
        grad_main = self.pw2.backward(grad_output)
        grad_main = self.act.backward(grad_main)
        grad_main = self.pw1.backward(grad_main)
        grad_main = self.norm.backward(grad_main)
        grad_main = self.dw.backward(grad_main)
        return grad_main + grad_output

    def parameters(self):
        params = []
        params.extend(self.dw.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.pw1.parameters())
        params.extend(self.pw2.parameters())
        return params

    def zero_grad(self):
        self.dw.zero_grad()
        self.norm.zero_grad()
        self.pw1.zero_grad()
        self.pw2.zero_grad()

    def train(self):
        self.dw.train()
        self.norm.train()
        self.pw1.train()
        self.act.train()
        self.pw2.train()
        return self

    def eval(self):
        self.dw.eval()
        self.norm.eval()
        self.pw1.eval()
        self.act.eval()
        self.pw2.eval()
        return self


class ConvNeXtTiny(Architecture):
    name = "convnext_tiny"

    def build(self, num_classes=10):
        layers = [
            Conv2D(1, 64, kernel_size=4, stride=2, padding=1),
            LayerNorm2D(64),
        ]
        stage_channels = [64, 128, 256, 512]
        stage_repeats = [3, 3, 6, 3]
        in_channels = stage_channels[0]
        for idx, repeats in enumerate(stage_repeats):
            for _ in range(repeats):
                layers.append(ConvNeXtBlock(in_channels))
            if idx < len(stage_channels) - 1:
                next_channels = stage_channels[idx + 1]
                layers.extend(
                    [
                        Conv2D(in_channels, next_channels, kernel_size=2, stride=2, padding=0),
                        LayerNorm2D(next_channels),
                    ]
                )
                in_channels = next_channels
        layers.extend(
            [
                GlobalAvgPool2D(),
                Dense(in_channels, num_classes),
            ]
        )
        return Sequential(*layers)


def convnext_tiny(num_classes=10, **kwargs):
    return ConvNeXtTiny().build(num_classes=num_classes, **kwargs)


__all__ = ["ConvNeXtBlock", "ConvNeXtTiny", "convnext_tiny"]
