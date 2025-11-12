from ..modules import (
    Module,
    Sequential,
    Conv2D,
    BatchNorm2D,
    SiLU,
    DepthwiseConv2D,
    SqueezeExcite,
    GlobalAvgPool2D,
    Dense,
)
from .base import Architecture


class MBConvBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, kernel_size=3, se_ratio=4):
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = int(in_channels * expand_ratio)
        padding = kernel_size // 2 if isinstance(kernel_size, int) else kernel_size[0] // 2
        self.expand = (
            Sequential(
                Conv2D(in_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                BatchNorm2D(mid_channels),
                SiLU(),
            )
            if expand_ratio != 1
            else None
        )
        dw_in = mid_channels if self.expand is not None else in_channels
        self.depthwise = Sequential(
            DepthwiseConv2D(dw_in, kernel_size=kernel_size, stride=stride, padding=padding),
            BatchNorm2D(dw_in),
            SiLU(),
        )
        self.se = SqueezeExcite(dw_in, reduction=se_ratio)
        self.project = Sequential(
            Conv2D(dw_in, out_channels, kernel_size=1, stride=1, padding=0),
            BatchNorm2D(out_channels),
        )
        self.training = True

    def forward(self, x):
        out = x
        if self.expand is not None:
            out = self.expand.forward(out)
        out = self.depthwise.forward(out)
        out = self.se.forward(out)
        out = self.project.forward(out)
        if self.use_residual:
            out = out + x
        return out

    def backward(self, grad_output):
        grad_main = self.project.backward(grad_output)
        grad_main = self.se.backward(grad_main)
        grad_main = self.depthwise.backward(grad_main)
        if self.expand is not None:
            grad_main = self.expand.backward(grad_main)
        if self.use_residual:
            return grad_main + grad_output
        return grad_main

    def parameters(self):
        params = []
        if self.expand is not None:
            params.extend(self.expand.parameters())
        params.extend(self.depthwise.parameters())
        params.extend(self.se.parameters())
        params.extend(self.project.parameters())
        return params

    def zero_grad(self):
        if self.expand is not None:
            self.expand.zero_grad()
        self.depthwise.zero_grad()
        self.se.zero_grad()
        self.project.zero_grad()

    def train(self):
        if self.expand is not None:
            self.expand.train()
        self.depthwise.train()
        self.se.train()
        self.project.train()
        return self

    def eval(self):
        if self.expand is not None:
            self.expand.eval()
        self.depthwise.eval()
        self.se.eval()
        self.project.eval()
        return self


class EfficientNetLite0(Architecture):
    name = "efficientnet_lite0"

    def build(self, num_classes=10):
        layers = [
            Conv2D(1, 32, kernel_size=3, stride=1, padding=1),
            BatchNorm2D(32),
            SiLU(),
        ]
        in_channels = 32
        stage_cfg = [
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 3, 1, 5),
            (6, 320, 1, 1, 3),
        ]
        for expand, out_channels, repeats, stride, k in stage_cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                layers.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        stride=s,
                        expand_ratio=expand,
                        kernel_size=k,
                        se_ratio=4,
                    )
                )
                in_channels = out_channels
        layers.extend(
            [
                Conv2D(in_channels, 1280, kernel_size=1, stride=1, padding=0),
                BatchNorm2D(1280),
                SiLU(),
                GlobalAvgPool2D(),
                Dense(1280, num_classes),
            ]
        )
        return Sequential(*layers)


def efficientnet_lite0(num_classes=10, **kwargs):
    return EfficientNetLite0().build(num_classes=num_classes, **kwargs)


__all__ = ["MBConvBlock", "EfficientNetLite0", "efficientnet_lite0"]
