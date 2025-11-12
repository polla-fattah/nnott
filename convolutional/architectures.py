"""Predefined convolutional model blueprints with class-based builders."""

from .modules import (
    Module,
    Sequential,
    Conv2D,
    BatchNorm2D,
    ReLU,
    LeakyReLU,
    MaxPool2D,
    Dropout,
    Flatten,
    Dense,
    DepthwiseConv2D,
    GlobalAvgPool2D,
    LayerNorm2D,
    SiLU,
    GELU,
    SqueezeExcite,
)


class Architecture:
    name = "base"

    def build(self, num_classes=10, **kwargs):
        raise NotImplementedError

    def __call__(self, num_classes=10, **kwargs):
        return self.build(num_classes=num_classes, **kwargs)


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


class ResidualBlock(Module):
    """Basic ResNet block with two 3x3 convolutions and an optional projection shortcut."""

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
        self._input = x
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


def lenet(num_classes=10, **kwargs):
    return LeNet().build(num_classes=num_classes, **kwargs)


def deeper_baseline(num_classes=10, **kwargs):
    return BaselineCNN().build(num_classes=num_classes, **kwargs)


def alexnet(num_classes=10, **kwargs):
    return AlexNet().build(num_classes=num_classes, **kwargs)


def vgg16(num_classes=10, **kwargs):
    return VGG16().build(num_classes=num_classes, **kwargs)


def resnet18(num_classes=10, **kwargs):
    return ResNet18().build(num_classes=num_classes, **kwargs)


def efficientnet_lite0(num_classes=10, **kwargs):
    return EfficientNetLite0().build(num_classes=num_classes, **kwargs)


def convnext_tiny(num_classes=10, **kwargs):
    return ConvNeXtTiny().build(num_classes=num_classes, **kwargs)


__all__ = [
    "Architecture",
    "LeNet",
    "BaselineCNN",
    "AlexNet",
    "VGG16",
    "ResNet18",
    "EfficientNetLite0",
    "ConvNeXtTiny",
    "lenet",
    "deeper_baseline",
    "alexnet",
    "vgg16",
    "resnet18",
    "efficientnet_lite0",
    "convnext_tiny",
]
