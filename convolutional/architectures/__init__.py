"""Collection of convolutional architecture builders."""

from .base import Architecture
from .lenet import LeNet, lenet
from .baseline import BaselineCNN, deeper_baseline
from .alexnet import AlexNet, alexnet
from .vgg import VGG16, vgg16
from .resnet import ResNet18, resnet18
from .efficientnet import EfficientNetLite0, efficientnet_lite0
from .convnext import ConvNeXtTiny, convnext_tiny

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
