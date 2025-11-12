import os
import sys
import argparse
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
import common.backend as backend
from convolutional.architectures import (
    LeNet,
    BaselineCNN,
    AlexNet,
    VGG16,
    ResNet18,
    EfficientNetLite0,
    ConvNeXtTiny,
)
from convolutional.trainer import ConvTrainer
from vectorized.optim import Adam


ARCH_REGISTRY = {
    "lenet": LeNet(),
    "baseline": BaselineCNN(),
    "alexnet": AlexNet(),
    "vgg16": VGG16(),
    "resnet18": ResNet18(),
    "efficientnet_lite0": EfficientNetLite0(),
    "convnext_tiny": ConvNeXtTiny(),
}


def build_cnn(name="baseline", num_classes=10):
    name = name.lower()
    if name not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}")
    builder = ARCH_REGISTRY[name]
    return builder.build(num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate convolutional architectures.")
    parser.add_argument("arch", nargs="?", default="baseline", help="Architecture key (e.g., baseline, lenet, resnet18).")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--save", type=str, help="Path to save trained weights.")
    parser.add_argument("--load", type=str, help="Load weights before training/evaluation.")
    parser.add_argument("--no-augment", action="store_true", help="Disable shift augmentation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip training phase (use with --load).")
    parser.add_argument("--gpu", action="store_true", help="Use CuPy GPU backend if available.")
    return parser.parse_args()


def main(opts=None):
    args = opts or parse_args()
    arch_name = args.arch.lower()
    if args.gpu:
        try:
            backend.use_gpu()
            print("GPU backend enabled via CuPy.")
        except RuntimeError as exc:
            print(f"[WARN] {exc} Falling back to CPU backend.")
            backend.use_cpu()
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    model = build_cnn(arch_name, num_classes=10)
    optim = Adam(lr=5e-4, weight_decay=1e-4)
    trainer = ConvTrainer(model, optim, num_classes=10)

    if args.load:
        meta = trainer.load_model(args.load)
        msg = f"Loaded weights from {args.load}"
        if meta:
            msg += f" | metadata: {meta}"
        print(msg)

    if not args.skip_train:
        trainer.train(
            X_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
            augment=not args.no_augment,
        )
        trainer.plot_loss()
    else:
        print("Skipping training as requested.")

    trainer.evaluate(X_test, y_test)
    trainer.show_misclassifications(X_test, y_test, max_images=25, cols=5)

    if args.save:
        metadata = {"arch": arch_name, "epochs": args.epochs}
        trainer.save_model(args.save, metadata=metadata)
        print(f"Saved weights to {args.save}")


if __name__ == "__main__":
    main()
