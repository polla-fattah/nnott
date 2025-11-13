"""
Quick-start scenarios for the convolutional training pipeline.

Examples:

    python scripts/quickstart_convolutional.py --scenario baseline-cpu --plot
    python scripts/quickstart_convolutional.py --scenario gpu-fast --epochs 2 --lookahead
    python scripts/quickstart_convolutional.py --scenario resume-demo --save-path checkpoints/demo.npz
    python scripts/quickstart_convolutional.py --scenario dataset-swap \
        --alt-train-images fashion_train_images.npy \
        --alt-train-labels fashion_train_labels.npy \
        --alt-test-images fashion_test_images.npy \
        --alt-test-labels fashion_test_labels.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from vectorized.optim import Adam, Lookahead
from convolutional.main import plot_loss, plot_misclassifications


ARCH_REGISTRY = {
    "lenet": LeNet(),
    "baseline": BaselineCNN(),
    "alexnet": AlexNet(),
    "vgg16": VGG16(),
    "resnet18": ResNet18(),
    "efficientnet_lite0": EfficientNetLite0(),
    "convnext_tiny": ConvNeXtTiny(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convolutional quick-start scenarios.")
    parser.add_argument(
        "--scenario",
        choices=["baseline-cpu", "gpu-fast", "resume-demo", "dataset-swap"],
        default="baseline-cpu",
        help="Select the walkthrough to execute.",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Epochs for each training run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument(
        "--arch",
        choices=list(ARCH_REGISTRY.keys()),
        default="baseline",
        help="Architecture to use for resume/demo scenarios.",
    )
    parser.add_argument("--grad-clip", type=float, default=None, help="Gradient clipping norm.")
    parser.add_argument("--lookahead", action="store_true", help="Wrap optimizer with Lookahead when supported.")
    parser.add_argument("--save-path", default="checkpoints/quickstart_demo.npz", help="Checkpoint path for resume-demo.")
    parser.add_argument("--data-dir", default="data", help="Directory containing datasets.")
    parser.add_argument("--train-images", default="train_images.npy")
    parser.add_argument("--train-labels", default="train_labels.npy")
    parser.add_argument("--test-images", default="test_images.npy")
    parser.add_argument("--test-labels", default="test_labels.npy")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display Matplotlib plots. Collecting misclassifications runs another inference sweep and can be slow.",
    )
    parser.add_argument("--alt-train-images", default="fashion_train_images.npy")
    parser.add_argument("--alt-train-labels", default="fashion_train_labels.npy")
    parser.add_argument("--alt-test-images", default="fashion_test_images.npy")
    parser.add_argument("--alt-test-labels", default="fashion_test_labels.npy")
    parser.add_argument(
        "--image-shape",
        default="1,28,28",
        help="For dataset-swap: channels,height,width (e.g., 3,32,32 for CIFAR-10).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scenario == "baseline-cpu":
        run_baseline_cpu(args)
    elif args.scenario == "gpu-fast":
        run_gpu_fast(args)
    elif args.scenario == "resume-demo":
        run_resume_demo(args)
    elif args.scenario == "dataset-swap":
        run_dataset_swap(args)
    else:
        raise ValueError(f"Unknown scenario {args.scenario}")


def run_baseline_cpu(args):
    print("== Baseline CNN on CPU ==")
    data = load_dataset(
        args.data_dir,
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels,
        shape=(1, 28, 28),
    )
    model = build_cnn("baseline")
    trainer = ConvTrainer(model, Adam(lr=5e-4, weight_decay=1e-4), grad_clip_norm=args.grad_clip)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        augment=True,
    )
    if args.plot:
        plot_loss(trainer.loss_history)
    trainer.evaluate(data.X_test, data.y_test)
    imgs, preds, trues, total = trainer.collect_misclassifications(data.X_test, data.y_test, max_images=25)
    if args.plot and total:
        plot_misclassifications(imgs, preds, trues, total)


def run_gpu_fast(args):
    print("== ResNet18 accelerated on GPU ==")
    try:
        backend.use_gpu()
        print("GPU backend enabled.")
    except RuntimeError as exc:
        print(f"[WARN] {exc} -- continuing on CPU.")
    data = load_dataset(
        args.data_dir,
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels,
        shape=(1, 28, 28),
    )
    model = build_cnn("resnet18")
    optim = Adam(lr=5e-4, weight_decay=1e-4)
    if args.lookahead:
        optim = Lookahead(optim, k=5, alpha=0.5)
    trainer = ConvTrainer(model, optim, grad_clip_norm=args.grad_clip or 5.0)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        augment=True,
    )
    trainer.evaluate(data.X_test, data.y_test)
    if args.plot:
        plot_loss(trainer.loss_history)


def run_resume_demo(args):
    print("== Resume-from-checkpoint demo ==")
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_dataset(
        args.data_dir,
        args.train_images,
        args.train_labels,
        args.test_images,
        args.test_labels,
        shape=(1, 28, 28),
    )
    model = build_cnn(args.arch)
    trainer = ConvTrainer(model, Adam(lr=5e-4, weight_decay=1e-4), grad_clip_norm=args.grad_clip)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=max(1, args.epochs // 2),
        batch_size=args.batch_size,
        verbose=True,
        augment=True,
    )
    trainer.save_model(str(save_path), metadata={"scenario": "resume-demo"})
    print(f"Checkpoint written to {save_path}")

    # Reload and continue
    trainer.load_model(str(save_path))
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=max(1, args.epochs - max(1, args.epochs // 2)),
        batch_size=args.batch_size,
        verbose=True,
        augment=True,
    )
    trainer.evaluate(data.X_test, data.y_test)
    if args.plot:
        plot_loss(trainer.loss_history)


def run_dataset_swap(args):
    print("== Dataset swap (custom .npy files) ==")
    shape = tuple(int(dim.strip()) for dim in args.image_shape.split(",") if dim.strip())
    if len(shape) != 3:
        raise ValueError("image-shape must be channels,height,width (e.g., 3,32,32)")
    alt_files = [
        args.alt_train_images,
        args.alt_train_labels,
        args.alt_test_images,
        args.alt_test_labels,
    ]
    for file in alt_files:
        full = Path(args.data_dir) / file
        if not full.exists():
            print(f"[WARN] {full} not found. Convert your dataset to .npy before running this scenario.")
    data = load_dataset(
        args.data_dir,
        args.alt_train_images,
        args.alt_train_labels,
        args.alt_test_images,
        args.alt_test_labels,
        shape=shape,
    )
    model = build_cnn(args.arch)
    trainer = ConvTrainer(model, Adam(lr=5e-4, weight_decay=1e-4), grad_clip_norm=args.grad_clip)
    trainer.train(
        data.X_train,
        data.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        augment=True,
    )
    trainer.evaluate(data.X_test, data.y_test)
    if args.plot:
        plot_loss(trainer.loss_history)
        imgs, preds, trues, total = trainer.collect_misclassifications(data.X_test, data.y_test, max_images=25)
        if total:
            plot_misclassifications(imgs, preds, trues, total)


def build_cnn(name: str):
    key = name.lower()
    if key not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[key].build(num_classes=10)


class DatasetBundle:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


def load_dataset(data_dir, train_images, train_labels, test_images, test_labels, shape):
    util = DataUtility(data_dir)
    X_train, y_train, X_test, y_test = util.load_data(
        train_images_file=train_images,
        train_labels_file=train_labels,
        test_images_file=test_images,
        test_labels_file=test_labels,
    )
    C, H, W = shape
    X_train = X_train.reshape(len(X_train), C, H, W).astype(np.float32)
    X_test = X_test.reshape(len(X_test), C, H, W).astype(np.float32)
    return DatasetBundle(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
