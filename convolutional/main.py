import os
import sys
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from common.data_utils import DataUtility
from convolutional.architectures import lenet, deeper_baseline, alexnet
from convolutional.trainer import ConvTrainer
from vectorized.optim import Adam


ARCH_REGISTRY = {
    "lenet": lenet,
    "baseline": deeper_baseline,
    "alexnet": alexnet,
}


def build_cnn(name="baseline", num_classes=10):
    name = name.lower()
    if name not in ARCH_REGISTRY:
        raise ValueError(f"Unknown architecture '{name}'. Available: {list(ARCH_REGISTRY)}")
    return ARCH_REGISTRY[name](num_classes=num_classes)


def main(arch_name="baseline"):
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
    X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

    model = build_cnn(arch_name, num_classes=10)
    optim = Adam(lr=5e-4, weight_decay=1e-4)
    trainer = ConvTrainer(model, optim, num_classes=10)
    trainer.train(X_train, y_train, epochs=8, batch_size=64, verbose=True, augment=True)
    trainer.plot_loss()
    trainer.evaluate(X_test, y_test)
    trainer.show_misclassifications(X_test, y_test, max_images=25, cols=5)


if __name__ == "__main__":
    arch = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    main(arch_name=arch)
