import time
import numpy as np

from data_utils import DataUtility
from network import Network
from trainer import Trainer
from optimizer import SGD


def run_experiment(name, trainer_args, train_args):
    print(f"\n=== Running: {name} ===")
    # Reproducibility for init; adjust if you want variation
    np.random.seed(42)

    # Fresh network per run
    net = Network(
        input_size=28 * 28,
        num_classes=10,
        hidden_sizes=(128, 64),
        learning_rate=0.01,
        activation='relu',
    )

    trainer = Trainer(net, num_classes=10, **trainer_args)
    t0 = time.time()
    trainer.train(X_train, y_train, **train_args)
    acc = trainer.evaluate(X_test, y_test)
    dt = time.time() - t0
    print(f"{name} | Accuracy: {acc*100:.2f}% | Time: {dt:.1f}s")
    return acc


if __name__ == "__main__":
    # Load data once
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()

    results = []

    # 1) Adam, lr=1e-3, batch_size=32, epochs=10
    results.append(
        run_experiment(
            "Adam_b32_e10_lr1e-3",
            trainer_args=dict(optimizer="adam", lr=1e-3),
            train_args=dict(epochs=10, batch_size=32, verbose=True),
        )
    )

    # 2) SGD + momentum=0.9, batch_size=32, epochs=10, try lr 0.05 and 0.1
    for lr in (0.05, 0.1):
        results.append(
            run_experiment(
                f"SGD_m0.9_b32_e10_lr{lr}",
                trainer_args=dict(optimizer=SGD(lr=lr, momentum=0.9)),
                train_args=dict(epochs=10, batch_size=32, verbose=True),
            )
        )

    # 3) Keep batch_size=64 but compensate with more epochs (20) using Adam
    results.append(
        run_experiment(
            "Adam_b64_e20_lr1e-3",
            trainer_args=dict(optimizer="adam", lr=1e-3),
            train_args=dict(epochs=20, batch_size=64, verbose=True),
        )
    )

    print("\nDone.")

