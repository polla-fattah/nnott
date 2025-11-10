import numpy as np
from data_utils import DataUtility
from vmodules import Sequential, Linear, ReLU
from voptim import Adam, SGD
from vtrainer import VTrainer


def main():
    # Load and flatten
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)

    # Build vectorized model (logits output)
    model = Sequential(
        Linear(28*28, 256, activation_hint='relu'),
        ReLU(),
        Linear(256, 128, activation_hint='relu'),
        ReLU(),
        Linear(128, 10, activation_hint=None),
    )

    # Optimizer: Adam default
    optim = Adam(lr=1e-3, weight_decay=1e-4)
    trainer = VTrainer(model, optim, num_classes=10)

    # Train
    trainer.train(X_train, y_train, epochs=10, batch_size=32, verbose=True)
    trainer.plot_loss()

    # Evaluate
    trainer.evaluate(X_test, y_test)


if __name__ == "__main__":
    main()

