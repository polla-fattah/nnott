import numpy as np
from cross_entropy import cross_entropy_from_logits, cross_entropy_grad_logits


def make_toy(seed=0):
    rng = np.random.default_rng(seed)
    # Three Gaussian blobs in 2D
    centers = np.array([[2, 2], [-2, 0], [0, -2]], dtype=np.float64)
    X = []
    y = []
    for c, center in enumerate(centers):
        X.append(center + rng.normal(scale=0.5, size=(50, 2)))
        y.append(np.full(50, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    # shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def train_linear_ce(X, y, lr=0.1, epochs=50, seed=0):
    rng = np.random.default_rng(seed)
    N, D = X.shape
    C = int(y.max()) + 1
    W = rng.normal(scale=0.01, size=(D, C))
    b = np.zeros(C)

    history = []
    for _ in range(epochs):
        # logits
        Z = X @ W + b
        loss = cross_entropy_from_logits(Z, y, reduction="mean")
        history.append(loss)

        # gradient w.r.t logits then params
        dZ = cross_entropy_grad_logits(Z, y) / N
        dW = X.T @ dZ
        db = dZ.sum(axis=0)

        W -= lr * dW
        b -= lr * db

    # compute accuracy
    logits = X @ W + b
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()
    return history, acc


def main():
    X, y = make_toy()
    h, acc = train_linear_ce(X, y, lr=0.2, epochs=100)
    print(f"Initial loss: {h[0]:.4f}  Final loss: {h[-1]:.4f}")
    print(f"Training accuracy: {acc*100:.2f}%")
    assert h[-1] < h[0] * 0.5, "Loss did not decrease sufficiently"
    assert acc > 0.85, "Accuracy too low for toy problem"
    print("sanity training passed ✔")


if __name__ == "__main__":
    main()

