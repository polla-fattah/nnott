import os, sys, numpy as np
import argparse
import matplotlib.pyplot as plt

# Allow running this file directly (adds project root to sys.path)
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from common.data_utils import DataUtility
from vectorized.modules import (
    ACTIVATION_KINDS,
    Sequential,
    Linear,
    Dropout,
    BatchNorm1D,
    activation_from_name,
)
from vectorized.optim import Adam, SGD
from vectorized.trainer import VTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate the vectorized MLP.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default="256,128,64",
        help="Comma-separated hidden layer sizes (e.g., 256,128,64).",
    )
    parser.add_argument(
        "--activation",
        choices=sorted(k for k in ACTIVATION_KINDS if k != "linear"),
        default="relu",
        help="Default hidden activation (used when --hidden-activations is omitted).",
    )
    parser.add_argument(
        "--hidden-activations",
        type=str,
        default=None,
        help="Comma-separated activation list per hidden layer (e.g., relu,leaky_relu,tanh).",
    )
    parser.add_argument(
        "--dropout",
        type=str,
        default="0.2",
        help="Dropout probabilities per hidden layer (single value or comma list). Use 0 to disable.",
    )
    parser.add_argument(
        "--batchnorm",
        action="store_true",
        help="Insert BatchNorm1D after each Linear layer before its activation.",
    )
    parser.add_argument("--bn-momentum", type=float, default=0.1, help="BatchNorm momentum (if enabled).")
    parser.add_argument(
        "--leaky-negative-slope",
        type=float,
        default=0.01,
        help="Negative slope for LeakyReLU activations.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "sgd"],
        default="adam",
        help="Optimizer to use for training.",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay value.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable matplotlib plots (disabled by default). Collecting misclassifications requires an extra full pass and can be slow.",
    )
    return parser.parse_args()


def main(opts=None):
    args = opts or parse_args()

    # Load and flatten
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)

    hidden_sizes = parse_hidden_sizes(args.hidden_sizes, default=(256, 128, 64))
    hidden_acts = parse_activation_list(
        args.hidden_activations,
        len(hidden_sizes),
        default_act=args.activation,
    )
    dropout_list = parse_dropout_list(args.dropout, len(hidden_sizes))

    model = build_mlp(
        hidden_sizes=hidden_sizes,
        hidden_activations=hidden_acts,
        dropout_list=dropout_list,
        use_batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
        negative_slope=args.leaky_negative_slope,
    )

    if args.optimizer == "sgd":
        optim = SGD(lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optim = Adam(lr=args.lr, weight_decay=args.weight_decay)
    trainer = VTrainer(model, optim, num_classes=10)

    trainer.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size, verbose=True)
    if args.plot:
        plot_loss(trainer.loss_history)

    trainer.evaluate(X_test, y_test)
    imgs, preds, trues, total = trainer.misclassification_data(X_test, y_test, max_images=100)
    if args.plot and total:
        if confirm_heavy_step("collect and plot misclassifications"):
            plot_misclassifications(imgs, preds, trues, total)


def plot_loss(loss_history):
    if not loss_history:
        return
    epochs = range(1, len(loss_history) + 1)
    plt.figure()
    plt.plot(epochs, loss_history, marker="o")
    plt.title("Vectorized Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_misclassifications(imgs, preds, trues, total, cols=5):
    n = len(imgs)
    cols = max(1, min(cols, n))
    rows = int(np.ceil(n / cols))
    side = int(np.sqrt(imgs.shape[1])) if imgs.size else 0
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, true, pred in zip(axes, imgs, trues, preds):
        disp = img
        if side and side * side == img.size:
            disp = img.reshape(side, side)
        imin, imax = float(disp.min()), float(disp.max())
        if imax > imin:
            disp = (disp - imin) / (imax - imin)
        ax.imshow(disp, cmap="gray")
        ax.set_title(f"T:{int(true)} P:{int(pred)}")
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Misclassifications: showing {n} of {total}")
    plt.tight_layout()
    plt.show()


def confirm_heavy_step(task):
    response = input(f"\nAbout to {task}, which requires an extra pass and may take time. Continue? [y/N]: ").strip().lower()
    return response in {"y", "yes"}


def parse_hidden_sizes(spec: str, default):
    values = tuple(int(h.strip()) for h in (spec or "").split(",") if h.strip())
    return values or tuple(default)


def parse_activation_list(spec: str, length: int, default_act: str):
    if spec:
        acts = [a.strip().lower() for a in spec.split(",") if a.strip()]
    else:
        acts = []
    if not acts:
        acts = [default_act] * length
    if len(acts) == 1 and length > 1:
        acts = acts * length
    if len(acts) != length:
        raise ValueError("hidden_activations must match number of hidden layers.")
    for act in acts:
        if act not in ACTIVATION_KINDS or act == "linear":
            raise ValueError(f"Unsupported hidden activation '{act}'.")
    return acts


def parse_dropout_list(spec: str, length: int):
    if not spec:
        return [0.0] * length
    raw = [s.strip().lower() for s in spec.split(",") if s.strip()]
    vals = []
    for item in raw:
        if item in {"none", "off"}:
            vals.append(0.0)
        else:
            vals.append(max(0.0, min(0.95, float(item))))
    if len(vals) == 1 and length > 1:
        vals = vals * length
    if len(vals) != length:
        raise ValueError("Dropout list must have one value or match hidden layer count.")
    return vals


def build_mlp(
    hidden_sizes,
    hidden_activations,
    dropout_list,
    use_batchnorm=False,
    bn_momentum=0.1,
    negative_slope=0.01,
):
    layers = []
    in_dim = 28 * 28
    for idx, (h, act_name) in enumerate(zip(hidden_sizes, hidden_activations)):
        layers.append(Linear(in_dim, h, activation_hint=act_name))
        if use_batchnorm:
            layers.append(BatchNorm1D(h, momentum=bn_momentum))
        layers.append(
            activation_from_name(
                act_name,
                negative_slope=negative_slope,
            )
        )
        drop_p = dropout_list[idx]
        if drop_p > 0.0:
            layers.append(Dropout(drop_p))
        in_dim = h
    layers.append(Linear(in_dim, 10, activation_hint=None))
    return Sequential(*layers)


if __name__ == "__main__":
    main()
