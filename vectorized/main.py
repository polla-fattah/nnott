import os, sys, numpy as np
import argparse
import matplotlib.pyplot as plt

# Allow running this file directly (adds project root to sys.path)
if __package__ is None or __package__ == "":
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from common.data_utils import DataUtility
from common.augment import build_augment_config
from common.seed import set_global_seed
from common.metrics import confusion_matrix, format_confusion_matrix
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
        "--gpu",
        action="store_true",
        help="Placeholder flag for compatibility; vectorized trainer currently runs on CPU only.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility.")
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training-time data augmentation.",
    )
    parser.add_argument("--augment-max-shift", type=int, default=2, help="Pixel shift radius for jitter augmentation.")
    parser.add_argument("--augment-rotate-deg", type=float, default=10.0, help="Max rotation (degrees). Set 0 to disable.")
    parser.add_argument("--augment-rotate-prob", type=float, default=0.5, help="Probability of applying rotation.")
    parser.add_argument("--augment-hflip-prob", type=float, default=0.5, help="Probability of horizontal flip.")
    parser.add_argument("--augment-vflip-prob", type=float, default=0.0, help="Probability of vertical flip.")
    parser.add_argument("--augment-noise-std", type=float, default=0.02, help="Stddev for Gaussian noise (0 disables).")
    parser.add_argument("--augment-noise-prob", type=float, default=0.3, help="Probability of injecting noise.")
    parser.add_argument("--augment-noise-clip", type=float, default=3.0, help="Clamp magnitude after noises/flips.")
    parser.add_argument("--augment-cutout-prob", type=float, default=0.0, help="Probability of applying cutout mask.")
    parser.add_argument("--augment-cutout-size", type=int, default=4, help="Cutout square size (pixels).")
    parser.add_argument("--augment-cutmix-prob", type=float, default=0.0, help="Probability of CutMix (label mixing enabled).")
    parser.add_argument("--augment-cutmix-alpha", type=float, default=1.0, help="Beta distribution alpha for CutMix lambda.")
    parser.add_argument("--augment-randaug-layers", type=int, default=0, help="Number of RandAugment layers (0 disables).")
    parser.add_argument("--augment-randaug-magnitude", type=float, default=0.0, help="RandAugment magnitude (0-1).")
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
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of training data used for validation/early stopping (0 disables).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["none", "cosine", "reduce_on_plateau"],
        default="none",
        help="Learning-rate schedule strategy.",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum LR for schedulers.")
    parser.add_argument("--reduce-factor", type=float, default=0.5, help="LR reduction factor for plateau scheduler.")
    parser.add_argument("--reduce-patience", type=int, default=3, help="Epochs to wait before reducing LR.")
    parser.add_argument("--reduce-delta", type=float, default=1e-4, help="Minimum improvement for plateau scheduler.")
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping based on validation loss.",
    )
    parser.add_argument("--early-patience", type=int, default=5, help="Validation epochs without improvement before stopping.")
    parser.add_argument("--early-delta", type=float, default=1e-4, help="Minimum validation loss improvement counted as better.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Enable matplotlib plots (disabled by default). Collecting misclassifications requires an extra full pass and can be slow.",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Print the confusion matrix after evaluating on the test set.",
    )
    return parser.parse_args()


def main(opts=None):
    args = opts or parse_args()
    set_global_seed(args.seed)
    if args.gpu:
        print("[WARN] --gpu requested but the vectorized trainer currently runs on CPU (NumPy) only.")

    # Load and flatten
    X_train, y_train, X_test, y_test = DataUtility("data").load_data()
    val_split = min(max(args.val_split, 0.0), 0.4)
    X_train, y_train, X_val, y_val = DataUtility.train_val_split(
        X_train, y_train, val_fraction=val_split, seed=args.seed
    )
    X_train = X_train.reshape(len(X_train), -1).astype(np.float32)
    X_test = X_test.reshape(len(X_test), -1).astype(np.float32)
    if X_val is not None:
        X_val = X_val.reshape(len(X_val), -1).astype(np.float32)

    if X_val is not None:
        print(f"[Data] Validation split: {len(X_val)} samples ({val_split*100:.1f}%). Training size: {len(X_train)}")

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

    scheduler_config = build_scheduler_config(args)
    early_config = build_early_config(args)
    augment_config = build_augment_config(
        max_shift=args.augment_max_shift,
        rotate_deg=args.augment_rotate_deg,
        rotate_prob=args.augment_rotate_prob,
        hflip_prob=args.augment_hflip_prob,
        vflip_prob=args.augment_vflip_prob,
        noise_std=args.augment_noise_std,
        noise_prob=args.augment_noise_prob,
        noise_clip=args.augment_noise_clip,
        cutout_prob=args.augment_cutout_prob,
        cutout_size=args.augment_cutout_size,
        cutmix_prob=args.augment_cutmix_prob,
        cutmix_alpha=args.augment_cutmix_alpha,
        randaugment_layers=args.augment_randaug_layers,
        randaugment_magnitude=args.augment_randaug_magnitude,
    )
    trainer = VTrainer(
        model,
        optim,
        num_classes=10,
        lr_scheduler_config=scheduler_config,
        early_stopping_config=early_config,
        augment_config=augment_config,
    )

    val_tuple = (X_val, y_val) if X_val is not None else None
    trainer.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
        val_data=val_tuple,
        augment=not args.no_augment,
    )
    if args.plot:
        plot_loss(trainer.loss_history)

    if args.confusion_matrix:
        acc, preds, targets = trainer.evaluate(X_test, y_test, return_preds=True)
        cm = confusion_matrix(preds, targets, num_classes=10)
        print("Confusion Matrix:\n" + format_confusion_matrix(cm))
    else:
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
    if len(acts) < length:
        if len(acts) == 1:
            acts = acts * length
        else:
            reps = (length + len(acts) - 1) // len(acts)
            acts = (acts * reps)[:length]
    elif len(acts) > length:
        acts = acts[:length]
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
    if len(vals) < length:
        if len(vals) == 1:
            vals = vals * length
        else:
            reps = (length + len(vals) - 1) // len(vals)
            vals = (vals * reps)[:length]
    elif len(vals) > length:
        vals = vals[:length]
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


def build_scheduler_config(args):
    schedule = args.lr_schedule
    if schedule == "none":
        return None
    if schedule == "cosine":
        return {
            "type": "cosine",
            "min_lr": args.min_lr,
            "total_epochs": args.epochs,
        }
    if schedule == "reduce_on_plateau":
        return {
            "type": "reduce_on_plateau",
            "factor": args.reduce_factor,
            "patience": args.reduce_patience,
            "min_lr": args.min_lr,
            "min_delta": args.reduce_delta,
        }
    return None


def build_early_config(args):
    if not args.early_stopping:
        return None
    return {
        "patience": args.early_patience,
        "min_delta": args.early_delta,
    }


if __name__ == "__main__":
    main()
