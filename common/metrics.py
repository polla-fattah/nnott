import numpy as np


def confusion_matrix(preds, targets, num_classes):
    """Compute a num_classes x num_classes confusion matrix."""
    preds = np.asarray(preds, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1
    return mat


def format_confusion_matrix(cm):
    num_classes = cm.shape[0]
    header = ["     "] + [f"P{c:>4}" for c in range(num_classes)]
    lines = [" ".join(header)]
    for idx in range(num_classes):
        row_vals = " ".join(f"{cm[idx, j]:>5}" for j in range(num_classes))
        lines.append(f"T{idx:>2} | {row_vals}")
    return "\n".join(lines)
