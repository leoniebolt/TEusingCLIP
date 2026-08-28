from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from project_config import TRAV_THRESHOLD


def binary_metrics(probabilities, targets, threshold=TRAV_THRESHOLD):
    probabilities = np.asarray(probabilities, dtype=float)
    targets = np.asarray(targets, dtype=float)
    pred = (probabilities >= threshold).astype(int)
    truth = (targets >= threshold).astype(int)
    return {
        "f1": f1_score(truth, pred, zero_division=0),
        "mcc": matthews_corrcoef(truth, pred),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1]),
        "n": len(truth),
    }


def save_confusion_matrix(cm, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["0", "1"])
    ax.set_yticks([0, 1], labels=["0", "1"])
    for (r, c), value in np.ndenumerate(cm):
        ax.text(c, r, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
