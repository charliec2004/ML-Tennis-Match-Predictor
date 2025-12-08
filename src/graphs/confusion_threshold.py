"""Confusion/precision-recall at a chosen threshold."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from sklearn.metrics import confusion_matrix, precision_score, recall_score

from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def generate(split: str = "test", threshold: float = 0.6):
    booster, feature_names = load_model_and_features()
    X, y, _ = load_split(split)
    proba = predict_split(booster, feature_names, X)
    preds = (proba > threshold).astype(int)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["P1 wins", "P2 wins"])
    ax.set_yticklabels(["Actual P1", "Actual P2"])
    ax.set_title(f"Confusion Matrix @ {threshold} ({split} set)\nPrecision={prec:.2f} Recall={rec:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, f"confusion_{split}_thr{int(threshold*100)}.png")


if __name__ == "__main__":
    generate()
