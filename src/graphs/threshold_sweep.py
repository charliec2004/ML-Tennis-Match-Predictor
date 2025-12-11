"""Threshold sweep on validation set: precision/recall/F1 vs threshold."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import json
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from graphs.common import load_model_and_features, load_split, predict_split, save_fig, ensure_output_dir, plt


def generate(split: str = "val", thresholds=None):
    booster, feature_names = load_model_and_features()
    X, y, _ = load_split(split)
    proba = predict_split(booster, feature_names, X)

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    precisions = []
    recalls = []
    f1s = []
    for thr in thresholds:
        preds = (proba > thr).astype(int)
        precisions.append(precision_score(y, preds, zero_division=0))
        recalls.append(recall_score(y, preds, zero_division=0))
        f1s.append(f1_score(y, preds, zero_division=0))

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1 = f1s[best_idx]
    max_prec = float(max(precisions))
    max_rec = float(max(recalls))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1")
    ax.axvline(best_thr, color="red", linestyle="--", alpha=0.6, label=f"Best F1 thr={best_thr:.2f}")
    ax.set_xlabel("Threshold (P2 wins)")
    ax.set_ylabel("Score")
    ax.set_title(f"Threshold Sweep ({split} set) - Best F1: {best_f1:.2f} @ {best_thr:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_fig(fig, f"threshold_sweep_{split}.png")

    out_dir = ensure_output_dir()
    with open(out_dir / "thresholds.json", "w") as f:
        json.dump(
            {
                "best_f1_threshold": best_thr,
                "best_f1": best_f1,
                "max_precision": max_prec,
                "max_recall": max_rec,
            },
            f,
            indent=2,
        )
    print(f"Saved best F1 threshold {best_thr:.3f} to {out_dir/'thresholds.json'}")


if __name__ == "__main__":
    generate()
