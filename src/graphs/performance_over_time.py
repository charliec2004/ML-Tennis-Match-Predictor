"""Performance over time: AUC/accuracy by year for train/val/test."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def _yearly_metrics(y_true: pd.Series, proba: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    df = meta[["date"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["target"] = y_true.values
    df["proba"] = proba
    out = []
    for year, grp in df.groupby("year"):
        if len(grp) < 10:
            continue
        out.append(
            {
                "year": year,
                "auc": roc_auc_score(grp["target"], grp["proba"]),
                "acc": accuracy_score(grp["target"], (grp["proba"] > 0.5).astype(int)),
            }
        )
    return pd.DataFrame(out)


def generate():
    booster, feature_names = load_model_and_features()
    fig, ax = plt.subplots(figsize=(10, 6))

    for split in ["train", "val", "test"]:
        X, y, meta = load_split(split)
        proba = predict_split(booster, feature_names, X)
        metrics = _yearly_metrics(y, proba, meta)
        if metrics.empty:
            continue
        ax.plot(metrics["year"], metrics["auc"], marker="o", label=f"{split} AUC")
        ax.plot(metrics["year"], metrics["acc"], marker="s", linestyle="--", label=f"{split} Acc")

    ax.set_title("Performance Over Time (AUC / Accuracy by Year)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "performance_over_time.png")


if __name__ == "__main__":
    generate()
