"""Accuracy/AUC by surface and by round on test set."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def _metric_by(values: pd.Series, y: pd.Series, proba, label: str) -> pd.DataFrame:
    df = pd.DataFrame({label: values, "target": y.values, "proba": proba})
    out = []
    for val, grp in df.groupby(label):
        if len(grp) < 20:
            continue
        out.append(
            {
                label: val,
                "auc": roc_auc_score(grp["target"], grp["proba"]),
                "acc": accuracy_score(grp["target"], (grp["proba"] > 0.5).astype(int)),
                "count": len(grp),
            }
        )
    return pd.DataFrame(out)


def generate(split: str = "test"):
    booster, feature_names = load_model_and_features()
    X, y, meta = load_split(split)
    proba = predict_split(booster, feature_names, X)

    surf_metrics = _metric_by(meta["surface"], y, proba, "surface") if "surface" in meta.columns else pd.DataFrame()

    round_source = None
    if "round" in meta.columns:
        round_source = meta["round"]
    elif "Round" in meta.columns:
        round_source = meta["Round"]
    elif "round" in X.columns:
        round_source = X["round"]
    round_metrics = _metric_by(round_source, y, proba, "round") if round_source is not None else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if not surf_metrics.empty:
        axes[0].barh(surf_metrics["surface"], surf_metrics["auc"], color="#4c72b0", alpha=0.8, label="AUC")
        axes[0].barh(surf_metrics["surface"], surf_metrics["acc"], color="#dd8452", alpha=0.6, label="Acc")
        axes[0].set_title("By Surface")
        axes[0].legend()
    else:
        axes[0].set_visible(False)

    if not round_metrics.empty:
        axes[1].barh(round_metrics["round"].astype(str), round_metrics["auc"], color="#4c72b0", alpha=0.8, label="AUC")
        axes[1].barh(round_metrics["round"].astype(str), round_metrics["acc"], color="#dd8452", alpha=0.6, label="Acc")
        axes[1].set_title("By Round")
        axes[1].legend()
    else:
        axes[1].set_visible(False)

    for ax in axes:
        if ax.get_visible():
            ax.grid(True, axis="x", alpha=0.3)
            ax.set_xlabel("Score")

    fig.suptitle(f"Surface/Round Breakdown ({split} set)")
    save_fig(fig, f"surface_round_breakdown_{split}.png")


if __name__ == "__main__":
    generate()
