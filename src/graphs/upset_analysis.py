"""Upset analysis: predicted probability vs actual outcome on test set."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import pandas as pd
from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def generate(split: str = "test", top_n: int = 10):
    booster, feature_names = load_model_and_features()
    X, y, meta = load_split(split)
    proba = predict_split(booster, feature_names, X)

    df = meta[["player_1", "player_2", "date"]].copy()
    df["proba_p2"] = proba
    df["target"] = y.values
    df["upset"] = ((df["proba_p2"] > 0.5) & (df["target"] == 0)) | ((df["proba_p2"] < 0.5) & (df["target"] == 1))
    df["confidence"] = np.maximum(df["proba_p2"], 1 - df["proba_p2"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["proba_p2"], df["target"], alpha=0.2, s=10, label="All")

    upsets = df[df["upset"]].sort_values("confidence", ascending=False).head(top_n)
    ax.scatter(upsets["proba_p2"], upsets["target"], color="red", edgecolor="black", s=40, label="Top upsets")
    for _, row in upsets.iterrows():
        label = f"{row['player_1']} vs {row['player_2']}"
        ax.annotate(label, (row["proba_p2"], row["target"]), fontsize=6)

    ax.set_title(f"Upset Analysis ({split} set)")
    ax.set_xlabel("Predicted P(player_2 wins)")
    ax.set_ylabel("Actual outcome (1=player_2 wins)")
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, f"upset_analysis_{split}.png")


if __name__ == "__main__":
    generate()
