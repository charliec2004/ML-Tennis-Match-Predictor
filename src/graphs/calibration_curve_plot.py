"""Reliability curve (calibration)."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from sklearn.calibration import calibration_curve
from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def generate(split: str = "test", n_bins: int = 10):
    booster, feature_names = load_model_and_features()
    X, y, _ = load_split(split)
    proba = predict_split(booster, feature_names, X)
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(mean_pred, frac_pos, marker="o", label=f"{split} set")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfectly calibrated")
    ax.set_title(f"Calibration Curve ({split} set)")
    ax.set_xlabel("Predicted win probability (player_2)")
    ax.set_ylabel("Observed win rate (player_2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, f"calibration_{split}.png")


if __name__ == "__main__":
    generate()
