"""Prediction confidence distribution."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import numpy as np
from graphs.common import load_model_and_features, load_split, predict_split, save_fig, plt


def generate(split: str = "test"):
    booster, feature_names = load_model_and_features()
    X, _, _ = load_split(split)
    proba = predict_split(booster, feature_names, X)
    confidence = np.maximum(proba, 1 - proba)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(confidence, bins=20, color="#55a868", edgecolor="white")
    ax.set_title(f"Prediction Confidence Distribution ({split} set)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    save_fig(fig, f"confidence_hist_{split}.png")


if __name__ == "__main__":
    generate()
