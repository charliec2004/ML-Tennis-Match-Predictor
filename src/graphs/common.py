import json
import os
import sys
from pathlib import Path
from typing import Tuple, Dict

# Ensure project root on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[2]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure matplotlib can write cache files in sandboxed environments
cache_dir = Path("data/outputs/mplconfig")
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/outputs/graphs")


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def load_model_and_features(model_path: Path = Path("data/outputs/model_xgb.json")) -> Tuple[xgb.Booster, list]:
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(model_path.parent / "feature_names.json", "r") as f:
        feature_names = json.load(f)
    return booster, feature_names


def load_split(name: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X, y (Series), and meta for split name in data/processed/splits."""
    base = Path("data/processed/splits")
    X = pd.read_csv(base / f"X_{name}.csv")
    y_df = pd.read_csv(base / f"y_{name}.csv")
    y = y_df["target"]
    meta = pd.read_csv(base / f"meta_{name}.csv")
    return X, y, meta


def predict_split(booster: xgb.Booster, feature_names: list, X: pd.DataFrame) -> np.ndarray:
    """Predict probability player_2 wins for given feature matrix."""
    dmat = xgb.DMatrix(X[feature_names], feature_names=feature_names)
    return booster.predict(dmat)


def save_fig(fig: plt.Figure, filename: str):
    out_dir = ensure_output_dir()
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {path}")
