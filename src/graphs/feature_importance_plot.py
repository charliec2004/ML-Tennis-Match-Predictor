"""Bar chart of top feature importances."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from graphs.common import save_fig, plt


def generate(top_n: int = 15):
    df = pd.read_csv("data/outputs/feature_importance.csv")
    df = df.sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["feature"], df["importance"], color="#4c72b0")
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Features by Importance")
    ax.set_xlabel("Gain")
    save_fig(fig, "feature_importance.png")


if __name__ == "__main__":
    generate()
