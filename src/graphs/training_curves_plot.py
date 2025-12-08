"""Plot training curves from saved evals_result.json."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import json
from pathlib import Path

from graphs.common import ensure_output_dir, plt


def generate():
    evals_path = Path("data/outputs/evals_result.json")
    if not evals_path.exists():
        raise FileNotFoundError(f"{evals_path} not found. Run training first.")
    with open(evals_path, "r") as f:
        evals_result = json.load(f)

    train_logloss = evals_result.get("train", {}).get("logloss", [])
    val_logloss = evals_result.get("eval", {}).get("logloss", [])

    if not train_logloss or not val_logloss:
        raise ValueError("evals_result.json missing logloss entries.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_logloss, label='Training Log Loss', color='blue')
    ax.plot(val_logloss, label='Validation Log Loss', color='red')
    ax.set_xlabel('Boosting Rounds')
    ax.set_ylabel('Log Loss')
    ax.set_title('XGBoost Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = ensure_output_dir()
    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    generate()
