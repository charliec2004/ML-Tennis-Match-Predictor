"""Elo trajectories for selected players using historical master data."""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from pathlib import Path
import pandas as pd

from elo.elo_processor import EloProcessor
from graphs.common import save_fig, plt


def generate(players=None):
    if players is None:
        players = ["Djokovic N.", "Alcaraz C.", "Sinner J.", "Medvedev D."]

    master_path = Path("data/raw/tennis-master-data.csv")
    df = pd.read_csv(master_path, usecols=["date", "player_1", "player_2", "surface", "target"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    elo = EloProcessor()
    history = {p: [] for p in players}

    for _, row in df.iterrows():
        p1, p2, surf, p1_won = row["player_1"], row["player_2"], row["surface"], row["target"] == 0
        # Record ratings before update
        date = row["date"]
        for p in players:
            try:
                r = elo.get_player_ratings(p)[0]
                history[p].append((date, r))
            except KeyError:
                pass
        elo.update_ratings(p1, p2, surf, p1_won)

    fig, ax = plt.subplots(figsize=(10, 6))
    for p, points in history.items():
        if not points:
            continue;
        dates, ratings = zip(*points)
        ax.plot(dates, ratings, label=p)

    ax.set_title("Elo Trajectories (Master Data)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Elo Rating")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_fig(fig, "elo_trajectories.png")


if __name__ == "__main__":
    generate()
