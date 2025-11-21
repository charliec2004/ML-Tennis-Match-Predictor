"""
Build a cleaned master match file from the raw ATP CSV before feature generation.

Reads data/raw/atp_tennis.csv, normalizes fields, enforces valid domains,
derives helper columns, and writes data/raw/tennis-master-data.csv sorted
chronologically with fresh MATCH_ID values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Input/output locations
DEFAULT_INPUT = Path("data/raw/atp_tennis.csv")
DEFAULT_OUTPUT = Path("data/raw/tennis-master-data.csv")
PLAYERS_DB_PATH = Path("data/raw/players_db.csv")

# Canonical column order expected by the pipeline
OUTPUT_COLUMNS = [
    "MATCH_ID",
    "tournament",
    "date",
    "series",
    "court",
    "surface",
    "player_1",
    "player_2",
    "winner",
    "score",
    "series_level",
    "best_of_3",
    "best_of_5",
    "round",
    "is_outdoor",
    "surf_fast",
    "surf_hard",
    "surf_clay",
    "surf_grass",
    "surf_carpet",
    "rank_1",
    "rank_2",
    "rank_avg",
    "rank_ratio",
    "rank_diff",
    "is_top10_match",
    "target",
]

# Series -> numeric level mapping (mode from existing master file)
SERIES_LEVEL: Dict[str, int] = {
    "Grand Slam": 6,
    "Masters 1000": 5,
    "Masters": 3,
    "Masters Cup": 4,
    "International": 0,
    "International Gold": 1,
    "ATP250": 1,
    "ATP500": 2,
}

# Round text -> ordinal mapping
ROUND_MAP: Dict[str, int] = {
    "Qualifying": 0,
    "Qualification": 0,
    "Round Robin": 0,
    "RR": 0,
    "1st Round": 1,
    "First Round": 1,
    "2nd Round": 2,
    "Second Round": 2,
    "3rd Round": 3,
    "Third Round": 3,
    "4th Round": 4,
    "Fourth Round": 4,
    "Quarterfinals": 5,
    "Quarter-Final": 5,
    "Quarterfinal": 5,
    "Semifinals": 6,
    "Semi-Final": 6,
    "Semifinal": 6,
    "Final": 7,
}


def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and standardize capitalization on key string columns."""
    for col in ["Tournament", "Series", "Court", "Surface", "Round", "Player_1", "Player_2", "Winner", "Score"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    # Normalize capitalization
    df["Series"] = df["Series"].str.title()
    df["Court"] = df["Court"].str.title()
    df["Surface"] = df["Surface"].str.title()
    df["Round"] = df["Round"].str.title()
    return df


def _map_round(text: str) -> int:
    """Convert round text to the numeric code."""
    if pd.isna(text):
        return 0
    normalized = text.strip().title()
    return ROUND_MAP.get(normalized, 0)


def _series_level(series: str) -> int:
    """Map series label to numeric level; unknowns fall back to 0."""
    if pd.isna(series):
        return 0
    normalized = series.strip().title()
    # Preserve uppercase suffix variants like ATP250/ATP500
    if "Atp250" in normalized or normalized == "Atp250":
        normalized = "ATP250"
    if "Atp500" in normalized or normalized == "Atp500":
        normalized = "ATP500"
    return SERIES_LEVEL.get(normalized, 0)


def _surface_flags(surface: str) -> Tuple[int, int, int, int, int]:
    """Return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet flags."""
    surf = (surface or "").strip().title()
    surf_hard = int(surf == "Hard")
    surf_clay = int(surf == "Clay")
    surf_grass = int(surf == "Grass")
    surf_carpet = int(surf == "Carpet")
    surf_fast = int(surf_hard or surf_grass)
    return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet


def build_master(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Transform the raw ATP CSV into the cleaned tennis-master-data.csv."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    df = _clean_strings(df)

    # Basic required columns presence check
    required = {"Tournament", "Date", "Series", "Court", "Surface", "Round", "Best of", "Player_1", "Player_2", "Winner", "Rank_1", "Rank_2", "Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_path}: {sorted(missing)}")

    # Dates to datetime; drop rows with invalid dates
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Derive numeric feature columns
    df["series"] = df["Series"].apply(lambda s: s.strip().title() if isinstance(s, str) else s)
    df["series_level"] = df["Series"].apply(_series_level)
    df["court"] = df["Court"].apply(lambda s: s.title() if isinstance(s, str) else s)
    df["surface"] = df["Surface"].apply(lambda s: s.title() if isinstance(s, str) else s)
    df["round"] = df["Round"].apply(_map_round)
    df["best_of_3"] = (df["Best of"].astype(str).str.strip() == "3").astype(int)
    df["best_of_5"] = (df["Best of"].astype(str).str.strip() == "5").astype(int)
    df["is_outdoor"] = (df["court"].str.lower() == "outdoor").astype(int)

    surf_flags = df["surface"].apply(_surface_flags)
    df[["surf_fast", "surf_hard", "surf_clay", "surf_grass", "surf_carpet"]] = pd.DataFrame(surf_flags.tolist(), index=df.index)

    # Clean names and winner
    df["tournament"] = df["Tournament"]
    df["player_1"] = df["Player_1"]
    df["player_2"] = df["Player_2"]
    df["winner"] = df["Winner"]
    df["score"] = df["Score"]

    # Numeric ranks; -1 or non-positive treated as NaN
    df["rank_1"] = pd.to_numeric(df["Rank_1"], errors="coerce")
    df["rank_2"] = pd.to_numeric(df["Rank_2"], errors="coerce")
    df.loc[df["rank_1"] <= 0, "rank_1"] = pd.NA
    df.loc[df["rank_2"] <= 0, "rank_2"] = pd.NA
    df["rank_avg"] = (df["rank_1"] + df["rank_2"]) / 2
    df["rank_ratio"] = df["rank_1"] / df["rank_2"]
    df["rank_diff"] = df["rank_1"] - df["rank_2"]
    df["is_top10_match"] = ((df["rank_1"] <= 10) & (df["rank_2"] <= 10)).fillna(False).astype(int)

    # Target: 0 if player_1 wins, 1 if player_2 wins
    def _target(row) -> int:
        if row["winner"] == row["player_1"]:
            return 0
        if row["winner"] == row["player_2"]:
            return 1
        return 0  # default to player 1 if ambiguous, but data should not hit this

    df["target"] = df.apply(_target, axis=1)

    # Sort chronologically then by tournament/name to stabilize MATCH_ID, then generate MATCH_ID
    df = df.sort_values(["date", "tournament", "player_1", "player_2"]).reset_index(drop=True)
    df.insert(0, "MATCH_ID", range(1, len(df) + 1))

    # Select and order columns, dropping any extras
    df_out = df[OUTPUT_COLUMNS].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Wrote cleaned master file: {output_path} ({len(df_out):,} rows)")
    return df_out


def update_players_db(df_master: pd.DataFrame, players_db_path: Path = PLAYERS_DB_PATH) -> None:
    """
    Update players_db.csv with any new player names found in the master file.
    Keeps existing names and appends missing ones (case-insensitive).
    """
    players = set(df_master["player_1"]).union(set(df_master["player_2"]))
    players_clean = {p.strip() for p in players if isinstance(p, str) and p.strip()}

    existing = []
    if players_db_path.exists():
        existing_df = pd.read_csv(players_db_path)
        col = existing_df.columns[0]
        existing = [p.strip() for p in existing_df[col].dropna().tolist() if p.strip()]

    existing_lower = {p.lower() for p in existing}
    new_players = sorted(p for p in players_clean if p.lower() not in existing_lower)

    all_players = existing + new_players
    players_df = pd.DataFrame({"player_name": all_players})
    players_db_path.parent.mkdir(parents=True, exist_ok=True)
    players_df.to_csv(players_db_path, index=False)
    print(f"Updated players DB: {players_db_path} (existing {len(existing)}, added {len(new_players)}, total {len(all_players)})")


def main():
    df_master = build_master()
    update_players_db(df_master)


if __name__ == "__main__":
    sys.exit(main())
