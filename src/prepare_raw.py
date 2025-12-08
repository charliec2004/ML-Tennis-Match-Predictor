"""
Build a cleaned master match file from the latest Kaggle ATP dataset.

Fetches data via Kaggle API, normalizes fields, enforces valid domains,
derives helper columns, and writes data/raw/tennis-master-data.csv sorted
chronologically with fresh MATCH_ID values.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import pandas as pd

try:
    import kagglehub
    from kagglehub import KaggleDatasetAdapter
except ImportError:
    print("Error: kagglehub not installed. Install with: pip install 'kagglehub[pandas-datasets]'")
    sys.exit(1)

# Output locations
DEFAULT_OUTPUT = Path("data/raw/tennis-master-data.csv")
PLAYERS_DB_PATH = Path("data/raw/players_db.csv")

# Kaggle dataset configuration
KAGGLE_DATASET = "dissfya/atp-tennis-2000-2023daily-pull"
KAGGLE_FILE_PATH = "tennis_atp-master/atp_matches_2000.csv"  # Specific file from the dataset

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
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
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


def _map_round(text: str) -> Optional[int]:
    """Convert round text to the numeric code; return None if unknown."""
    if pd.isna(text):
        return None
    raw = text.strip()
    normalized = raw.title()
    raw_lower = raw.lower()
    if raw_lower in {"qf", "quarter", "quarterfinal", "quarter-finals", "quarter-final"}:
        normalized = "Quarterfinal"
    elif raw_lower in {"sf", "semi", "semifinal", "semi-finals", "semi-final"}:
        normalized = "Semifinal"
    elif raw_lower in {"f", "final", "the final"}:
        normalized = "Final"
    elif raw_lower in {"r16", "round of 16", "round-of-16"}:
        normalized = "R16"
    elif raw_lower in {"r32", "round of 32"}:
        normalized = "R32"
    elif raw_lower in {"r64", "round of 64"}:
        normalized = "R64"
    elif raw_lower in {"r128", "round of 128"}:
        normalized = "R128"
    elif raw_lower in {"q1", "q2", "q3", "qualifying"}:
        normalized = "Qualifying"
    elif raw_lower in {"1st round", "1 st round"}:
        normalized = "1st Round"
    elif raw_lower in {"2nd round", "2 nd round"}:
        normalized = "2nd Round"
    elif raw_lower in {"3rd round", "3 rd round"}:
        normalized = "3rd Round"
    elif raw_lower in {"4th round", "4 th round"}:
        normalized = "4th Round"
    return ROUND_MAP.get(normalized)


def _series_level(series: str) -> Optional[int]:
    """Map series label to numeric level; return None if unknown."""
    if pd.isna(series):
        return None
    normalized = series.strip().title()
    # Preserve uppercase suffix variants like ATP250/ATP500
    if "Atp250" in normalized or normalized == "Atp250":
        normalized = "ATP250"
    if "Atp500" in normalized or normalized == "Atp500":
        normalized = "ATP500"
    return SERIES_LEVEL.get(normalized)


def _surface_flags(surface: str) -> Tuple[int, int, int, int, int]:
    """Return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet flags."""
    surf = (surface or "").strip().title()
    surf_hard = int(surf == "Hard")
    surf_clay = int(surf == "Clay")
    surf_grass = int(surf == "Grass")
    surf_carpet = int(surf == "Carpet")
    surf_fast = int(surf_hard or surf_grass)
    return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet


def fetch_latest_dataset() -> pd.DataFrame:
    """Fetch the latest ATP tennis dataset from Kaggle."""
    print("\n" + "="*60)
    print("FETCHING LATEST DATASET FROM KAGGLE")
    print("="*60)
    print(f"Dataset: {KAGGLE_DATASET}")
    print("Downloading latest version...")
    
    try:
        # First, let's download the dataset and see what files are available
        import kagglehub
        
        # Download the dataset to local cache
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"✓ Dataset downloaded to: {path}")
        
        # List all CSV files in the downloaded dataset
        from pathlib import Path
        dataset_path = Path(path)
        csv_files = list(dataset_path.rglob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset")
        
        print(f"\n✓ Found {len(csv_files)} CSV file(s):")
        for i, f in enumerate(csv_files, 1):
            print(f"  {i}. {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        
        # Use the largest CSV file (likely the combined dataset)
        largest_csv = max(csv_files, key=lambda f: f.stat().st_size)
        print(f"\n✓ Loading largest file: {largest_csv.name}")
        
        df = pd.read_csv(largest_csv)
        print(f"✓ Successfully loaded {len(df):,} matches")
        
        # Try to infer date range if 'Date' column exists
        if 'Date' in df.columns:
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
            min_date = df_temp['Date'].min()
            max_date = df_temp['Date'].max()
            if pd.notna(min_date) and pd.notna(max_date):
                print(f"✓ Date range: {min_date.date()} to {max_date.date()}")
        
        return df
    except Exception as e:
        print(f"✗ Error fetching dataset: {e}")
        print("\nMake sure you have:")
        print("  1. Installed kagglehub: pip install 'kagglehub[pandas-datasets]'")
        print("  2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)")
        print("  3. Accepted the dataset terms at: https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull")
        raise


def build_master(output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Transform the raw ATP CSV from Kaggle into the cleaned tennis-master-data.csv."""
    print("\n" + "="*60)
    print("BUILDING MASTER DATA FILE")
    print("="*60)
    
    df = fetch_latest_dataset()
    print("\nCleaning and transforming data...")
    df = _clean_strings(df)

    # Basic required columns presence check
    required = {"Tournament", "Date", "Series", "Court", "Surface", "Round", "Best of", "Player_1", "Player_2", "Winner", "Rank_1", "Rank_2", "Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    print("✓ All required columns present")

    # Dates to datetime; drop rows with invalid dates
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    invalid_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    if invalid_dates > 0:
        print(f"✓ Dropped {invalid_dates:,} rows with invalid dates")
    
    print("✓ Dates validated and parsed")

    # Derive numeric feature columns
    print("\nDeriving features...")
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
    print("✓ Surface flags created")

    # Clean names and winner
    df["tournament"] = df["Tournament"]
    df["player_1"] = df["Player_1"]
    df["player_2"] = df["Player_2"]
    df["winner"] = df["Winner"]
    df["score"] = df["Score"]
    print("✓ Player names and tournament data cleaned")

    # Numeric ranks; -1 or non-positive treated as NaN
    print("\nProcessing rankings...")
    df["rank_1"] = pd.to_numeric(df["Rank_1"], errors="coerce")
    df["rank_2"] = pd.to_numeric(df["Rank_2"], errors="coerce")
    df.loc[df["rank_1"] <= 0, "rank_1"] = pd.NA
    df.loc[df["rank_2"] <= 0, "rank_2"] = pd.NA
    df["rank_avg"] = (df["rank_1"] + df["rank_2"]) / 2
    df["rank_ratio"] = df["rank_1"] / df["rank_2"]
    df["rank_diff"] = df["rank_1"] - df["rank_2"]
    df["is_top10_match"] = ((df["rank_1"] <= 10) & (df["rank_2"] <= 10)).fillna(False).astype(int)
    df["rank_ratio"] = df["rank_ratio"].replace([float("inf"), float("-inf")], pd.NA)
    print("✓ Ranking features derived")

    # Validate known domains for series and round; fail fast on unknowns
    print("\nValidating domains...")
    unknown_series = df[df["series_level"].isna()]["series"].dropna().unique().tolist()
    if unknown_series:
        raise ValueError(
            f"Unknown series values: {unknown_series}. "
            f"Add them to SERIES_LEVEL in src/prepare_raw.py."
        )
    print("✓ All series values recognized")

    unknown_rounds = df[df["round"].isna()]["Round"].dropna().unique().tolist()
    if unknown_rounds:
        raise ValueError(
            f"Unknown round values: {unknown_rounds}. "
            f"Add them to ROUND_MAP in src/prepare_raw.py."
        )
    print("✓ All round values recognized")

    # Target: 0 if player_1 wins, 1 if player_2 wins
    print("\nCreating target variable...")
    def _target(row) -> int:
        if row["winner"] == row["player_1"]:
            return 0
        if row["winner"] == row["player_2"]:
            return 1
        return 0  # default to player 1 if ambiguous, but data should not hit this

    df["target"] = df.apply(_target, axis=1)
    print("✓ Target variable created (0=player_1 wins, 1=player_2 wins)")

    # Sort chronologically then by tournament/name to stabilize MATCH_ID, then generate MATCH_ID
    print("\nFinalizing dataset...")
    df = df.sort_values(["date", "tournament", "player_1", "player_2"]).reset_index(drop=True)
    df.insert(0, "MATCH_ID", range(1, len(df) + 1))
    print("✓ Sorted chronologically and assigned MATCH_IDs")

    # Select and order columns, dropping any extras
    df_out = df[OUTPUT_COLUMNS].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"✓ Master file saved: {output_path}")
    print(f"✓ Total matches: {len(df_out):,}")
    print(f"✓ Date range: {df_out['date'].min().date()} to {df_out['date'].max().date()}")
    print(f"{'='*60}")
    return df_out


def update_players_db(df_master: pd.DataFrame, players_db_path: Path = PLAYERS_DB_PATH) -> None:
    """
    Update players_db.csv with any new player names found in the master file.
    Keeps existing names and appends missing ones (case-insensitive).
    """
    print("\n" + "="*60)
    print("UPDATING PLAYERS DATABASE")
    print("="*60)
    
    players = set(df_master["player_1"]).union(set(df_master["player_2"]))
    players_clean = {p.strip() for p in players if isinstance(p, str) and p.strip()}
    print(f"Found {len(players_clean):,} unique players in dataset")

    existing = []
    if players_db_path.exists():
        existing_df = pd.read_csv(players_db_path)
        col = existing_df.columns[0]
        existing = [p.strip() for p in existing_df[col].dropna().tolist() if p.strip()]
        print(f"Existing players in DB: {len(existing):,}")

    existing_lower = {p.lower() for p in existing}
    new_players = sorted(p for p in players_clean if p.lower() not in existing_lower)

    all_players = existing + new_players
    players_df = pd.DataFrame({"player_name": all_players})
    players_db_path.parent.mkdir(parents=True, exist_ok=True)
    players_df.to_csv(players_db_path, index=False)
    
    print(f"\n✓ Players DB updated: {players_db_path}")
    print(f"  - Previous: {len(existing):,}")
    print(f"  - New: {len(new_players):,}")
    print(f"  - Total: {len(all_players):,}")
    print("="*60)


def main():
    df_master = build_master()
    update_players_db(df_master)


if __name__ == "__main__":
    sys.exit(main())
