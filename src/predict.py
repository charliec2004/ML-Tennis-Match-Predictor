"""
Tennis match prediction inference module.

Loads trained XGBoost model and generates predictions for new matches.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import sys
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from features import generate_features
from elo.elo_processor import EloProcessor
from matches.match_history_processor import MatchHistoryProcessor

# Lightweight copies of mapping helpers to avoid pulling full prepare_raw (which requires kagglehub)
SERIES_LEVEL = {
    "Grand Slam": 6,
    "Masters 1000": 5,
    "Masters": 3,
    "Masters Cup": 4,
    "ATP Next Gen": 2,
    "International": 0,
    "International Gold": 1,
    "ATP250": 1,
    "ATP500": 2,
}

ROUND_MAP = {
    "Qualifying": 0,
    "Qualification": 0,
    "Round Robin": 0,
    "Rr": 0,
    "Group Stage": 0,
    "Group": 0,
    "1st Round": 1,
    "First Round": 1,
    "2nd Round": 2,
    "Second Round": 2,
    "3rd Round": 3,
    "Third Round": 3,
    "4th Round": 4,
    "Fourth Round": 4,
    "R16": 5,
    "Round Of 16": 5,
    "Round-Of-16": 5,
    "R32": 4,
    "Round Of 32": 4,
    "R64": 3,
    "Round Of 64": 3,
    "R128": 2,
    "Round Of 128": 2,
    "Q1": 1,
    "Q2": 1,
    "Q3": 1,
    "Quarterfinal": 6,
    "Semi-Final": 7,
    "Semi": 7,
    "Semifinal": 7,
    "Semi-Finals": 7,
    "Semi-Final": 7,
    "Sf": 7,
    "Final": 8,
    "F": 8,
}


def _series_level(series: str) -> Optional[int]:
    """Map series label to numeric level; return None if unknown."""
    if pd.isna(series):
        return None
    normalized = series.strip().title()
    if normalized.lower().startswith("atp"):
        normalized = "ATP" + normalized[3:]
    if "Atp250" in normalized or normalized == "Atp250":
        normalized = "ATP250"
    if "Atp500" in normalized or normalized == "Atp500":
        normalized = "ATP500"
    return SERIES_LEVEL.get(normalized)


def _map_round(raw_round: str) -> Optional[int]:
    """Map round labels to an ordered numeric scale used in training."""
    if pd.isna(raw_round):
        return None
    raw = str(raw_round).strip()
    if not raw:
        return None
    normalized = raw.title()
    if normalized.lower() in {"sf", "semi", "semifinal", "semi-finals", "semi-final"}:
        normalized = "Semifinal"
    elif normalized.lower() in {"qf", "quarterfinal", "quarter-final"}:
        normalized = "Quarterfinal"
    elif normalized.lower() in {"r16", "round of 16", "round-of-16"}:
        normalized = "R16"
    elif normalized.lower() in {"r32", "round of 32"}:
        normalized = "R32"
    elif normalized.lower() in {"r64", "round of 64"}:
        normalized = "R64"
    elif normalized.lower() in {"r128", "round of 128"}:
        normalized = "R128"
    elif normalized.lower() in {"q1", "q2", "q3", "qualifying"}:
        normalized = "Qualifying"
    elif normalized.lower() in {"1st round", "1 st round"}:
        normalized = "1st Round"
    elif normalized.lower() in {"2nd round", "2 nd round"}:
        normalized = "2nd Round"
    elif normalized.lower() in {"3rd round", "3 rd round"}:
        normalized = "3rd Round"
    elif normalized.lower() in {"4th round", "4 th round"}:
        normalized = "4th Round"
    return ROUND_MAP.get(normalized)


def _surface_flags(surface: str) -> Tuple[int, int, int, int, int]:
    """Return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet flags."""
    surf = (surface or "").strip().title()
    surf_hard = int(surf == "Hard")
    surf_clay = int(surf == "Clay")
    surf_grass = int(surf == "Grass")
    surf_carpet = int(surf == "Carpet")
    surf_fast = int(surf_hard or surf_grass)
    return surf_fast, surf_hard, surf_clay, surf_grass, surf_carpet


def load_trained_model(model_path: str = "data/outputs/model_xgb.json") -> Tuple[xgb.Booster, List[str]]:
    """Load the trained XGBoost model and feature names."""
    print("Loading trained model...")
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    feature_names_path = Path(model_path).parent / "feature_names.json"
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    
    print(f"   Model loaded from {model_path}")
    print(f"   Features: {len(feature_names)} columns")
    
    return model, feature_names


def load_known_players(players_db: Path = Path("data/raw/players_db.csv")) -> List[str]:
    """Load known player names from players_db.csv."""
    if not players_db.exists():
        return []
    df = pd.read_csv(players_db)
    col = df.columns[0]
    return [p.strip() for p in df[col].dropna() if str(p).strip()]


def resolve_player_name(name: str, known: List[str]) -> str:
    """Resolve an unknown player name by asking the user to pick a closest match."""
    if not known:
        return name
    # Exact (case-insensitive) match
    for k in known:
        if k.lower() == name.lower():
            return k
    # Suggest closest names
    suggestions = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    if not sys.stdin.isatty():
        # Non-interactive: fall back to original to avoid blocking
        return name
    if suggestions:
        print(f"Unknown player '{name}'. Did you mean:")
        for idx, s in enumerate(suggestions, 1):
            print(f"   {idx}. {s}")
        print("   0. Keep as-is (may fail if truly unknown)")
        while True:
            choice = input("Select a number: ").strip()
            if choice.isdigit():
                choice_int = int(choice)
                if choice_int == 0:
                    return name
                if 1 <= choice_int <= len(suggestions):
                    chosen = suggestions[choice_int - 1]
                    if chosen != name:
                        print(f"   Using '{chosen}' instead of '{name}'")
                    return chosen
            print("Please enter 0 or a valid suggestion number.")
    return name


def prepare_match_data(matches_df: pd.DataFrame, interactive_resolution: bool = True) -> pd.DataFrame:
    """
    Prepare new match data for prediction by generating features.
    
    Args:
        matches_df: DataFrame with columns ['date', 'player_1', 'player_2', 'surface', etc.]
                   Must have same format as your original raw data
    
    Returns:
        DataFrame with all features needed for prediction
    """
    print("Preparing match data for prediction...")
    
    required_cols = ['date', 'player_1', 'player_2', 'surface']
    missing_cols = [col for col in required_cols if col not in matches_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    matches_prep = matches_df.copy()

    # Resolve unknown player names with suggestions
    if interactive_resolution:
        known_players = load_known_players()
        for col in ["player_1", "player_2"]:
            matches_prep[col] = matches_prep[col].apply(lambda n: resolve_player_name(str(n), known_players))

    # Normalize categorical inputs so they mirror training-time preprocessing
    matches_prep["surface"] = matches_prep["surface"].apply(lambda s: s.title() if isinstance(s, str) else s)
    matches_prep["series"] = matches_prep.get("series", pd.Series([None] * len(matches_prep))).apply(
        lambda s: s.strip().title() if isinstance(s, str) else s
    )
    matches_prep["series_level"] = matches_prep["series"].apply(_series_level)

    matches_prep["round"] = matches_prep.get("round", pd.Series([None] * len(matches_prep))).apply(_map_round)

    if "best_of" not in matches_prep:
        matches_prep["best_of"] = 3  # default to best-of-3 if unspecified
    matches_prep["best_of_3"] = (matches_prep["best_of"].astype(str).str.strip() == "3").astype(int)
    matches_prep["best_of_5"] = (matches_prep["best_of"].astype(str).str.strip() == "5").astype(int)

    matches_prep["court"] = matches_prep.get("court", pd.Series(["Outdoor"] * len(matches_prep))).apply(
        lambda s: s.title() if isinstance(s, str) else s
    )
    matches_prep["is_outdoor"] = (matches_prep["court"].str.lower() == "outdoor").astype(int)

    surf_flags = matches_prep["surface"].apply(_surface_flags)
    matches_prep[["surf_fast", "surf_hard", "surf_clay", "surf_grass", "surf_carpet"]] = pd.DataFrame(
        surf_flags.tolist(), index=matches_prep.index
    )

    # Ranking-derived features (robust to missing rankings)
    matches_prep["rank_1"] = pd.to_numeric(matches_prep.get("rank_1", np.nan), errors="coerce")
    matches_prep["rank_2"] = pd.to_numeric(matches_prep.get("rank_2", np.nan), errors="coerce")
    matches_prep.loc[matches_prep["rank_1"] <= 0, "rank_1"] = np.nan
    matches_prep.loc[matches_prep["rank_2"] <= 0, "rank_2"] = np.nan
    matches_prep["rank_avg"] = (matches_prep["rank_1"] + matches_prep["rank_2"]) / 2
    matches_prep["rank_ratio"] = matches_prep["rank_1"] / matches_prep["rank_2"]
    matches_prep["rank_diff"] = matches_prep["rank_1"] - matches_prep["rank_2"]
    matches_prep["rank_ratio"] = matches_prep["rank_ratio"].replace([np.inf, -np.inf], np.nan)
    matches_prep["is_top10_match"] = (
        (matches_prep["rank_1"] <= 10) & (matches_prep["rank_2"] <= 10)
    ).fillna(False).astype(int)
    
    if 'winner' not in matches_prep.columns:
        matches_prep['winner'] = matches_prep['player_1']
    if 'target' not in matches_prep.columns:
        matches_prep['target'] = 0
    
    matches_prep['date'] = pd.to_datetime(matches_prep['date'])
    
    # Seed historical state so features reflect real past performance
    history_path = Path("data/raw/tennis-master-data.csv")
    elo_proc = EloProcessor()
    history_proc = MatchHistoryProcessor()
    if history_path.exists():
        min_future_date = pd.to_datetime(matches_prep["date"]).min()
        print(f"   Loading historical matches from {history_path} to seed ratings...")
        hist_df = pd.read_csv(history_path, usecols=[
            "date",
            "player_1",
            "player_2",
            "surface",
            "target",
        ])
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df = hist_df[hist_df["date"] < min_future_date].sort_values("date")
        if hist_df.empty:
            print("   Warning: no historical rows before future match dates; seeding skipped")
        else:
            for _, row in hist_df.iterrows():
                elo_proc.update_ratings(row["player_1"], row["player_2"], row["surface"], row["target"] == 0)
                history_proc.update_match_history(
                    row["player_1"],
                    row["player_2"],
                    row["date"].date(),
                    row["target"] == 0,
                )
            print(f"   Seeded ratings from {len(hist_df):,} historical matches (< {min_future_date.date()})")
    else:
        print("   Warning: historical ratings not seeded (data/raw/tennis-master-data.csv missing)")

    print("   Generating ELO and match history features...")
    # For inference do not drop rows and do not let future rows mutate state
    features_df = generate_features(
        matches_prep,
        warmup_matches=0,
        min_player_matches=0,
        elo_processor=elo_proc,
        history_processor=history_proc,
        update_state=False,
    )
    
    print(f"   Features generated for {len(features_df)} matches")
    return features_df


def make_predictions(features_df: pd.DataFrame, model_path: str = "data/outputs/model_xgb.json") -> pd.DataFrame:
    """Generate predictions for processed match features."""
    try:
        model, feature_names = load_trained_model(model_path)
        
        print("Generating predictions...")
        
        X_features = features_df[feature_names].copy()
        
        print(f"   Feature matrix shape: {X_features.shape}")
        print(f"   Expected: ({len(features_df)}, {len(feature_names)})")
        
        if X_features.isnull().any().any():
            missing_counts = X_features.isnull().sum()
            missing_cols = missing_counts[missing_counts > 0]
            total_missing = int(missing_counts.sum())
            print(f"   NaN values found ({total_missing} total) across {len(missing_cols)} columns:")
            print(f"      {missing_cols.to_dict()}")
            print("   Imputing with median...")
            all_nan_cols = X_features.columns[X_features.isnull().all()]
            medians = X_features.median(numeric_only=True)
            X_features_imputed = X_features.copy()
            if len(all_nan_cols) > 0:
                print(f"   Filling all-NaN columns with 0: {list(all_nan_cols)}")
                X_features_imputed[all_nan_cols] = 0
            X_features_imputed = X_features_imputed.fillna(medians)
        else:
            X_features_imputed = X_features
        
        print(f"   Final feature matrix shape: {X_features_imputed.shape}")
        assert X_features_imputed.shape[1] == len(feature_names), f"Shape mismatch: {X_features_imputed.shape[1]} vs {len(feature_names)}"
        
        # Use both-ways prediction to remove position bias
        print("   Making predictions (averaging both player orderings)...")
        from data_augmentation import predict_both_ways
        
        # Create a simple wrapper that has predict_proba method for compatibility
        class XGBoostWrapper:
            def __init__(self, booster, feature_names):
                self.booster = booster
                self.feature_names = feature_names
            
            def predict_proba(self, X):
                dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
                preds = self.booster.predict(dmatrix)
                # Return shape (n_samples, 2) for binary classification
                return np.column_stack([1 - preds, preds])
        
        model_wrapper = XGBoostWrapper(model, feature_names)
        predictions = predict_both_ways(model_wrapper, X_features_imputed, feature_names)
        
        results_df = features_df[['date', 'player_1', 'player_2']].copy()
        results_df['prob_p1_wins'] = 1 - predictions  # predictions is P(player_2 wins)
        results_df['prob_p2_wins'] = predictions
        # If P(player_2 wins) > 0.5, pick player_2; otherwise player_1
        results_df['predicted_winner'] = np.where(predictions > 0.5, results_df['player_2'], results_df['player_1'])
        results_df['confidence'] = np.maximum(predictions, 1 - predictions)
        
        print(f"   Predictions generated for {len(results_df)} matches")
        
        return results_df
        
    except Exception as e:
        print(f"   Error during prediction: {str(e)}")
        print("   Check your CSV file format and try again.")
        raise


def predict_matches(matches_df: pd.DataFrame, 
                   model_path: str = "data/outputs/model_xgb.json",
                   save_results: bool = True,
                   interactive_resolution: bool = True) -> pd.DataFrame:
    """
    Complete prediction pipeline for new matches.
    
    Args:
        matches_df: DataFrame with match data to predict
        model_path: Path to trained model
        save_results: Whether to save results to CSV
    
    Returns:
        DataFrame with predictions
    """
    print("Tennis Match Prediction - Inference Pipeline")
    print("=" * 60)
    
    model, feature_names = load_trained_model(model_path)
    
    features_df = prepare_match_data(matches_df, interactive_resolution=interactive_resolution)
    
    predictions_df = make_predictions(features_df, model_path)
    
    if save_results:
        output_path = Path("data/outputs/predictions.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        print(f"   Predictions saved to {output_path}")
    
    print(f"\nPREDICTION SUMMARY:")
    print(f"   Total matches: {len(predictions_df)}")
    print(f"   High confidence (>70%): {len(predictions_df[predictions_df['confidence'] > 0.7])}")
    print(f"   Medium confidence (60-70%): {len(predictions_df[(predictions_df['confidence'] >= 0.6) & (predictions_df['confidence'] <= 0.7)])}")
    print(f"   Low confidence (<60%): {len(predictions_df[predictions_df['confidence'] < 0.6])}")
    
    return predictions_df


def predict_from_csv(input_csv: str, model_path: str = "data/outputs/model_xgb.json", interactive_resolution: bool = True) -> pd.DataFrame:
    """
    Predict matches from a CSV file.
    
    Args:
        input_csv: Path to CSV with match data
        model_path: Path to trained model
    
    Returns:
        DataFrame with predictions
    """
    print(f"Loading matches from {input_csv}...")
    matches_df = pd.read_csv(input_csv)
    print(f"   Loaded {len(matches_df)} matches")
    
    return predict_matches(matches_df, model_path, interactive_resolution=interactive_resolution)


def create_example_matches() -> pd.DataFrame:
    """Create example future matches for testing."""
    example_matches = pd.DataFrame({
        'date': ['2024-01-15', '2024-01-15', '2024-01-16'],
        'player_1': ['Djokovic N.', 'Federer R.', 'Nadal R.'],
        'player_2': ['Alcaraz C.', 'Murray A.', 'Tsitsipas S.'],
        'surface': ['Hard', 'Hard', 'Clay'],
        'tournament': ['Australian Open', 'Australian Open', 'Example Tournament'],
        'round': ['QF', 'SF', 'Final'],
        'best_of': [5, 5, 3]
    })
    return example_matches


if __name__ == "__main__":
    """Example usage of the prediction module."""
    
    print("Testing with example matches...")
    example_df = create_example_matches()
    predictions = predict_matches(example_df)
    
    print("\nExample Predictions:")
    for _, row in predictions.iterrows():
        print(f"   {row['player_1']} vs {row['player_2']}")
        print(f"   Predicted winner: {row['predicted_winner']} ({row['confidence']:.1%} confidence)")
        print(f"   Surface: {row['surface']} | Date: {row['date']}")
        print()
    
    print("To predict from your own CSV file:")
    print("   predictions = predict_from_csv('path/to/your/matches.csv')")
    print("\nYour CSV should have columns: date, player_1, player_2, surface")
    print("   Optional columns: tournament, round, best_of, etc.")
