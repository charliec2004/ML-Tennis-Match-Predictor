"""
Time-based splitter for tennis match prediction dataset.

Splits feature-rich data into train/validation/test sets based on date cutoffs,
maintaining temporal order and ensuring no data leakage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import os


def split_columns(
    df: pd.DataFrame, 
    meta_cols: List[str], 
    y_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Split DataFrame columns into meta, X (features + MATCH_ID), and y groups.
    
    Args:
        df: Input DataFrame with all columns
        meta_cols: List of metadata column names
        y_col: Target column name
        
    Returns:
        Tuple of (meta_df, X_df_with_MATCH_ID, y_df_with_MATCH_ID, x_feature_names_without_MATCH_ID)
    """
    missing_meta = [col for col in meta_cols if col not in df.columns]
    if missing_meta:
        raise ValueError(f"Missing meta columns: {missing_meta}")
    
    if y_col not in df.columns:
        raise ValueError(f"Missing target column: {y_col}")
    
    if "MATCH_ID" not in df.columns:
        raise ValueError("Missing MATCH_ID column")
    
    meta_df = df[meta_cols].copy()
    
    y_df = df[["MATCH_ID", y_col]].copy()
    
    excluded_cols = set(meta_cols) | {y_col}
    x_feature_names = [col for col in df.columns if col not in excluded_cols and col != "MATCH_ID"]
    
    X_df = df[["MATCH_ID"] + x_feature_names].copy()
    
    return meta_df, X_df, y_df, x_feature_names


def single_cutoff_masks(
    df: pd.DataFrame, 
    date_col: str, 
    train_end: str, 
    val_end: str
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Create boolean masks for train/validation/test splits based on date cutoffs.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        train_end: End date for training set (inclusive, format: "YYYY-MM-DD")
        val_end: End date for validation set (inclusive, format: "YYYY-MM-DD")
        
    Returns:
        Tuple of (train_mask, val_mask, test_mask) boolean Series
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    dates = pd.to_datetime(df[date_col])
    
    train_mask = dates <= train_end_dt
    val_mask = (dates > train_end_dt) & (dates <= val_end_dt)
    test_mask = dates > val_end_dt
    
    assert not (train_mask & val_mask).any(), "Train and validation masks overlap"
    assert not (train_mask & test_mask).any(), "Train and test masks overlap"
    assert not (val_mask & test_mask).any(), "Validation and test masks overlap"
    assert (train_mask | val_mask | test_mask).all(), "Masks don't cover all rows"
    
    return train_mask, val_mask, test_mask


def make_splits(
    df: pd.DataFrame,
    date_col: str = "date", 
    meta_cols: Optional[List[str]] = None,
    y_col: str = "target",
    train_end: str = "2018-12-31",
    val_end: str = "2022-12-31"
) -> Dict[str, Any]:
    """
    Orchestrate the complete splitting process: sort, split columns, create masks, slice.
    
    Args:
        df: Input DataFrame with all features
        date_col: Name of the date column
        meta_cols: List of metadata columns (optional)
        y_col: Target column name
        train_end: End date for training (inclusive)
        val_end: End date for validation (inclusive)
        
    Returns:
        Dictionary with train/val/test splits and feature column names
    """
    if meta_cols is None:
        meta_cols = [
            "MATCH_ID", "tournament", "date", "series", "court", 
            "surface", "player_1", "player_2", "winner", "score"
        ]
    
    print("Creating time-based splits...")
    print(f"   Train: up to {train_end}")
    print(f"   Valid: {train_end} < date <= {val_end}")
    print(f"   Test:  after {val_end}")
    
    df_sorted = df.sort_values([date_col, "MATCH_ID"], ascending=True).reset_index(drop=True)
    print(f"   Sorted {len(df_sorted):,} matches chronologically")
    
    meta_df, X_df, y_df, x_feature_names = split_columns(df_sorted, meta_cols, y_col)
    print(f"   Split into {len(x_feature_names)} features (excluding MATCH_ID)")
    
    train_mask, val_mask, test_mask = single_cutoff_masks(df_sorted, date_col, train_end, val_end)
    
    splits = {
        "train": {
            "meta": meta_df[train_mask].reset_index(drop=True),
            "X": X_df[train_mask].reset_index(drop=True),
            "y": y_df[train_mask].reset_index(drop=True)
        },
        "val": {
            "meta": meta_df[val_mask].reset_index(drop=True),
            "X": X_df[val_mask].reset_index(drop=True),
            "y": y_df[val_mask].reset_index(drop=True)
        },
        "test": {
            "meta": meta_df[test_mask].reset_index(drop=True),
            "X": X_df[test_mask].reset_index(drop=True),
            "y": y_df[test_mask].reset_index(drop=True)
        },
        "x_cols": x_feature_names
    }
    
    for split_name in ["train", "val", "test"]:
        split_data = splits[split_name]
        n_meta = len(split_data["meta"])
        n_X = len(split_data["X"])
        n_y = len(split_data["y"])
        
        assert n_meta == n_X == n_y, f"Row count mismatch in {split_name}: meta={n_meta}, X={n_X}, y={n_y}"
        
        expected_X_cols = ["MATCH_ID"] + x_feature_names
        actual_X_cols = list(split_data["X"].columns)
        assert actual_X_cols == expected_X_cols, f"X columns mismatch in {split_name}"
        
        expected_y_cols = ["MATCH_ID", y_col]
        actual_y_cols = list(split_data["y"].columns)
        assert actual_y_cols == expected_y_cols, f"y columns mismatch in {split_name}"
    
    print(f"   Train: {len(splits['train']['X']):,} matches")
    print(f"   Valid: {len(splits['val']['X']):,} matches")
    print(f"   Test:  {len(splits['test']['X']):,} matches")
    print(f"   Total: {len(splits['train']['X']) + len(splits['val']['X']) + len(splits['test']['X']):,} matches")
    
    return splits


def save_splits(splits: Dict[str, Any], out_dir: str) -> None:
    """
    Save train/val/test splits to CSV files.
    
    Args:
        splits: Dictionary returned by make_splits()
        out_dir: Output directory path
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving splits to {out_dir}...")
    
    x_cols = splits["x_cols"]
    
    for split_name in ["train", "val", "test"]:
        split_data = splits[split_name]
        
        meta_file = out_path / f"meta_{split_name}.csv"
        split_data["meta"].to_csv(meta_file, index=False)
        
        X_file = out_path / f"X_{split_name}.csv"
        X_to_save = split_data["X"][["MATCH_ID"] + x_cols]
        X_to_save.to_csv(X_file, index=False)
        
        y_file = out_path / f"y_{split_name}.csv"
        split_data["y"].to_csv(y_file, index=False)
        
        print(f"   {split_name}: {len(split_data['X']):,} rows -> {meta_file.name}, {X_file.name}, {y_file.name}")
    
    feature_file = out_path / "feature_names.txt"
    with open(feature_file, 'w') as f:
        f.write("# X Feature columns (excluding MATCH_ID)\n")
        for col in x_cols:
            f.write(f"{col}\n")
    
    print(f"   Feature names: {len(x_cols)} features -> {feature_file.name}")
    print(f"   All splits saved to {out_dir}")


if __name__ == "__main__":
    """
    Example usage: Load processed data and create time-based splits.
    """
    
    data_path = "data/processed/with_features.csv"
    output_dir = "data/processed/splits"
    
    print("Tennis Match Prediction - Time-Based Data Splitting")
    print("=" * 60)
    
    try:
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, parse_dates=["date"])
        print(f"   Loaded {len(df):,} matches from {df['date'].min().date()} to {df['date'].max().date()}")
        
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        splits = make_splits(df)
        
        save_splits(splits, output_dir)
        
        print(f"\nSPLIT SUMMARY:")
        print(f"   Features: {len(splits['x_cols'])} columns")
        print(f"   Train dates: {splits['train']['meta']['date'].min()} to {splits['train']['meta']['date'].max()}")
        print(f"   Valid dates: {splits['val']['meta']['date'].min()} to {splits['val']['meta']['date'].max()}")
        print(f"   Test dates:  {splits['test']['meta']['date'].min()} to {splits['test']['meta']['date'].max()}")
        
        print(f"\nFEATURE PREVIEW (first 10 of {len(splits['x_cols'])}):")
        for i, col in enumerate(splits['x_cols'][:10]):
            print(f"   {i+1:2d}. {col}")
        if len(splits['x_cols']) > 10:
            print(f"   ... and {len(splits['x_cols']) - 10} more")
        
        print(f"\nSUCCESS: Splits ready for model training!")
        print(f"   Next: Use X_train.csv + y_train.csv for XGBoost training")
        
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}")
        print("   Please run your feature generation first to create the processed dataset.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
