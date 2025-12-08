"""
Data augmentation to remove player position bias.

Randomly swaps player_1 and player_2 positions during training to ensure
the model learns position-invariant features.
"""

import pandas as pd
import numpy as np


def swap_player_columns(df: pd.DataFrame, swap_mask: pd.Series) -> pd.DataFrame:
    """
    Swap player_1 and player_2 columns for rows where swap_mask is True.
    
    Args:
        df: DataFrame with player features
        swap_mask: Boolean series indicating which rows to swap
        
    Returns:
        DataFrame with swapped columns
    """
    df_swapped = df.copy()
    
    # Columns that need swapping (all _p1 <-> _p2 pairs)
    p1_cols = [col for col in df.columns if col.endswith('_p1')]
    p2_cols = [col for col in df.columns if col.endswith('_p2')]
    
    # Also swap rank columns
    if 'rank_1' in df.columns and 'rank_2' in df.columns:
        p1_cols.extend(['rank_1'])
        p2_cols.extend(['rank_2'])
    
    # Swap the columns where mask is True
    for p1_col, p2_col in zip(p1_cols, p2_cols):
        if p1_col in df.columns and p2_col in df.columns:
            df_swapped.loc[swap_mask, [p1_col, p2_col]] = df.loc[swap_mask, [p2_col, p1_col]].values
    
    # Swap derived features that depend on player order
    if 'elo_diff' in df.columns:
        df_swapped.loc[swap_mask, 'elo_diff'] = -df.loc[swap_mask, 'elo_diff']
    
    if 'rank_diff' in df.columns:
        df_swapped.loc[swap_mask, 'rank_diff'] = -df.loc[swap_mask, 'rank_diff']
    
    if 'rank_ratio' in df.columns:
        # rank_ratio = rank_1 / rank_2, so swapped becomes rank_2 / rank_1 = 1 / old_ratio
        df_swapped.loc[swap_mask, 'rank_ratio'] = 1.0 / df.loc[swap_mask, 'rank_ratio']
    
    if 'h2h_win_rate_p1' in df.columns:
        # h2h is already relative, flip it: if p1 won 0.6, swapped p1 (old p2) won 0.4
        df_swapped.loc[swap_mask, 'h2h_win_rate_p1'] = 1.0 - df.loc[swap_mask, 'h2h_win_rate_p1']
    
    # Flip target: if player_1 won (0), after swap player_1 (old player_2) lost, so target=1
    if 'target' in df.columns:
        df_swapped.loc[swap_mask, 'target'] = 1 - df.loc[swap_mask, 'target']
    
    return df_swapped


def augment_training_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Augment training data by randomly swapping 50% of matches.
    
    This removes position bias and effectively doubles the training data
    by teaching the model that player position doesn't matter.
    
    Args:
        df: DataFrame with features
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with randomly swapped player positions
    """
    np.random.seed(seed)
    
    # Randomly select 50% of rows to swap
    swap_mask = pd.Series(
        np.random.random(len(df)) < 0.5,
        index=df.index
    )
    
    df_augmented = swap_player_columns(df, swap_mask)
    
    swapped_count = swap_mask.sum()
    print(f"   Augmented data: swapped {swapped_count:,} of {len(df):,} matches ({swapped_count/len(df)*100:.1f}%)")
    
    return df_augmented


def predict_both_ways(model, X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """
    Make predictions in both player orderings and average the probabilities.
    
    This gives more robust predictions by removing position bias at inference time.
    
    Args:
        model: Trained model with predict_proba method
        X: Feature matrix
        feature_names: List of feature names to ensure correct ordering
        
    Returns:
        Array of averaged probabilities for player_2 winning
    """
    # Original prediction (P(player_2 wins))
    X_original = X[feature_names]
    proba_original = model.predict_proba(X_original)[:, 1]
    
    # Swapped prediction
    swap_mask = pd.Series([True] * len(X), index=X.index)
    X_swapped_df = swap_player_columns(X.copy(), swap_mask)
    X_swapped = X_swapped_df[feature_names]
    
    # After swapping, model predicts P(swapped_player_2 wins) = P(original_player_1 wins)
    # So we need 1 - proba to get P(original_player_2 wins)
    proba_swapped = 1 - model.predict_proba(X_swapped)[:, 1]
    
    # Average both predictions
    proba_avg = (proba_original + proba_swapped) / 2
    
    return proba_avg
