"""
Tennis match prediction inference module.

Loads trained XGBoost model and generates predictions for new matches.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from features import generate_features


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


def prepare_match_data(matches_df: pd.DataFrame) -> pd.DataFrame:
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
    
    if 'winner' not in matches_prep.columns:
        matches_prep['winner'] = matches_prep['player_1']
    if 'target' not in matches_prep.columns:
        matches_prep['target'] = 0
    
    matches_prep['date'] = pd.to_datetime(matches_prep['date'])
    
    print("   Generating ELO and match history features...")
    features_df = generate_features(matches_prep)
    
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
            print("   NaN values found, imputing with median...")
            from sklearn.impute import SimpleImputer
            
            all_nan_cols = X_features.columns[X_features.isnull().all()]
            if len(all_nan_cols) > 0:
                print(f"   Filling all-NaN columns with 0: {list(all_nan_cols)}")
                X_features[all_nan_cols] = 0
            
            imputer = SimpleImputer(strategy='median')
            X_features_imputed = pd.DataFrame(
                imputer.fit_transform(X_features), 
                columns=X_features.columns, 
                index=X_features.index
            )
        else:
            X_features_imputed = X_features
        
        print(f"   Final feature matrix shape: {X_features_imputed.shape}")
        assert X_features_imputed.shape[1] == len(feature_names), f"Shape mismatch: {X_features_imputed.shape[1]} vs {len(feature_names)}"
        
        dmatrix = xgb.DMatrix(X_features_imputed, feature_names=feature_names)
        predictions = model.predict(dmatrix)
        
        results_df = features_df[['date', 'player_1', 'player_2']].copy()
        results_df['prob_p1_wins'] = predictions
        results_df['prob_p2_wins'] = 1 - predictions
        results_df['predicted_winner'] = np.where(predictions > 0.5, results_df['player_1'], results_df['player_2'])
        results_df['confidence'] = np.maximum(predictions, 1 - predictions)
        
        print(f"   Predictions generated for {len(results_df)} matches")
        
        return results_df
        
    except Exception as e:
        print(f"   Error during prediction: {str(e)}")
        print("   Check your CSV file format and try again.")
        raise


def predict_matches(matches_df: pd.DataFrame, 
                   model_path: str = "data/outputs/model_xgb.json",
                   save_results: bool = True) -> pd.DataFrame:
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
    
    features_df = prepare_match_data(matches_df)
    
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


def predict_from_csv(input_csv: str, model_path: str = "data/outputs/model_xgb.json") -> pd.DataFrame:
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
    
    return predict_matches(matches_df, model_path)


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
