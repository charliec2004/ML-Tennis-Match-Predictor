import pandas as pd
import os
import glob
from pathlib import Path
from typing import Optional
from features import generate_features
from timesplits import make_splits, save_splits

def ask_predict_future_matches() -> bool:
    """Ask user if they want to predict future matches with input validation."""
    while True:
        try:
            response = input("\nDo you want to predict future matches? (y/n): ").strip().lower()
            
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("   Please enter 'y' for yes or 'n' for no.")
                
        except KeyboardInterrupt:
            print("\n\n   Goodbye!")
            return False
        except EOFError:
            print("\n\n   No input received. Skipping predictions.")
            return False


def find_future_match_files() -> list[str]:
    """Find CSV files in data/future_matches/ directory."""
    future_matches_dir = Path("data/future_matches")
    
    if not future_matches_dir.exists():
        return []
    
    csv_files = list(future_matches_dir.glob("*.csv"))
    return [str(f) for f in csv_files]


def select_match_file(csv_files: list[str]) -> Optional[str]:
    """Let user select which CSV file to predict from."""
    if len(csv_files) == 1:
        print(f"   Found: {csv_files[0]}")
        return csv_files[0]
    
    print(f"   Found {len(csv_files)} match files:")
    for i, file_path in enumerate(csv_files, 1):
        filename = Path(file_path).name
        print(f"   {i}. {filename}")
    
    while True:
        try:
            choice = input(f"\n   Select file (1-{len(csv_files)}): ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    return csv_files[idx]
            
            print(f"   Please enter a number between 1 and {len(csv_files)}")
            
        except KeyboardInterrupt:
            print("\n   Skipping predictions.")
            return None
        except EOFError:
            print("\n   No input received. Using first file.")
            return csv_files[0]


def predict_future_matches_interactive() -> None:
    """Interactive future match prediction with error handling."""
    print("\nStep 6: Future Match Predictions")
    print("-" * 50)
    
    future_matches_dir = Path("data/future_matches")
    if not future_matches_dir.exists():
        print("   Creating data/future_matches/ directory...")
        future_matches_dir.mkdir(parents=True, exist_ok=True)
        
        print("   No future matches directory found.")
        print("   To predict matches, add CSV files to: data/future_matches/")
        print("   CSV format: date,player_1,player_2,surface,tournament,round,best_of")
        print("   Example: 2025-08-26,Djokovic N.,Alcaraz C.,Hard,US Open,R1,5")
        return
    
    csv_files = find_future_match_files()
    
    if not csv_files:
        print("   data/future_matches/ directory exists but no CSV files found.")
        print("   Add CSV files with future matches to predict.")
        print("   CSV format: date,player_1,player_2,surface,tournament,round,best_of")
        return
    
    want_predictions = ask_predict_future_matches()
    
    if not want_predictions:
        print("   Skipping future match predictions.")
        return
    
    selected_file = select_match_file(csv_files)
    if not selected_file:
        return
    
    try:
        print(f"\n   Loading prediction module...")
        from predict import predict_from_csv
        
        print(f"   Predicting matches from: {Path(selected_file).name}")
        predictions = predict_from_csv(selected_file)
        
        print(f"\n   PREDICTION RESULTS:")
        print(f"   {'Match':<40} {'Winner':<20} {'Confidence':<12}")
        print(f"   {'-'*40} {'-'*20} {'-'*12}")
        
        for _, row in predictions.head(10).iterrows():
            match_name = f"{row['player_1']} vs {row['player_2']}"[:38]
            winner = row['predicted_winner'][:18]
            confidence = f"{row['confidence']:.1%}"
            
            print(f"   {match_name:<40} {winner:<20} {confidence}")
        
        if len(predictions) > 10:
            print(f"   ... and {len(predictions) - 10} more matches")
        
        print(f"\n   Full results saved to: data/outputs/predictions.csv")
        
    except ImportError:
        print("   Error: predict.py module not found.")
        print("   Make sure predict.py exists in the src/ directory.")
        
    except FileNotFoundError as e:
        print(f"   Error: Could not find required model files.")
        print(f"   Make sure you've trained the model first: python src/model_xgb.py")
        print(f"   Details: {e}")
        
    except Exception as e:
        print(f"   Error during prediction: {e}")
        print(f"   Check your CSV file format and try again.")


def main() -> Optional[dict]:
    """
    Complete pipeline: Load raw data -> Generate features -> Create time-based splits -> Train XGBoost -> Predict Future
    """
    print("Tennis Match Prediction Pipeline")
    print("=" * 50)
    
    raw = None
    df_feat = None
    splits = None
    
    print("\nStep 1: Loading raw data...")
    try:
        raw = pd.read_csv("data/raw/tennis-master-data.csv")
        print(f"   Loaded {len(raw):,} raw matches")
    except FileNotFoundError:
        print("   Error: Could not find data/raw/tennis-master-data.csv")
        print("   Please ensure the raw data file exists.")
        return None
    
    print("\nStep 2: Generating features...")
    try:
        df_feat = generate_features(raw)
        print(f"   Features generated! Shape: {df_feat.shape}")
        print(f"   Saved to: data/processed/with_elo.csv")
    except Exception as e:
        print(f"   Error generating features: {e}")
        return None
    
    print("\nStep 3: Creating time-based splits...")
    try:
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        
        splits = make_splits(
            df=df_feat,
            date_col="date",
            y_col="target",
            train_end="2018-12-31",
            val_end="2022-12-31"
        )
        
        print(f"   Time-based splits created successfully!")
        
    except Exception as e:
        print(f"   Error creating splits: {e}")
        return None
    
    print("\nStep 4: Saving splits...")
    try:
        save_splits(splits, "data/processed/splits")
        
    except Exception as e:
        print(f"   Error saving splits: {e}")
        return None
    
    print("\nStep 5: Training XGBoost Model...")
    model_results = None
    try:
        from model_xgb import train_xgboost_pipeline
        
        model, model_results = train_xgboost_pipeline()
        
        print(f"   XGBoost training completed!")
        print(f"   Test AUC: {model_results['test']['auc']:.4f}")
        print(f"   Test Accuracy: {model_results['test']['accuracy']:.4f}")
        
    except Exception as e:
        print(f"   Error training XGBoost: {e}")
        print(f"   You can still train manually with: python src/model_xgb.py")
    
    predict_future_matches_interactive()
    
    print("\n" + "=" * 50)
    print("COMPLETE PIPELINE FINISHED!")
    print("=" * 50)
    
    if raw is not None and df_feat is not None and splits is not None:
        print(f"\nFINAL SUMMARY:")
        print(f"   Raw matches:      {len(raw):,}")
        print(f"   Feature matches:  {len(df_feat):,}")
        print(f"   Features created: {len(splits['x_cols'])}")
        print(f"   Train matches:    {len(splits['train']['X']):,}")
        print(f"   Valid matches:    {len(splits['val']['X']):,}")
        print(f"   Test matches:     {len(splits['test']['X']):,}")
        
        print(f"\nDATE RANGES:")
        print(f"   Train: {splits['train']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['train']['meta']['date'].max().strftime('%Y-%m-%d')}")
        print(f"   Valid: {splits['val']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['val']['meta']['date'].max().strftime('%Y-%m-%d')}")
        print(f"   Test:  {splits['test']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['test']['meta']['date'].max().strftime('%Y-%m-%d')}")
        
        print(f"\nFILES CREATED:")
        print(f"   Features:      data/processed/with_elo.csv")
        print(f"   Data Splits:   data/processed/splits/")
        print(f"   Trained Model: data/outputs/model_xgb.json")
        print(f"   Predictions:   data/outputs/predictions.csv (if generated)")
        
        print(f"\nPIPELINE COMPLETE!")
        print(f"   Add future matches to data/future_matches/ for predictions!")
        
        return splits
    else:
        print(f"\nPIPELINE INCOMPLETE!")
        print(f"   Some steps failed. Check error messages above.")
        return None


if __name__ == "__main__":
    main()
