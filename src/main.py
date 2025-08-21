import pandas as pd
from features import generate_features
from timesplits import make_splits, save_splits

def main():
    """
    Complete pipeline: Load raw data → Generate features → Create time-based splits
    """
    print("🎾 Tennis Match Prediction Pipeline")
    print("=" * 50)
    
    # Step 1: Load raw data
    print("\n📂 Step 1: Loading raw data...")
    try:
        raw = pd.read_csv("data/raw/tennis-master-data.csv")
        print(f"   ✅ Loaded {len(raw):,} raw matches")
    except FileNotFoundError:
        print("   ❌ Error: Could not find data/raw/tennis-master-data.csv")
        print("   Please ensure the raw data file exists.")
        return
    
    # Step 2: Generate features (ELO + match history)
    print("\n🔧 Step 2: Generating features...")
    try:
        df_feat = generate_features(raw)
        print(f"   ✅ Features generated! Shape: {df_feat.shape}")
        print(f"   💾 Saved to: data/processed/with_elo.csv")
    except Exception as e:
        print(f"   ❌ Error generating features: {e}")
        return
    
    # Step 3: Create time-based splits
    print("\n⏰ Step 3: Creating time-based splits...")
    try:
        # Ensure date column is datetime for splitting
        df_feat['date'] = pd.to_datetime(df_feat['date'])
        
        # Create splits with default time cutoffs
        # Train: 2000-2018, Valid: 2019-2022, Test: 2023+
        splits = make_splits(
            df=df_feat,
            date_col="date",
            y_col="target",
            train_end="2018-12-31",
            val_end="2022-12-31"
        )
        
        print(f"   ✅ Time-based splits created successfully!")
        
    except Exception as e:
        print(f"   ❌ Error creating splits: {e}")
        return
    
    # Step 4: Save splits to CSV files
    print("\n💾 Step 4: Saving splits...")
    try:
        save_splits(splits, "data/processed/splits")
        
    except Exception as e:
        print(f"   ❌ Error saving splits: {e}")
        return
    
    # Step 5: Pipeline summary
    print("\n" + "=" * 50)
    print("🏁 PIPELINE COMPLETE!")
    print("=" * 50)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Raw matches:      {len(raw):,}")
    print(f"   Feature matches:  {len(df_feat):,}")
    print(f"   Features created: {len(splits['x_cols'])}")
    print(f"   Train matches:    {len(splits['train']['X']):,}")
    print(f"   Valid matches:    {len(splits['val']['X']):,}")
    print(f"   Test matches:     {len(splits['test']['X']):,}")
    
    print(f"\n📅 DATE RANGES:")
    print(f"   Train: {splits['train']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['train']['meta']['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Valid: {splits['val']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['val']['meta']['date'].max().strftime('%Y-%m-%d')}")
    print(f"   Test:  {splits['test']['meta']['date'].min().strftime('%Y-%m-%d')} to {splits['test']['meta']['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"   📊 Features:      data/processed/with_elo.csv")
    print(f"   🗂️  Splits:        data/processed/splits/")
    print(f"       ├── X_train.csv, y_train.csv, meta_train.csv")
    print(f"       ├── X_val.csv, y_val.csv, meta_val.csv")
    print(f"       ├── X_test.csv, y_test.csv, meta_test.csv")
    print(f"       └── feature_names.txt")
    
    print(f"\n🚀 READY FOR MODEL TRAINING:")
    print(f"   Use X_train.csv + y_train.csv for XGBoost")
    print(f"   Validate on X_val.csv + y_val.csv")
    print(f"   Final test on X_test.csv + y_test.csv")
    
    return splits

if __name__ == "__main__":
    main()