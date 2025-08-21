"""
XGBoost model training for tennis match prediction.

Loads time-based splits, trains XGBoost with early stopping and hyperparameter tuning,
evaluates performance, and saves the trained model.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import matplotlib.pyplot as plt


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load pre-made time splits from CSV files."""
    print("📂 Loading time splits...")
    
    # Load feature matrices
    X_train = pd.read_csv("data/processed/splits/X_train.csv")
    X_val = pd.read_csv("data/processed/splits/X_val.csv")
    X_test = pd.read_csv("data/processed/splits/X_test.csv")
    
    # Load targets and explicitly convert to numpy arrays
    y_train = np.asarray(pd.read_csv("data/processed/splits/y_train.csv")["target"])
    y_val = np.asarray(pd.read_csv("data/processed/splits/y_val.csv")["target"])
    y_test = np.asarray(pd.read_csv("data/processed/splits/y_test.csv")["target"])
    
    # Load feature names (excludes MATCH_ID)
    with open("data/processed/splits/feature_names.txt", 'r') as f:
        feature_names = [line.strip() for line in f if not line.startswith('#') and line.strip()]
    
    print(f"   ✅ Train: {X_train.shape[0]:,} matches")
    print(f"   ✅ Valid: {X_val.shape[0]:,} matches")
    print(f"   ✅ Test:  {X_test.shape[0]:,} matches")
    print(f"   ✅ Features: {len(feature_names)} columns")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def prepare_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                    feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract feature columns and handle missing values."""
    print("\n🔧 Preparing features...")
    
    # Validate required columns
    assert "MATCH_ID" in X_train.columns, "MATCH_ID missing from X_train"
    missing_features = [f for f in feature_names if f not in X_train.columns]
    assert not missing_features, f"Missing features in X_train: {missing_features}"
    
    # Extract feature columns (exclude MATCH_ID)
    X_train_feats = X_train[feature_names].copy()
    X_val_feats = X_val[feature_names].copy()
    X_test_feats = X_test[feature_names].copy()
    
    print(f"   ✅ Feature matrix shape: {X_train_feats.shape}")
    print(f"   ✅ MATCH_ID excluded from features")
    
    # Handle NaN values
    train_nans = X_train_feats.isnull().sum().sum()
    val_nans = X_val_feats.isnull().sum().sum()
    test_nans = X_test_feats.isnull().sum().sum()
    
    if train_nans + val_nans + test_nans > 0:
        print(f"   ⚠️  NaN values found: Train={train_nans}, Val={val_nans}, Test={test_nans}")
        print(f"      Imputing with median values...")
        
        # XGBoost can handle NaNs natively, but we'll impute for consistency
        imputer = SimpleImputer(strategy='median')
        
        # Transform and create new DataFrames
        X_train_feats = pd.DataFrame(
            np.asarray(imputer.fit_transform(X_train_feats)),
            columns=feature_names,
            index=X_train_feats.index
        )
        X_val_feats = pd.DataFrame(
            np.asarray(imputer.transform(X_val_feats)),
            columns=feature_names,
            index=X_val_feats.index
        )
        X_test_feats = pd.DataFrame(
            np.asarray(imputer.transform(X_test_feats)),
            columns=feature_names,
            index=X_test_feats.index
        )
        
        print(f"   ✅ NaN values imputed successfully")
    else:
        print(f"   ✅ No NaN values detected")
    
    return X_train_feats, X_val_feats, X_test_feats


def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                  feature_names: List[str], params: Optional[Dict[str, Any]] = None) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Train XGBoost model with early stopping and validation tracking."""
    print("\n🚀 Training XGBoost...")
    
    # Default parameters optimized for tennis prediction
    if params is None:
        params = {
            'objective': 'binary:logistic',  # Binary classification
            'eval_metric': 'logloss',        # Optimize for log loss
            'max_depth': 6,                  # Tree depth
            'learning_rate': 0.1,            # Step size shrinkage
            'subsample': 0.8,                # Fraction of samples per tree
            'colsample_bytree': 0.8,         # Fraction of features per tree
            'min_child_weight': 1,           # Minimum sum of instance weight in child
            'reg_alpha': 0.1,                # L1 regularization
            'reg_lambda': 1.0,               # L2 regularization
            'random_state': 42,              # Reproducibility
            'n_jobs': -1,                    # Use all CPU cores
            'verbosity': 0                   # Suppress warnings
        }
    
    # Create DMatrix objects for efficient XGBoost training
    print(f"   📊 Creating DMatrix objects...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # Set up evaluation and early stopping
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result: Dict[str, Any] = {}
    
    print(f"   🏋️  Training with early stopping...")
    print(f"      Max rounds: 1000")
    print(f"      Early stopping: 50 rounds")
    print(f"      Evaluation metric: {params['eval_metric']}")
    
    # Train model with early stopping
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,              # Maximum trees
        evals=evallist,
        evals_result=evals_result,
        early_stopping_rounds=50,          # Stop if no improvement
        verbose_eval=50                    # Print every 50 rounds
    )
    
    print(f"   ✅ Training completed!")
    print(f"   🌳 Best iteration: {model.best_iteration}")
    print(f"   🏆 Best eval score: {model.best_score:.6f}")
    
    return model, evals_result


def evaluate_model(model: xgb.Booster, X_feats: pd.DataFrame, y_true: np.ndarray, 
                  feature_names: List[str], split_name: str) -> Dict[str, Any]:
    """Evaluate XGBoost model and return metrics."""
    
    # Create DMatrix and get predictions
    dmatrix = xgb.DMatrix(X_feats, feature_names=feature_names)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    logloss = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"{split_name:4s} | logloss: {logloss:.6f} | auc: {auc:.6f} | acc: {accuracy:.6f}")
    
    return {
        'logloss': logloss,
        'auc': auc,
        'accuracy': accuracy,
        'predictions': y_pred_proba.tolist()  # Convert to list for JSON serialization
    }


def analyze_feature_importance(model: xgb.Booster, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
    """Analyze and display feature importance from XGBoost model."""
    print(f"\n🔍 Top {top_k} Features by Importance:")
    print("-" * 50)
    
    # Get feature importance (gain-based)
    importance = model.get_score(importance_type='gain')
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame([
        {'feature': feature, 'importance': importance.get(feature, 0.0)}
        for feature in feature_names
    ]).sort_values('importance', ascending=False)
    
    # Display top features using enumerate for proper integer indexing
    for i, (_, row) in enumerate(importance_df.head(top_k).iterrows()):
        print(f"{i+1:2d}. 📊 {row['feature']:28s} {row['importance']:8.4f}")
    
    return importance_df


def plot_training_curves(evals_result: Dict[str, Any], output_dir: str = "data/outputs") -> None:
    """Plot training and validation curves."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Extract training history
    train_logloss = evals_result['train']['logloss']
    val_logloss = evals_result['eval']['logloss']
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_logloss, label='Training Log Loss', color='blue')
    plt.plot(val_logloss, label='Validation Log Loss', color='red')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_path = Path(output_dir) / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Training curves saved to {plot_path}")


def save_model_and_results(model: xgb.Booster, feature_names: List[str], results: Dict[str, Any], 
                          importance_df: pd.DataFrame, output_dir: str = "data/outputs") -> None:
    """Save trained model and results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save XGBoost model
    model_path = output_path / "model_xgb.json"
    model.save_model(str(model_path))
    print(f"   💾 Model saved to {model_path}")
    
    # Save feature names
    feature_path = output_path / "feature_names.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   📝 Feature names saved to {feature_path}")
    
    # Save results (predictions are already converted to lists)
    results_path = output_path / "results.json"
    json_results = {}
    for split, metrics in results.items():
        json_results[split] = {k: v for k, v in metrics.items()}
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   📊 Results saved to {results_path}")
    
    # Save feature importance
    importance_path = output_path / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   🔍 Feature importance saved to {importance_path}")


def main() -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Main XGBoost training pipeline."""
    print("🎾 Tennis Match Prediction - XGBoost Training")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()
        
        # Step 2: Prepare features
        X_train_feats, X_val_feats, X_test_feats = prepare_features(
            X_train, X_val, X_test, feature_names
        )
        
        # Step 3: Train XGBoost model
        model, evals_result = train_xgboost(
            X_train_feats, y_train, X_val_feats, y_val, feature_names
        )
        
        # Step 4: Evaluate on all splits
        print("\n📊 Model Evaluation:")
        print("-" * 60)
        
        results: Dict[str, Any] = {}
        results['train'] = evaluate_model(model, X_train_feats, y_train, feature_names, "TRAIN")
        results['val'] = evaluate_model(model, X_val_feats, y_val, feature_names, "VAL ")
        results['test'] = evaluate_model(model, X_test_feats, y_test, feature_names, "TEST")
        
        # Step 5: Analyze feature importance
        importance_df = analyze_feature_importance(model, feature_names)
        
        # Step 6: Create visualizations
        print(f"\n📈 Creating visualizations...")
        plot_training_curves(evals_result)
        
        # Step 7: Save everything
        print(f"\n💾 Saving model and results...")
        save_model_and_results(model, feature_names, results, importance_df)
        
        # Step 8: Final summary
        print(f"\n" + "=" * 60)
        print("🏁 XGBOOST TRAINING COMPLETE!")
        print("=" * 60)
        
        val_auc = results['val']['auc']
        test_auc = results['test']['auc']
        val_acc = results['val']['accuracy']
        test_acc = results['test']['accuracy']
        
        print(f"📈 Validation: AUC {val_auc:.4f} | Accuracy {val_acc:.4f}")
        print(f"🎯 Test:       AUC {test_auc:.4f} | Accuracy {test_acc:.4f}")
        print(f"🌳 Trees used: {model.best_iteration}")
        print(f"📊 Features:   {len(feature_names)}")
        
        # Compare to logistic regression baseline
        baseline_test_auc = 0.7154  # From your logistic regression results
        improvement = test_auc - baseline_test_auc
        print(f"\n🚀 Improvement over Logistic Regression:")
        print(f"   Test AUC: {improvement:+.4f} ({improvement/baseline_test_auc*100:+.1f}%)")
        
        if improvement > 0.01:
            print("   ✅ Significant improvement! XGBoost is working well.")
        elif improvement > 0.005:
            print("   ✅ Modest improvement. Consider hyperparameter tuning.")
        else:
            print("   ⚠️  Limited improvement. Your linear baseline was already strong!")
        
        print(f"\n📁 Files created in data/outputs/:")
        print(f"   🤖 model_xgb.json       - Trained XGBoost model")
        print(f"   📝 feature_names.json   - Feature list")
        print(f"   📊 results.json         - Performance metrics")
        print(f"   🔍 feature_importance.csv - Feature rankings")
        print(f"   📈 training_curves.png  - Training visualization")
        
        return model, results
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files.")
        print(f"   {str(e)}")
        print(f"   Please run 'python src/main.py' first to generate splits.")
        raise
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()