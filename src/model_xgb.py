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
    print("Loading time splits...")
    
    X_train = pd.read_csv("data/processed/splits/X_train.csv")
    X_val = pd.read_csv("data/processed/splits/X_val.csv")
    X_test = pd.read_csv("data/processed/splits/X_test.csv")
    
    y_train = np.asarray(pd.read_csv("data/processed/splits/y_train.csv")["target"])
    y_val = np.asarray(pd.read_csv("data/processed/splits/y_val.csv")["target"])
    y_test = np.asarray(pd.read_csv("data/processed/splits/y_test.csv")["target"])
    
    with open("data/processed/splits/feature_names.txt", 'r') as f:
        feature_names = [
            line.strip()
            for line in f
            if line.strip()
            and not line.lstrip().startswith('#')
            and "Feature columns" not in line
        ]
    
    print(f"   Train: {X_train.shape[0]:,} matches")
    print(f"   Valid: {X_val.shape[0]:,} matches")
    print(f"   Test:  {X_test.shape[0]:,} matches")
    print(f"   Features: {len(feature_names)} columns")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


def prepare_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame, 
                    feature_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract feature columns and handle missing values."""
    print("\nPreparing features...")
    
    assert "MATCH_ID" in X_train.columns, "MATCH_ID missing from X_train"
    missing_features = [f for f in feature_names if f not in X_train.columns]
    assert not missing_features, f"Missing features in X_train: {missing_features}"
    
    X_train_feats = X_train[feature_names].copy()
    X_val_feats = X_val[feature_names].copy()
    X_test_feats = X_test[feature_names].copy()
    
    print(f"   Feature matrix shape: {X_train_feats.shape}")
    print(f"   MATCH_ID excluded from features")
    
    train_nans = X_train_feats.isnull().sum().sum()
    val_nans = X_val_feats.isnull().sum().sum()
    test_nans = X_test_feats.isnull().sum().sum()
    
    if train_nans + val_nans + test_nans > 0:
        print(f"   NaN values found: Train={train_nans}, Val={val_nans}, Test={test_nans}")
        print(f"      Imputing with median values...")
        
        imputer = SimpleImputer(strategy='median')
        
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
        
        print(f"   NaN values imputed successfully")
    else:
        print(f"   No NaN values detected")
    
    return X_train_feats, X_val_feats, X_test_feats


def train_xgboost(X_train: pd.DataFrame, y_train: np.ndarray, X_val: pd.DataFrame, y_val: np.ndarray,
                  feature_names: List[str], params: Optional[Dict[str, Any]] = None) -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Train XGBoost model with early stopping and validation tracking."""
    print("\nTraining XGBoost...")
    
    if params is None:
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 4,                  
            'learning_rate': 0.05,           
            'subsample': 0.7,                
            'colsample_bytree': 0.7,         
            'min_child_weight': 5,           
            'reg_alpha': 1.0,                
            'reg_lambda': 2.0,               
            'gamma': 1.0,                    
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    print(f"   Creating DMatrix objects...")
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    evals_result: Dict[str, Any] = {}
    
    print(f"   Training with aggressive early stopping...")
    print(f"      Max rounds: 500")
    print(f"      Early stopping: 20 rounds")
    print(f"      Evaluation metric: {params['eval_metric']}")
    
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evallist,
        evals_result=evals_result,
        early_stopping_rounds=20,
        verbose_eval=25
    )
    
    print(f"   Training completed!")
    print(f"   Best iteration: {model.best_iteration}")
    print(f"   Best eval score: {model.best_score:.6f}")
    
    return model, evals_result


def evaluate_model(model: xgb.Booster, X_feats: pd.DataFrame, y_true: np.ndarray, 
                  feature_names: List[str], split_name: str) -> Dict[str, Any]:
    """Evaluate XGBoost model and return metrics."""
    
    dmatrix = xgb.DMatrix(X_feats, feature_names=feature_names)
    y_pred_proba = model.predict(dmatrix)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    logloss = log_loss(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"{split_name:4s} | logloss: {logloss:.6f} | auc: {auc:.6f} | acc: {accuracy:.6f}")
    
    return {
        'logloss': logloss,
        'auc': auc,
        'accuracy': accuracy,
        'predictions': y_pred_proba.tolist()
    }


def analyze_feature_importance(model: xgb.Booster, feature_names: List[str], top_k: int = 20) -> pd.DataFrame:
    """Analyze and display feature importance from XGBoost model."""
    print(f"\nTop {top_k} Features by Importance:")
    print("-" * 50)
    
    importance = model.get_score(importance_type='gain')
    
    importance_df = pd.DataFrame([
        {'feature': feature, 'importance': importance.get(feature, 0.0)}
        for feature in feature_names
    ]).sort_values('importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(top_k).iterrows()):
        print(f"{i+1:2d}. {row['feature']:28s} {row['importance']:8.4f}")
    
    return importance_df


def plot_training_curves(evals_result: Dict[str, Any], output_dir: str = "data/outputs") -> None:
    """Plot training and validation curves."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_logloss = evals_result['train']['logloss']
    val_logloss = evals_result['eval']['logloss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_logloss, label='Training Log Loss', color='blue')
    plt.plot(val_logloss, label='Validation Log Loss', color='red')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path(output_dir) / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Training curves saved to {plot_path}")


def save_model_and_results(model: xgb.Booster, feature_names: List[str], results: Dict[str, Any], 
                          importance_df: pd.DataFrame, output_dir: str = "data/outputs") -> None:
    """Save trained model and results."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model_path = output_path / "model_xgb.json"
    model.save_model(str(model_path))
    print(f"   Model saved to {model_path}")
    
    feature_path = output_path / "feature_names.json"
    with open(feature_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"   Feature names saved to {feature_path}")
    
    results_path = output_path / "results.json"
    json_results = {}
    for split, metrics in results.items():
        json_results[split] = {k: v for k, v in metrics.items()}
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"   Results saved to {results_path}")
    
    importance_path = output_path / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   Feature importance saved to {importance_path}")


def train_xgboost_pipeline() -> Tuple[xgb.Booster, Dict[str, Any]]:
    """XGBoost training pipeline that can be called from main.py"""
    print("Tennis Match Prediction - XGBoost Training")
    print("=" * 60)
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()
        
        X_train_feats, X_val_feats, X_test_feats = prepare_features(
            X_train, X_val, X_test, feature_names
        )
        
        model, evals_result = train_xgboost(
            X_train_feats, y_train, X_val_feats, y_val, feature_names
        )
        
        print("\nModel Evaluation:")
        print("-" * 60)
        
        results: Dict[str, Any] = {}
        results['train'] = evaluate_model(model, X_train_feats, y_train, feature_names, "TRAIN")
        results['val'] = evaluate_model(model, X_val_feats, y_val, feature_names, "VAL ")
        results['test'] = evaluate_model(model, X_test_feats, y_test, feature_names, "TEST")
        
        importance_df = analyze_feature_importance(model, feature_names)
        
        print(f"\nCreating visualizations...")
        plot_training_curves(evals_result)
        
        print(f"\nSaving model and results...")
        save_model_and_results(model, feature_names, results, importance_df)
        
        val_auc = results['val']['auc']
        test_auc = results['test']['auc']
        val_acc = results['val']['accuracy']
        test_acc = results['test']['accuracy']
        
        print(f"\nXGBoost Results:")
        print(f"   Validation: AUC {val_auc:.4f} | Accuracy {val_acc:.4f}")
        print(f"   Test:       AUC {test_auc:.4f} | Accuracy {test_acc:.4f}")
        print(f"   Trees used: {model.best_iteration}")
        
        baseline_test_auc = 0.7154
        improvement = test_auc - baseline_test_auc
        if improvement > 0.005:
            print(f"   Improvement over baseline: +{improvement:.4f} AUC")
        else:
            print(f"   Performance vs baseline: {improvement:+.4f} AUC")
        
        return model, results
        
    except Exception as e:
        print(f"XGBoost training failed: {str(e)}")
        raise


def main() -> Tuple[xgb.Booster, Dict[str, Any]]:
    """Main XGBoost training pipeline (standalone version)."""
    return train_xgboost_pipeline()


if __name__ == "__main__":
    main()
