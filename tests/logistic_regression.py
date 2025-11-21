"""
Logistic Regression baseline for tennis match prediction.

Loads pre-made time splits, trains a reproducible LogisticRegression model,
evaluates on validation and test sets, and analyzes feature importance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.impute import SimpleImputer
import warnings


def load_splits():
    """Load pre-made time splits from CSV files."""
    print("Loading time splits...")
    
    X_train = pd.read_csv("data/processed/splits/X_train.csv")
    X_val = pd.read_csv("data/processed/splits/X_val.csv")
    X_test = pd.read_csv("data/processed/splits/X_test.csv")
    
    y_train = pd.read_csv("data/processed/splits/y_train.csv")["target"].values
    y_val = pd.read_csv("data/processed/splits/y_val.csv")["target"].values
    y_test = pd.read_csv("data/processed/splits/y_test.csv")["target"].values
    
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


def prepare_features(X_train, X_val, X_test, feature_names):
    """Extract feature columns, excluding MATCH_ID from training."""
    print("\nPreparing features...")
    
    assert "MATCH_ID" in X_train.columns, "MATCH_ID missing from X_train"
    assert "MATCH_ID" in X_val.columns, "MATCH_ID missing from X_val"
    assert "MATCH_ID" in X_test.columns, "MATCH_ID missing from X_test"
    
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
        
        X_train_feats.loc[:, :] = imputer.fit_transform(X_train_feats)
        X_val_feats.loc[:, :] = imputer.transform(X_val_feats)
        X_test_feats.loc[:, :] = imputer.transform(X_test_feats)
        
        print(f"   NaN values imputed successfully")
    else:
        print(f"   No NaN values detected")
    
    return X_train_feats, X_val_feats, X_test_feats


def train_logistic_regression(X_train_feats, y_train):
    """Train reproducible LogisticRegression model."""
    print("\nTraining Logistic Regression...")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        model = LogisticRegression(
            solver="liblinear", 
            random_state=42, 
            max_iter=1000
        )
        model.fit(X_train_feats, y_train)
    
    print(f"   Model trained on {len(X_train_feats):,} samples")
    print(f"   Features: {len(model.coef_[0])} coefficients")
    
    return model


def evaluate_model(model, X_feats, y_true, split_name):
    """Evaluate model and return metrics."""
    p_proba = model.predict_proba(X_feats)[:, 1]
    y_pred = (p_proba >= 0.5).astype(int)
    
    logloss = log_loss(y_true, p_proba)
    auc = roc_auc_score(y_true, p_proba)
    acc = accuracy_score(y_true, y_pred)
    
    print(f"{split_name:4s} | logloss: {logloss:.6f} | auc: {auc:.6f} | acc: {acc:.6f}")
    
    return logloss, auc, acc


def analyze_coefficients(model, feature_names, top_k=20):
    """Analyze and display top coefficients by absolute value."""
    print(f"\nTop {top_k} Features by Coefficient Magnitude:")
    print("-" * 45)
    
    coefs = model.coef_[0]
    
    pairs = sorted(zip(feature_names, coefs), key=lambda t: abs(t[1]), reverse=True)[:top_k]
    
    for i, (name, w) in enumerate(pairs, 1):
        sign = "Positive" if w > 0 else "Negative"
        print(f"{i:2d}. {sign:8s} {name:28s} {w:+.4f}")
    
    print(f"\nInterpretation:")
    print(f"   Positive coefficients -> Higher chance Player 2 wins")
    print(f"   Negative coefficients -> Higher chance Player 1 wins")


def main():
    """Main execution function."""
    print("Tennis Match Prediction - Logistic Regression Baseline")
    print("=" * 65)
    
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_splits()
        
        X_train_feats, X_val_feats, X_test_feats = prepare_features(
            X_train, X_val, X_test, feature_names
        )
        
        model = train_logistic_regression(X_train_feats, y_train)
        
        print("\nModel Evaluation:")
        print("-" * 65)
        val_logloss, val_auc, val_acc = evaluate_model(model, X_val_feats, y_val, "VAL ")
        
        test_logloss, test_auc, test_acc = evaluate_model(model, X_test_feats, y_test, "TEST")
        
        analyze_coefficients(model, feature_names, top_k=20)
        
        print(f"\n" + "=" * 65)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 65)
        print(f"Validation AUC: {val_auc:.4f} | Accuracy: {val_acc:.4f}")
        print(f"Test AUC:       {test_auc:.4f} | Accuracy: {test_acc:.4f}")
        print(f"Total Features: {len(feature_names)}")
        
        if test_auc >= 0.65:
            print("Strong baseline performance!")
        elif test_auc >= 0.60:
            print("Good baseline performance!")
        elif test_auc >= 0.55:
            print("Moderate baseline performance")
        else:
            print("Weak baseline performance - features may need improvement")
        
        return model, (val_logloss, val_auc, val_acc), (test_logloss, test_auc, test_acc)
        
    except FileNotFoundError as e:
        print("Error: Could not find required files.")
        print(f"   {str(e)}")
        print("   Please run 'python src/main.py' first to generate splits.")
        
    except AssertionError as e:
        print(f"Data validation error: {str(e)}")
        
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
