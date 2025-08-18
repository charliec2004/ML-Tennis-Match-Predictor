# Module roles and handoffs

## config.py

Role: holds paths, column lists, constants.
Inputs: none
Outputs: constants consumed by all modules

## data_io.py

Role: load raw CSV, clean, sort
Inputs: data/raw/*.csv
Outputs: clean_df (DataFrame)
Writes: optional data/processed/clean.csv

## elo.py

Role: Elo math helpers
Inputs: ratings, match outcome
Outputs: updated ratings, expected scores
Used by: features.py

## features.py

Role: build pre-match features sequentially
Inputs: clean_df
Outputs: feat_df (meta + X + y)
Writes: data/processed/with_elo.csv

## timesplits.py

Role: time-based splits
Inputs: feat_df
Outputs: X_train, y_train, X_valid, y_valid, (optional X_test, y_test)
Writes: data/processed/train.csv, valid.csv, test.csv

## model_xgb.py

Role: train XGBoost with early stopping
Inputs: splits from timesplits, feature list from config
Outputs: trained model, validation preds
Writes: data/outputs/model_xgb.json, feature_list.json, training_log.csv

## metrics.py

Role: evaluate holdout
Inputs: y_true, y_pred
Outputs: metrics dict, plots
Writes: data/outputs/metrics.json, roc.png, calibration.png

## predict.py

Role: batch inference
Inputs: model_xgb.json, processed dataset with same X, meta for join
Outputs: predictions with MATCH_ID
Writes: data/outputs/predictions.csv

## main.py

Role: orchestrator CLI
Inputs: command (build, split, train, eval, predict)
Outputs: drives the flow by calling modules

## Handoffs (left produces, right consumes)

data_io → features: clean_df  
features → timesplits: feat_df  
timesplits → model_xgb: X/y splits  
model_xgb → metrics: holdout preds + y_true  
model_xgb → predict: saved model artifact  
predict → outputs: predictions with meta reattached
