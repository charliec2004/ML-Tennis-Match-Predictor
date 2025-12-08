# Workflow

## Step 0: Prepare Raw Master Data

- `main.py` → `prepare_raw.prepare_master_data()`
- Reads from `data/raw/atp_tennis.csv`
- Cleans dates, normalizes player names, enforces dtypes
- Derives helper columns (rank_avg, rank_ratio, rank_diff, is_top10_match)
- Writes `data/raw/tennis-master-data.csv` sorted chronologically
- Updates `data/raw/players_db.csv` with new players

## Step 1: Load Raw Data

- `main.py` → loads `data/raw/tennis-master-data.csv`
- Data already cleaned and sorted by date

## Step 2: Feature Build

- `main.py` → `features.generate_features(df)`
- Sorts data chronologically (by date) to prevent leakage
- Uses `elo.EloProcessor()` and `matches.MatchHistoryProcessor()`
- Walks through matches chronologically
- Computes overall Elo, surface-specific Elo (hard/clay/grass/carpet), diffs
- Computes match history: win rates (all/5/10 matches), h2h, recency
- Filters: first 2000 matches as warmup (ELO burn-in), drops matches where players have <5 prior matches
- Updates ratings after recording features
- Returns feature-complete DataFrame (38 features)

## Step 3: Save Processed

- `main.py` → saves enriched DataFrame
- Saves to:
  - `data/processed/with_elo.csv` (CSV format only)

## Step 4: Time Splits

- `main.py` → `timesplits.make_splits(df_feat)` and `timesplits.save_splits()`
- Creates train/valid/test sets by date cutoffs (train: up to 2018-12-31, valid: 2019-2022, test: 2023+)
- Saves to `data/processed/splits/`:
  - `X_train.csv`, `y_train.csv`, `meta_train.csv`
  - `X_val.csv`, `y_val.csv`, `meta_val.csv`
  - `X_test.csv`, `y_test.csv`, `meta_test.csv`
  - `feature_names.txt`

## Step 5: Train

- `main.py` → calls `model_xgb.train_xgboost_pipeline()` (subprocess)
- Loads splits from `data/processed/splits/`
- Trains on X only (drops MATCH_ID and meta columns)
- Imputes NaN values with median
- Early stopping on validation logloss (20 rounds, max 500 trees)
- Saves to `data/outputs/`:
  - `model_xgb.json`
  - `feature_names.json`
  - `results.json`
  - `feature_importance.csv`
  - `training_curves.png`

## Step 6: Evaluate

- Performed automatically in Step 5 training
- Produces metrics for train/val/test:
  - Log loss, AUC, Accuracy
  - Feature importance rankings
- Saves results to `data/outputs/results.json`
- Training curves saved to `data/outputs/training_curves.png`

## Step 7: Predict (Optional)

- `main.py` → prompts user if they want to predict future matches
- If yes: calls `predict.predict_from_csv(csv_path)`
- Loads model from `data/outputs/model_xgb.json`
- Loads feature names from `data/outputs/feature_names.json`
- Reads future match CSV from `data/future_matches/`
- Generates same 38 features using current ELO/history state
- Predicts winner and confidence
- Saves `data/outputs/predictions.csv` with columns:
  - date, player_1, player_2, predicted_winner, confidence, prob_p1_wins, prob_p2_wins
