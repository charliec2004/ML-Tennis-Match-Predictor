# Workflow

## Step 1: Ingest → Validate

- `main.py` → `data_io.load_raw()`
- Reads from `data/raw/tennis-master-data.csv`
- Cleans dates, enforces dtypes, trims strings.
- Sorts by `date, MATCH_ID`.

## Step 2: Feature Build

- `main.py` → `features.build(df)` → uses `elo.py`
- Walks through matches chronologically.
- Computes Elo, surface Elo, diffs, optional rolling stats.
- Updates ratings after recording features.
- Returns feature-complete DataFrame.

## Step 3: Save Processed

- `main.py` → `data_io.save_processed(df_feat)`
- Saves:
  - `data/processed/with_elo.csv`
  - parquet optional for speed.

## Step 4: Time Splits

- `main.py` → `timesplits.split(df_feat)`
- Creates train/valid/test sets by date.
- Saves to `data/processed/`.

## Step 5: Train

- `main.py` → `model_xgb.train()`
- Trains on X only (drops meta).
- Early stopping on validation logloss.
- Saves:
  - `model_xgb.json`
  - `feature_list.json`
  - `training_log.csv`

## Step 6: Evaluate

- Produces:
  - logloss, Brier, AUC, calibration curve.
  - Saves plots and JSON in `outputs/`.

## Step 7: Predict

- `predict.py`
- Loads model and feature list.
- Reattaches meta with `MATCH_ID`.
- Saves `outputs/predictions.csv`.
