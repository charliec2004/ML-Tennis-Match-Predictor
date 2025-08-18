# Project Structure

## Folders

- **data/**
  - **raw/**: source CSVs exported from Excel.
  - **processed/**: feature-ready datasets, time splits, and intermediate artifacts.
  - **outputs/**: trained models, predictions, plots, and metrics.

- **src/**
  - **config.py**: central paths, constants, and column definitions.
  - **data_io.py**: load/validate raw data, save processed datasets.
  - **elo.py**: Elo and surface Elo updates.
  - **features.py**: builds full feature set (Elo, rolling stats, etc).
  - **timesplits.py**: train/valid/test splits by time.
  - **model_xgb.py**: XGBoost training and saving.
  - **metrics.py**: evaluation metrics + plots.
  - **predict.py**: run predictions on new data.
  - **main.py**: orchestrator; runs pipeline steps in sequence.

- **notebooks/**  
  Ad-hoc EDA, sanity checks, quick visuals. Not part of the pipeline.

- **requirements.txt**: dependencies.
- **README.md**: setup, usage, and high-level explanation.
- **venv/**: virtual environment (ignored by git).
