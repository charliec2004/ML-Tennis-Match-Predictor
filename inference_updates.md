## Prediction Inference Updates

### What changed
- Added seeded Elo/history state when generating features for future matches.
- Stopped future rows from updating Elo/history during inference.
- Added lightweight mapping utilities in `src/predict.py` and support for passing pre-seeded processors in `src/features.py`.

### Why
- Previously, inference built Elo and history only from the future CSV, so the model “learned” from those same rows and assumed player_1 won by default, producing inflated confidence.
- Unknown names (e.g., `Jodar R.`) caused crashes or mismatched resolutions that further skewed outputs.

### How
- `src/predict.py` now loads `data/raw/tennis-master-data.csv` to initialize `EloProcessor` and `MatchHistoryProcessor`, then computes features with `update_state=False` so future matches don’t contaminate ratings/history.
- `src/features.py` accepts optional pre-seeded processors and a `update_state` flag to keep state fixed during inference.
- Name normalization and feature-building are aligned with training-time preprocessing to ensure required columns/flags exist.***
