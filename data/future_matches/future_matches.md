# Future Matches Input Guide

Use this guide to add upcoming matches for prediction with the XGBoost model.

## Quick Start
1. Copy `data/future_matches/future_matches_example.csv` and edit it, or create your own CSV in the same folder.
2. Run: `python src/main.py` and follow the prompt to pick your file.
   - Or non-interactive: `python -c "from predict import predict_from_csv; predict_from_csv('data/future_matches/your_file.csv')"`

## Required Columns
- `date` (YYYY-MM-DD)
- `player_1`, `player_2` (as they appear in the player DB; the tool will suggest close matches if a name is unknown)
- `surface` (Hard, Clay, Grass, Carpet)

## Recommended Columns (better accuracy)
- `tournament`, `round`, `best_of`
- `series` (e.g., Grand Slam, Masters 1000, ATP500, ATP250, International)
- `court` (Indoor/Outdoor)
- `series_level`, `is_outdoor`, `surf_fast`, `surf_hard`, `surf_clay`, `surf_grass`, `surf_carpet`, `best_of_3`, `best_of_5`, `rank_1`, `rank_2`, `rank_avg`, `rank_ratio`, `rank_diff`, `is_top10_match`

If you omit these, defaults are inferred where possible, but providing them improves feature quality.

## Name Resolution
- The predictor loads known players from `data/raw/players_db.csv`.
- If a name is unknown, you’ll be prompted with the 3 closest matches (case-insensitive). Choose a suggestion or keep the original (may fail if truly unknown).
- To avoid prompts (e.g., in batch jobs), pass `interactive_resolution=False` to `predict_from_csv`.

## CSV Example
See `data/future_matches/future_matches_example.csv` for a ready-to-edit template with realistic values.

## Tips
- Keep dates in the future or at least after your training cutoff to avoid leakage.
- Ensure ranks are positive; leave blank if unknown.
- Surfaces are case-insensitive but standardize to Hard/Clay/Grass/Carpet.
- If you add new players to your input, also add them to `data/raw/players_db.csv` (or rerun `src/prepare_raw.py` on a fresh ATP dump) so Elo seeding works cleanly.
