# Future Matches Prediction Guide

A step-by-step guide to predicting upcoming tennis matches using the trained model.

---

## Quick Start (5 Minutes)

### Step 1: Create Your Match File

Create a new CSV file in `data/future_matches/` with any name (e.g., `my_matches.csv`):

```csv
date,player_1,player_2,surface,tournament,round,best_of,series,court,rank_1,rank_2
2025-12-15,Djokovic N.,Alcaraz C.,Hard,ATP Finals,SF,3,Masters,Indoor,1,2
2025-12-16,Sinner J.,Medvedev D.,Hard,ATP Finals,SF,3,Masters,Indoor,4,5
```

### Step 2: Run Predictions

**Option A: Interactive (recommended)**
```bash
python src/main.py
```
- Answer 'n' to skip retraining
- Answer 'y' when asked about future predictions
- Select your CSV file from the list

**Option B: Direct prediction**
```bash
python -c "from predict import predict_from_csv; predict_from_csv('data/future_matches/my_matches.csv')"
```

### Step 3: View Results

Check `data/outputs/predictions.csv` for your predictions with confidence scores!

---

## CSV Format Guide

### Minimum Required Columns

These 4 columns are **mandatory**:

| Column | Format | Example | Notes |
|--------|--------|---------|-------|
| `date` | YYYY-MM-DD | `2025-12-15` | Future date recommended |
| `player_1` | Last Name F. | `Djokovic N.` | Must match player database |
| `player_2` | Last Name F. | `Alcaraz C.` | Must match player database |
| `surface` | Hard/Clay/Grass/Carpet | `Hard` | Case-insensitive |

**Minimal CSV example:**
```csv
date,player_1,player_2,surface
2025-12-15,Djokovic N.,Alcaraz C.,Hard
2025-12-16,Nadal R.,Federer R.,Clay
```

### Recommended Columns (Better Accuracy)

Adding these columns **significantly improves prediction accuracy**:

| Column | Format | Example | Impact |
|--------|--------|---------|--------|
| `rank_1` | Number | `1` | ⭐⭐⭐ Very important |
| `rank_2` | Number | `2` | ⭐⭐⭐ Very important |
| `tournament` | Text | `US Open` | ⭐⭐ Important |
| `series` | See below | `Grand Slam` | ⭐⭐ Important |
| `round` | See below | `SF` | ⭐ Helpful |
| `best_of` | 3 or 5 | `5` | ⭐ Helpful |
| `court` | Indoor/Outdoor | `Outdoor` | ⭐ Helpful |

**Series values:**
- `Grand Slam` - Australian Open, French Open, Wimbledon, US Open
- `Masters 1000` - Indian Wells, Miami, Monte Carlo, Madrid, Rome, Canada, Cincinnati, Shanghai, Paris
- `ATP500` - Rotterdam, Dubai, Barcelona, etc.
- `ATP250` - Smaller ATP tournaments
- `International` - Other professional matches

**Round values:**
- `R128` - First round (128 draw)
- `R64` - Second round (64 draw)
- `R32` - Third round
- `R16` - Round of 16
- `QF` - Quarterfinals
- `SF` - Semifinals
- `F` - Final

**Recommended CSV example:**
```csv
date,player_1,player_2,surface,tournament,round,best_of,series,court,rank_1,rank_2
2025-08-25,Djokovic N.,Alcaraz C.,Hard,US Open,F,5,Grand Slam,Outdoor,1,3
2025-08-26,Sinner J.,Medvedev D.,Hard,US Open,SF,5,Grand Slam,Outdoor,4,5
```

---

## Player Name Format

### Finding Player Names

Player names must match those in `data/raw/players_db.csv`. Check this file for exact spelling:

```bash
cat data/raw/players_db.csv | grep -i "djokovic"
```

### Name Format Rules

1. **Format:** `LastName F.` (last name, space, first initial, period)
2. **Examples:**
   - ✅ `Djokovic N.`
   - ✅ `Alcaraz C.`
   - ✅ `Nadal R.`
   - ❌ `Novak Djokovic` (full names don't work)
   - ❌ `N. Djokovic` (wrong order)
   - ❌ `djokovic n` (needs capitalization and period)

### Unknown Players

If you enter a player name not in the database, you'll see:

```
Unknown player 'Djokovic N'. Did you mean:
   1. Djokovic N.
   2. Djokovic M.
   0. Keep as-is (may fail if truly unknown)
Select a number:
```

The system will suggest close matches - just type the number and press Enter.

---

## Complete Example Workflow

### Example: Predicting US Open Finals

**1. Create `us_open_finals.csv`:**
```csv
date,player_1,player_2,surface,tournament,round,best_of,series,court,rank_1,rank_2
2025-09-08,Djokovic N.,Alcaraz C.,Hard,US Open,F,5,Grand Slam,Outdoor,1,2
2025-09-08,Sinner J.,Medvedev D.,Hard,US Open,SF,5,Grand Slam,Outdoor,4,5
2025-09-08,Fritz T.,Rublev A.,Hard,US Open,QF,5,Grand Slam,Outdoor,7,8
```

**2. Run prediction:**
```bash
python src/main.py
```

**3. Output in `data/outputs/predictions.csv`:**
```csv
date,player_1,player_2,prob_p1_wins,prob_p2_wins,predicted_winner,confidence
2025-09-08,Djokovic N.,Alcaraz C.,0.48,0.52,Alcaraz C.,0.52
2025-09-08,Sinner J.,Medvedev D.,0.61,0.39,Sinner J.,0.61
2025-09-08,Fritz T.,Rublev A.,0.44,0.56,Rublev A.,0.56
```

**Interpreting results:**
- **Alcaraz C.** predicted to beat Djokovic with 52% confidence (toss-up!)
- **Sinner J.** predicted to beat Medvedev with 61% confidence (slight favorite)
- **Rublev A.** predicted to beat Fritz with 56% confidence (slight favorite)

---

## Tips & Best Practices

### For Best Accuracy

1. **Always include rankings** - `rank_1` and `rank_2` are the most important optional fields
2. **Match the surface** - Ensure surface matches the actual tournament surface
3. **Use current rankings** - Get latest ATP rankings from [atptour.com](https://www.atptour.com/en/rankings/singles)
4. **Specify tournament context** - Grand Slams vs smaller tournaments affect predictions

### Multiple Match Files

You can have multiple CSV files for different tournaments:

```
data/future_matches/
├── us_open_2025.csv
├── wimbledon_2025.csv
├── atp_finals.csv
└── weekly_predictions.csv
```

When you run `python src/main.py`, you'll be prompted to select which file to use.

### Batch Processing (Non-Interactive)

For automated predictions without prompts:

```python
from predict import predict_from_csv

# Disable name resolution prompts
predictions = predict_from_csv(
    'data/future_matches/my_matches.csv',
    interactive_resolution=False
)

print(predictions)
```

### Common Issues

**Issue:** `FileNotFoundError: No model found`
- **Solution:** Train the model first with `python src/main.py` (answer 'y' to training)

**Issue:** `Unknown player 'Smith J.'`
- **Solution:** Check `data/raw/players_db.csv` for exact name format

**Issue:** Predictions seem random (all ~50%)
- **Solution:** Add `rank_1` and `rank_2` columns for better accuracy

**Issue:** `KeyError: 'elo_p1'`
- **Solution:** Ensure you have at least `date,player_1,player_2,surface` columns

---

## Understanding Confidence Scores

The `confidence` column shows how certain the model is:

| Confidence | Meaning | Example |
|------------|---------|---------|
| 50-55% | **Toss-up** | Very close match, could go either way |
| 55-65% | **Slight favorite** | One player has a small edge |
| 65-75% | **Clear favorite** | One player is expected to win |
| 75-85% | **Strong favorite** | Very likely outcome |
| 85%+ | **Heavy favorite** | Extremely confident prediction |

**Note:** Even 75% confidence means the underdog wins 1 in 4 times. Tennis is unpredictable!

---

## Advanced: All Available Columns

For maximum control (though most are derived automatically), you can include:

```csv
date,player_1,player_2,surface,tournament,round,best_of,series,court,rank_1,rank_2,series_level,is_outdoor,surf_fast,surf_hard,surf_clay,surf_grass,surf_carpet,best_of_3,best_of_5
```

Most users should stick to the recommended columns - the model derives ELO ratings, ranking ratios, and other features automatically.
