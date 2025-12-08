# Kaggle API Setup Guide

This project now fetches the latest ATP tennis dataset directly from Kaggle using the Kaggle API. Follow these steps to set it up:

## Prerequisites

1. **Kaggle Account**: Create a free account at [kaggle.com](https://www.kaggle.com)

2. **Install kagglehub**: Already included in `requirements.txt`
   ```bash
   pip install 'kagglehub[pandas-datasets]'
   ```

## Setup Steps

### 1. Get Your Kaggle API Credentials

1. Log in to [kaggle.com](https://www.kaggle.com)
2. Click on your profile picture (top right) → **Settings**
3. Scroll down to **API** section
4. Click **Create New Token**
5. This downloads a `kaggle.json` file containing your credentials

### 2. Install the Credentials

**macOS/Linux:**
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

### 3. Accept Dataset Terms

Visit the dataset page and accept the terms:
https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull

Click **Download** or **New Notebook** to accept the terms (you don't need to actually download it manually).

### 4. Test the Setup

Run the pipeline:
```bash
python src/main.py
```

You should see output like:
```
============================================================
FETCHING LATEST DATASET FROM KAGGLE
============================================================
Dataset: dissfya/atp-tennis-2000-2023daily-pull
Downloading latest version...
✓ Successfully downloaded 66,681 matches
✓ Date range: 2000-01-03 to 2025-11-16
```

## Troubleshooting

### Error: "401 Unauthorized"
- Make sure `~/.kaggle/kaggle.json` exists and has correct permissions (chmod 600)
- Verify your API token is valid (you can regenerate it in Kaggle Settings)

### Error: "403 Forbidden"
- You need to accept the dataset terms at the Kaggle dataset page
- Visit: https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull

### Error: "ModuleNotFoundError: No module named 'kagglehub'"
- Install: `pip install 'kagglehub[pandas-datasets]'`

## Benefits of Using Kaggle API

- **Always up-to-date**: Fetches the latest match data automatically
- **No manual downloads**: Dataset is cached locally after first download
- **Version tracking**: Kaggle datasets are versioned and reproducible
- **Cleaner repo**: No need to store large CSV files in the repository

## Offline Mode

If you need to work offline or already have the data:
1. Download the dataset manually from Kaggle
2. Place `atp_tennis.csv` in `data/raw/`
3. The pipeline will automatically fall back to the local file if the API fails

## Dataset Information

- **Source**: [ATP Tennis 2000-2023 (Daily Pull)](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull)
- **Updates**: Dataset is updated regularly with new matches
- **Size**: ~66,000+ professional ATP matches
- **Coverage**: 2000-present
