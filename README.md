# 🎾 US Open 2025 Match Predictor

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green.svg)](https://scikit-learn.org/)

A machine learning system for predicting professional tennis match outcomes using advanced ELO ratings, match history analysis, and gradient boosting algorithms.

## 📖 Project Overview

### What This Project Does

This system predicts the outcome of professional tennis matches by analyzing historical match data and player performance patterns. Given two players and match conditions (surface, tournament type, etc.), it calculates the probability of each player winning. For a more formal high-level overview of this project refer to notebooks/tennis_match_predictor.md.

### How It Works

The prediction pipeline consists of four main components:

1. **Feature Engineering**: Transforms raw match data into 38 meaningful features
2. **ELO Rating System**: Calculates dynamic player strength ratings that evolve over time
3. **XGBoost Model**: Uses gradient boosting to learn complex patterns from historical data
4. **Prediction Pipeline**: Applies the trained model to future matches

### Why This Approach

Tennis match prediction is inherently challenging due to:

- **Player form fluctuations** over time and surfaces
- **Head-to-head dynamics** between specific players
- **Surface specialization** (some players excel on clay, others on hard courts)
- **Tournament pressure** varying by event importance

This system addresses these challenges through:

- **Surface-specific ELO ratings** that track performance on different court types
- **Time-based training splits** that prevent data leakage and simulate real-world prediction scenarios
- **Historical match patterns** including win rates and recent form
- **Head-to-head statistics** for player matchup analysis

### Performance Context

**Industry Standard**: Professional tennis prediction platforms (with access to betting odds, injury reports, and real-time data) typically achieve **70-75% accuracy**.

**This Model's Performance**:

- **Training Data Accuracy**: 69.96% (on historical matches 2000-2018)
- **Future Prediction Accuracy**: 65.45% (on unseen matches 2023+)
- **Test AUC**: 0.7192 (good discrimination ability)

The 4.5% accuracy drop from training to future predictions is expected and healthy - it indicates the model generalizes well without overfitting to historical patterns.

### Model Validation & Testing

The system was rigorously tested using multiple approaches:

1. **Baseline Comparisons**:
   - **ELO-only predictions**: Used surface-specific ELO ratings alone
   - **Logistic Regression**: Traditional linear model with the same features
   - **XGBoost**: Final gradient boosting model (best performance)

2. **Time-Based Validation**:
   - **Training**: 2000-2018 (50,280 matches)
   - **Validation**: 2019-2022 (8,675 matches)
   - **Test**: 2023+ (7,019 matches) - completely unseen data

3. **Feature Importance Analysis**:
   - ELO difference is the strongest predictor (132.6 importance)
   - Ranking ratios and surface factors follow
   - 38 features selected from 49 generated features

The model's **65.45% accuracy on future matches** is competitive considering:

- Limited to publicly available historical data (65,974 matches)
- No access to real-time factors (injuries, recent form, weather)
- No betting market information or expert insights
- Purely algorithmic approach without human expertise

## 🚀 Features

- **Advanced ELO Rating System**: Surface-specific ELO ratings (Hard, Clay, Grass, Carpet)
- **Comprehensive Feature Engineering**: 38 features including rankings, match statistics, and historical performance
- **Time-Based Training**: Chronological data splits to prevent data leakage
- **XGBoost Model**: Optimized gradient boosting with early stopping and regularization
- **Match Prediction Pipeline**: End-to-end system from raw data to predictions
- **Professional Tournament Support**: ATP, Grand Slam, and other professional tournaments

## 📊 Model Performance

### Training vs. Future Performance

- **Training Accuracy**: 69.96% (historical matches 2000-2018)
- **Validation Accuracy**: 64.77% (validation set 2019-2022)
- **Test Accuracy**: 65.45% (future matches 2023+)
- **Test AUC**: 0.7192 (strong discrimination ability)

### Model Comparison Results

- **XGBoost**: 65.45% accuracy (final model)
- **Logistic Regression**: ~62% accuracy (baseline)
- **ELO-only**: ~58% accuracy (simple baseline)

### Industry Context

- **Professional Platforms**: 70-75% accuracy (with extensive data access)
- **This Model**: 65.45% accuracy (publicly available data only)
- **Random Baseline**: 50% accuracy

**Training Data**: 65,974 professional matches (2000-2025)  
**Best Trees Used**: 332 (with early stopping)

## 🏗️ Architecture

```text
├── data/
│   ├── raw/                    # Original tennis match data
│   ├── processed/              # Feature-engineered datasets
│   ├── future_matches/         # Prediction input files
│   └── outputs/                # Model artifacts and predictions
├── src/
│   ├── main.py                 # Complete ML pipeline
│   ├── features.py             # Feature engineering module
│   ├── model_xgb.py           # XGBoost training and evaluation
│   ├── predict.py             # Inference pipeline
│   ├── timesplits.py          # Time-based data splitting
│   └── elo/                   # ELO rating system
└── notebooks/                  # Analysis and documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd US-OPEN-PREDICTOR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **Core ML**: `xgboost`, `scikit-learn`, `pandas`, `numpy`
- **Visualization**: `matplotlib`
- **Data Processing**: `python-dateutil`, `pytz`

## 🎯 Quick Start

### 1. Train the Model

Run the complete ML pipeline:

```bash
python src/main.py
```

This will:

- ✅ Load 65,974 raw tennis matches
- ✅ Generate 38 engineered features (from 49 total features)
- ✅ Create time-based train/validation/test splits
- ✅ Train XGBoost model with early stopping (332 trees)
- ✅ Evaluate model performance and save results
- ✅ Optionally predict future matches

### 2. Make Predictions

Create a CSV file in `data/future_matches/` with your matches:

```csv
date,player_1,player_2,surface,tournament,round,best_of,series,court,series_level,is_outdoor,surf_fast,surf_hard,surf_clay,surf_grass,surf_carpet,best_of_3,best_of_5,rank_1,rank_2,rank_avg,rank_ratio,rank_diff,is_top10_match
2025-08-25,Djokovic N.,Alcaraz C.,Hard,US Open,1,5,ATP,Outdoor,1,1,1,1,0,0,0,0,1,7,2,4.5,0.286,5,1
```

Run predictions:

```bash
python -c "
import sys
sys.path.append('src')
from predict import predict_from_csv
predictions = predict_from_csv('data/future_matches/your_matches.csv')
print('✅ Predictions saved to data/outputs/predictions.csv')
"
```

### 3. View Results

Predictions are saved to `data/outputs/predictions.csv`:

```csv
date,player_1,player_2,predicted_winner,confidence,prob_p1_wins,prob_p2_wins
2025-08-25,Djokovic N.,Alcaraz C.,Alcaraz C.,0.73,0.27,0.73
```

## 📈 Feature Engineering

The system generates 38 features for each match from an initial 49 feature set:

### Top Features by Importance

1. **ELO Difference** (132.60) - Most important predictor
2. **Rank Ratio** (83.19) - Relative ranking strength
3. **Rank Difference** (24.19) - Absolute ranking gap
4. **Surface Fast** (10.86) - Fast court indicator
5. **Player ELO Ratings** - Individual ELO scores

### Feature Categories

- **ELO Ratings**: Overall and surface-specific ELO for both players
- **Rankings**: Current ATP rankings, ratios, and differences
- **Match Format**: Best-of-3 vs Best-of-5, tournament level
- **Surface Types**: Hard, Clay, Grass, Carpet indicators
- **Historical Performance**: Win rates over different time windows
- **Head-to-Head**: Historical matchup statistics
- **Recency**: Days since last match for both players

## 🎾 Supported Data Format

Input matches should include:

**Required Columns:**

- `date`: Match date (YYYY-MM-DD)
- `player_1`, `player_2`: Player names
- `surface`: Hard/Clay/Grass/Carpet
- `tournament`: Tournament name
- `rank_1`, `rank_2`: ATP rankings
- `best_of_3`, `best_of_5`: Match format (0/1)
- Surface indicators: `surf_hard`, `surf_clay`, etc.

## 📊 Model Details

### XGBoost Configuration

- **Objective**: Binary logistic regression
- **Evaluation Metric**: Log loss
- **Early Stopping**: 20 rounds (stopped at 332 trees)
- **Max Rounds**: 500
- **Regularization**: L1 and L2 penalties
- **Learning Rate**: Adaptive
- **Max Depth**: 6
- **Subsample**: 0.8

### Training Strategy

- **Time-Based Splits**: Chronological to prevent data leakage
  - **Training**: 2000-2018 (50,280 matches)
  - **Validation**: 2019-2022 (8,675 matches)
  - **Test**: 2023+ (7,019 matches)
- **Feature Importance**: ELO difference is the strongest predictor
- **Missing Data**: Median imputation with special handling for new players
- **NaN Handling**: Train=1,389, Val=156, Test=141 missing values imputed

### Performance Metrics

```text
TRAIN | LogLoss: 0.5671 | AUC: 0.7748 | Accuracy: 69.96%
VAL   | LogLoss: 0.6167 | AUC: 0.7126 | Accuracy: 64.77%
TEST  | LogLoss: 0.6132 | AUC: 0.7192 | Accuracy: 65.45%
```

**Interpretation**: The 4.5% drop from training to test accuracy indicates good generalization. The model doesn't overfit to historical patterns and maintains predictive power on completely unseen future matches.

## 🔧 Advanced Usage

### Custom Model Training

```python
from src.model_xgb import train_xgboost_pipeline

# Train with custom parameters
model, results = train_xgboost_pipeline()
print(f"Test AUC: {results['test']['auc']:.4f}")
```

### Feature Analysis

```python
from src.features import generate_features
import pandas as pd

# Generate features for custom data
df = pd.read_csv("your_data.csv")
features_df = generate_features(df)
print(f"Generated {len(features_df.columns)} features")
```

## 📝 Project Structure

- **`main.py`**: Complete pipeline orchestrator
- **`features.py`**: Feature engineering with ELO and match history
- **`model_xgb.py`**: XGBoost training with hyperparameter optimization
- **`predict.py`**: Inference pipeline for new matches
- **`timesplits.py`**: Time-aware data splitting
- **`elo/`**: ELO rating system implementation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 US Open Match Predictor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

- ATP and tennis data providers ([ATP Tennis Dataset (2000-2023)](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull/versions/829?resource=download))
- XGBoost development team
- Professional tennis statistical analysis community
- Open source machine learning ecosystem

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**⭐ Star this repository if you find it useful!**
