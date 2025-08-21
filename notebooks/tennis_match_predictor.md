# Tennis Match Outcome Prediction Using Machine Learning

## 1. Project Objective

The objective of this project is to predict outcomes of professional men’s and women’s singles tennis matches using machine learning models. The final system achieves **65.45% accuracy** on completely unseen matches (2023–2025).  

The project demonstrates:

- Feature engineering from raw historical data.  
- Time-based modeling to prevent future data leakage.  
- Comparison of linear and non-linear approaches (Logistic Regression vs. Gradient Boosting).  
- Application of predictive modeling principles to sports analytics.  

---

## 2. Dataset

- **Source**: Historical match data (65,000+ professional matches, 2000–2025).  
- **Scope**: Men’s and women’s singles only.  
- **Processing**:  
  - Raw data cleaned and structured in Excel.  
  - Exported to CSV.  
  - Enriched with 38 engineered features via Python (pandas, scikit-learn, custom scripts).  
- **Target Variable (Y)**: Binary outcome (Player 1 win = 1, Player 2 win = 0).  
- **Input Features (X)**: Numerical and categorical features capturing skill, form, and context.  

---

## 3. Feature Engineering

A total of 38 features were engineered. Key categories:

- **Elo Ratings**  
  - **Overall Elo**: Updated after every match, strong proxy for player skill.  
  - **Surface-Specific Elo**: Ratings maintained for hard, clay, and grass to capture surface specialization.  

- **Ranking Dynamics**  
  - Current ATP/WTA rank, rank differences, rank ratios.  

- **Form and Recency**  
  - Win rates in last 5 matches.  
  - Days since last match.  
  - Streak length (wins/losses).  

- **Contextual Features**  
  - Surface win rates.  
  - Tournament format (best of 3 vs. best of 5).  
  - Surface speed metrics.  
  - Player performance in tournament to date.  

- **Head-to-Head**  
  - Previous encounters between players.  
  - Head-to-head win rates.  

- **Volume and Consistency**  
  - Total matches played.  
  - Consistency indicators (variation in performance).  

---

## 4. Modeling Approach

To mimic real-world prediction conditions, time-based splits were used:

- **Training set**: Matches up to 2018.  
- **Validation set**: Matches from 2019–2022.  
- **Test set**: Matches from 2023–August 2025.  

This ensures models never train on data from the future relative to predictions.  

### Models Implemented

1. **Elo-Only Predictor** (baseline)  
   - Predicts winner as the higher Elo-rated player.  
   - Accuracy ~65%.  

2. **Logistic Regression**  
   - Linear model assigning feature weights.  
   - Provides interpretability: Elo difference and rank difference emerged as strongest predictors.  
   - Accuracy ~65%.  

3. **XGBoost (Gradient Boosting)**  
   - Ensemble of decision trees trained sequentially.  
   - Uses early stopping (332 trees) to prevent overfitting.  
   - Achieved higher **AUC (0.72)**, indicating stronger ranking performance than logistic regression.  

---

## 5. Performance Metrics

- **Training Accuracy**: 69.9%  
- **Test Accuracy**: 65.45%  
- **Test AUC**: 0.72  

Interpretation: The system generalizes reasonably well, balancing fit and predictive stability. Performance aligns with industry standards given feature limitations.  

---

## 6. Overfitting Mitigation

- **Time-Based Data Splits**: Prevented information leakage.  
- **Early Stopping**: Halted training when validation performance plateaued.  
- **Regularization**: L1 (Lasso) and L2 (Ridge) applied to reduce model complexity.  

Despite these measures, mild overfitting remained visible in training curves.  

---

## 7. Application Example: 2025 US Open Predictions

The model was used to predict Round 1 Men's Singles matches for the 2025 US Open:

- **Tiafoe F. vs. Nishioka Y.** → Predicted upset: Nishioka Y. (66.6%)  
- **Lehecka J. vs. Coric B.** → Predicted upset: Coric B. (64.8%)  
- **Machac T. vs. Nardi L.** → Predicted: Nardi L. (64.7%)  
- **Shapovalov D. vs. Fucsovics M.** → Predicted: Fucsovics M. (61.4%)  

> **Note**: Predictions are for demonstration only and **not betting advice**.

---

## 8. Skills Demonstrated

- **Data Engineering**: Cleaning, transformation, enrichment of 65k+ matches.  
- **Feature Engineering**: Designed 38 custom features across multiple domains.  
- **Machine Learning**: Logistic Regression and XGBoost model development.  
- **Model Evaluation**: Accuracy, AUC, and overfitting assessment.  
- **Best Practices**: Baseline benchmarks, chronological data splits, regularization.  

---

## 9. Limitations and Future Improvements

- **Current Limitation**: Data restricted to historical results and player statistics.  
- **Enhancements**:  
  - Incorporation of real-time data (injuries, fatigue, weather, travel).  
  - More granular recency features (Elo volatility, rolling averages).  
  - Tournament-pressure context (grand slam vs. challenger).  
  - Neural models for richer feature interactions.  

---

## 10. Repository

The complete pipeline, including feature generation scripts and models, is available at:  
**[github.com/charliec2004/ML-Tennis-Predictor](https://github.com/charliec2004/ML-Tennis-Predictor)**
