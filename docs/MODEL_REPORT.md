# RailSarthi AI — Model Evaluation & Performance Report

**Project:** RailSarthi AI (Railway Journey Intelligence Platform)  
**Author / Engineer:** Final Year Engineering Student  
**Models Evaluated:** 
1. **Delay Prediction Model**: XGBoost Regressor (Stop-by-stop cascading delay prediction)
2. **Crowd Severity Classifier**: Random Forest Classifier (Passenger demand & waitlist risk)

---

## 1. Executive Summary

RailSarthi AI employs predictive machine learning models to solve railway passenger uncertainty regarding train delays and crowd congestion. This report details the evaluation metrics, baseline comparisons, feature importance rankings, and operational performance of both ML models.

---

## 2. Delay Prediction Model (XGBoost Regressor)

### Objective
Predict the exact delay in minutes for a train at every stop along its route before departure, taking into account temporal features, weather seasonality, festival surges, single-line bottlenecks, and cascading previous-stop delays.

### Evaluation Methodology
- **Test Split**: 20% chronologically held-out journey samples ($N = 1,620$ test instances).
- **Baselines**: Compared against a naive baseline model predicting the median historical delay for each route.
- **Metrics**: 
  - **MAE (Mean Absolute Error)**: Average magnitude of prediction errors in minutes.
  - **RMSE (Root Mean Squared Error)**: Penalizes larger delay outliers.
  - **$R^2$ Score**: Proportion of variance in delay explained by the model.

### Quantitative Results

| Metric | RailSarthi XGBoost Model | Naive Median Baseline | Improvement / Gain |
|---|---|---|---|
| **MAE** | **6.25 minutes** | 26.99 minutes | **76.9% reduction in error** |
| **RMSE** | **7.84 minutes** | 39.92 minutes | **80.4% reduction in error** |
| **$R^2$ Score** | **0.9586** | 0.0000 | **Near-perfect fit (95.8% variance explained)** |

### Top 5 Feature Importances
1. **`prev_stop_delay`** ($0.5824$): Proves that delay propagation from the previous stop is the single strongest predictor of downstream delay.
2. **`fog_intensity`** ($0.0831$): Quantifies winter visibility slowdowns in Northern/North-Central railway zones.
3. **`fog_season`** ($0.0705$): Seasonal indicator capturing general winter schedule adjustments.
4. **`is_festival`** ($0.0576$): Captures network congestion during peak festival holiday windows (Diwali, Holi, Chhath).
5. **`distance_covered_pct`** ($0.0424$): Tracks cumulative schedule drift as a train progresses along its route.

---

## 3. Crowd Severity Classifier (Random Forest)

### Objective
Classify passenger demand and overcrowding risk into discrete severity tiers (`Low`, `Medium`, `High`, `Severe`) based on waitlist growth rates, RAC counts, class capacity ratios, and festival seasonality.

### Evaluation Methodology
- **Test Split**: 20% held-out waitlist samples ($N = 400$ test instances).
- **Metrics**: Accuracy, Macro-Averaged F1-Score, Precision, Recall, and Confusion Matrix.

### Quantitative Results
- **Overall Accuracy**: **92.50%**
- **Macro-Averaged F1-Score**: **0.8971**

### Class-Wise Performance (Classification Report)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **Low** | 0.77 | 0.86 | 0.81 | 28 |
| **Medium** | 0.92 | 0.91 | 0.92 | 139 |
| **High** | 0.95 | 0.95 | 0.95 | 195 |
| **Severe** | 0.92 | 0.89 | 0.91 | 38 |

### Confusion Matrix Analysis
```
          Predicted
          High  Low  Medium  Severe
Actual
High      185    0       7       3
Low         0   24       4       0
Medium      5    7     127       0
Severe      4    0       0      34
```
- **Observations**: The confusion matrix demonstrates minimal off-diagonal misclassifications. High and Medium congestion categories are separated with high precision ($92\%\text{--}95\%$), ensuring passengers receive reliable crowd warnings before booking.

---

## 4. Operational Insights & Interview Talking Points

1. **Why XGBoost over Deep Learning (LSTM)?**
   - XGBoost trains in under 5 seconds, requires minimal hyperparameter tuning, provides explainable feature importances, and achieves an outstanding $R^2$ of $0.9586$ on tabular stop-sequence data.
2. **Addressing Train-Serve Skew:**
   - Feature engineering logic is encapsulated in modular functions (`src/data_pipeline/data_manager.py`) shared identically between training notebooks and the FastAPI/Streamlit inference serving layer.
3. **Handling API Quota Constraints:**
   - Implemented a cache-first local memory architecture (`TrainDataManager`) that serves frequent train queries from local JSON memory ($0$ API cost), preserving RailRadar quotas for live verification.
