# AI-Based Train Delay & Crowd Prediction System — MVP

A working prototype for the final year project: predicts train delay stop-by-stop
and passenger crowd severity, and tests whether the two are connected.

## What's in this MVP

| Component | File | What it does |
|---|---|---|
| Synthetic data generator | `data/generate_synthetic_data.py` | Creates realistic train running + waitlist data |
| Crowd severity model | `src/crowd_model.py` | Random Forest classifier (Low/Medium/High/Severe) |
| Delay prediction model | `src/delay_model.py` | XGBoost regressor, baseline vs. crowd-integrated, with the integration comparison built in |
| Dashboard | `app.py` | Streamlit app tying both models together |

## Why synthetic data?

Live scraping of NTES/PNR services needs a scheduler running continuously for
weeks (to capture waitlist growth over time) and a check of each service's
terms of use — that's infrastructure for your team to set up in your own
environment, not something to fake in a demo session. The generator here
encodes the **real patterns** documented in your synopsis (fog-season delays,
festival-season crowding, single-line congestion, cascading stop-to-stop
delay) so every downstream script — features, models, dashboard — is
already built and tested against the correct schema. When your team's
scraper starts producing real CSVs with the same column names, you swap the
file and nothing else changes.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Run order

```bash
# Step 1 — generate the synthetic dataset
cd data
python generate_synthetic_data.py
cd ..

# Step 2 — train the crowd severity classifier
cd src
python crowd_model.py

# Step 3 — train the delay prediction models (baseline + integrated)
python delay_model.py
cd ..

# Step 4 — launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Pick a train, date, and class
in the sidebar to see predicted delay at each stop and predicted crowd
severity.

## Current results (on synthetic data)

- **Crowd severity classifier**: ~90% accuracy, macro F1 ~0.88
- **Delay prediction (baseline)**: MAE ≈ 6.3 minutes, R² ≈ 0.96
- **Integration test**: adding a crowd-pressure feature gave a negligible
  improvement on this synthetic data (expected — the synthetic generator
  only loosely couples the two). **Re-run this test once real collected
  data is in place** — actual station dwell-time effects from overcrowding
  should show a stronger signal, and this is one of your report's key
  results either way (a null result honestly reported is still a result).

## Moving from MVP to your real project

1. **Replace the data**: build your scraper (per your synopsis's Data
   Collection module) to produce CSVs matching the same columns as
   `train_running_data.csv` and `waitlist_data.csv`. Everything downstream
   keeps working.
2. **Add the sequential model**: the current delay model treats each stop as
   an independent row with a `prev_stop_delay` feature — a reasonable MVP
   approach. For your final submission, consider adding a proper LSTM that
   processes each journey as a full sequence (mentioned as a stretch goal in
   your synopsis).
3. **Expand routes/trains**: the generator currently covers 10 sample trains
   across a few zones — extend this list, or better, replace it entirely
   once real data covers your chosen zone(s).
4. **Tune models further**: try hyperparameter search (GridSearchCV /
   Optuna) once you have real data volumes, and compare additional
   algorithms as a "model comparison" section in your report.

## Folder structure

```
railway_mvp/
├── data/
│   ├── generate_synthetic_data.py
│   ├── train_running_data.csv      (generated)
│   └── waitlist_data.csv           (generated)
├── src/
│   ├── crowd_model.py
│   └── delay_model.py
├── models/                          (generated .joblib files)
├── app.py
├── requirements.txt
└── README.md
```
