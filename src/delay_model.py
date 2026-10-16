"""
Train Delay Prediction Model
===============================
Predicts delay (in minutes) at each stop of a journey, using train,
route, temporal, and weather features. Also builds a SECOND version
of the model that additionally includes a derived "crowd_pressure"
feature, so we can directly test the integration hypothesis:

    Does knowing how crowded a train's demand is help predict delay?

Run:
    python delay_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "train_running_data.csv")
WAITLIST_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "waitlist_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_FEATURES = [
    "single_line", "day_of_week", "month", "is_festival", "fog_season",
    "fog_intensity", "rain_intensity", "heat_index", "stop_sequence",
    "total_stops", "distance_covered_pct", "prev_stop_delay",
    "train_historical_avg_delay",
]
CATEGORICAL_COLS = ["train_type", "zone"]
TARGET_COL = "delay_minutes"


def build_crowd_pressure_lookup():
    """Aggregate waitlist data into a simple per-train 'typical demand
    pressure' score that can be merged onto the running data as a proxy
    for how crowd-driven a train's delays might be. This is a simplified
    stand-in for a full model-to-model pipeline -- in a production
    version, you'd feed the CROWD MODEL'S live prediction in here instead
    of a historical average."""
    wdf = pd.read_csv(WAITLIST_PATH, dtype={"train_no": str})
    pressure = wdf.groupby("train_no")["wl_to_capacity_ratio"].mean().rename("crowd_pressure")
    return pressure.reset_index()


def load_and_prepare(include_crowd_feature=False):
    df = pd.read_csv(DATA_PATH, dtype={"train_no": str})

    if include_crowd_feature:
        pressure_lookup = build_crowd_pressure_lookup()
        df = df.merge(pressure_lookup, on="train_no", how="left")
        df["crowd_pressure"] = df["crowd_pressure"].fillna(df["crowd_pressure"].mean())

    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    feature_cols = BASE_FEATURES.copy()
    if include_crowd_feature:
        feature_cols.append("crowd_pressure")
    feature_cols += [c for c in df_encoded.columns
                      if c.startswith("train_type_") or c.startswith("zone_")]

    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]
    return X, y, feature_cols


def train_model(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n--- {label} ---")
    print(f"MAE:  {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R^2:  {r2:.4f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def run_integration_comparison():
    print("=" * 60)
    print("DELAY PREDICTION — BASELINE vs CROWD-INTEGRATED")
    print("=" * 60)

    # Baseline model (no crowd feature)
    X_base, y_base, base_cols = load_and_prepare(include_crowd_feature=False)
    model_base, metrics_base = train_model(X_base, y_base, "Baseline Model (no crowd feature)")

    # Crowd-integrated model
    X_crowd, y_crowd, crowd_cols = load_and_prepare(include_crowd_feature=True)
    model_crowd, metrics_crowd = train_model(X_crowd, y_crowd, "Integrated Model (with crowd_pressure feature)")

    print("\n" + "=" * 60)
    print("INTEGRATION RESULT")
    print("=" * 60)
    mae_delta = metrics_base["mae"] - metrics_crowd["mae"]
    rmse_delta = metrics_base["rmse"] - metrics_crowd["rmse"]
    print(f"MAE improvement from adding crowd feature:  {mae_delta:+.3f} minutes")
    print(f"RMSE improvement from adding crowd feature: {rmse_delta:+.3f} minutes")
    if mae_delta > 0:
        print("-> Crowd severity feature IMPROVED delay prediction accuracy.")
    else:
        print("-> Crowd severity feature did NOT improve accuracy in this run "
              "(expected with synthetic data where the two are only loosely linked; "
              "re-test once real collected data is used, where station dwell-time "
              "effects should be stronger).")

    # Feature importance for the integrated model
    importance = pd.Series(model_crowd.feature_importances_, index=crowd_cols).sort_values(ascending=False)
    print("\nTop 10 Feature Importances (Integrated Model):")
    print(importance.head(10))

    # Save both models
    joblib.dump(model_base, os.path.join(MODEL_DIR, "delay_model_baseline.joblib"))
    joblib.dump(model_crowd, os.path.join(MODEL_DIR, "delay_model_integrated.joblib"))
    joblib.dump(base_cols, os.path.join(MODEL_DIR, "delay_baseline_feature_cols.joblib"))
    joblib.dump(crowd_cols, os.path.join(MODEL_DIR, "delay_integrated_feature_cols.joblib"))
    print(f"\nModels saved to {MODEL_DIR}/")

    return model_base, model_crowd, metrics_base, metrics_crowd


if __name__ == "__main__":
    run_integration_comparison()
