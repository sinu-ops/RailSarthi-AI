"""
Comprehensive Model Evaluation Script (Windows ASCII Safe)
==========================================================
Evaluates both the Delay Prediction (XGBoost Regressor) and
Crowd Severity (Random Forest Classifier) models.
Generates performance metrics, baseline comparisons, and evaluation summaries.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR

def evaluate_delay_model():
    print("=" * 70)
    print("[EVALUATION] DELAY PREDICTION MODEL (XGBoost Regressor)")
    print("=" * 70)

    try:
        model = joblib.load(MODEL_DIR / "delay_model_integrated.joblib")
        feature_cols = joblib.load(MODEL_DIR / "delay_integrated_feature_cols.joblib")
    except Exception as e:
        print(f"[ERROR] Could not load delay model: {e}")
        return

    running_path = DATA_DIR / "train_running_data.csv"
    waitlist_path = DATA_DIR / "waitlist_data.csv"

    if not running_path.exists():
        print("[ERROR] train_running_data.csv not found.")
        return

    running_df = pd.read_csv(running_path, dtype={"train_no": str})
    waitlist_df = pd.read_csv(waitlist_path, dtype={"train_no": str})

    crowd_lookup = waitlist_df.groupby("train_no")["wl_to_capacity_ratio"].mean().to_dict()
    running_df["crowd_pressure"] = running_df["train_no"].map(crowd_lookup).fillna(0.3)

    X_encoded = pd.get_dummies(running_df[[c for c in feature_cols if c in running_df.columns or c in ['train_type', 'zone']]])

    for c in feature_cols:
        if c not in X_encoded.columns:
            X_encoded[c] = 0
    X_encoded = X_encoded[feature_cols]

    y = running_df["delay_minutes"]

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Baseline: predicting median delay
    baseline_pred = np.full_like(y_test, y_test.median())
    base_mae = mean_absolute_error(y_test, baseline_pred)
    base_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

    print(f"Test Set Size: {len(y_test)} samples")
    print(f"Model Performance Metrics:")
    print(f"   - MAE:  {mae:.2f} minutes (Baseline median MAE: {base_mae:.2f} min)")
    print(f"   - RMSE: {rmse:.2f} minutes (Baseline median RMSE: {base_rmse:.2f} min)")
    print(f"   - R^2 Score: {r2:.4f}")
    print(f"   - Improvement over median baseline: {((base_mae - mae) / base_mae) * 100:.1f}% reduction in error")

    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        print("\nTop 5 Most Important Features for Delay Prediction:")
        for feat, imp in importances.head(5).items():
            print(f"   - {feat}: {imp:.4f}")

    print("\n")


def evaluate_crowd_model():
    print("=" * 70)
    print("[EVALUATION] CROWD SEVERITY CLASSIFIER (Random Forest)")
    print("=" * 70)

    try:
        model = joblib.load(MODEL_DIR / "crowd_severity_model.joblib")
        le = joblib.load(MODEL_DIR / "crowd_label_encoder.joblib")
        feature_cols = joblib.load(MODEL_DIR / "crowd_feature_cols.joblib")
    except Exception as e:
        print(f"[ERROR] Could not load crowd model: {e}")
        return

    waitlist_path = DATA_DIR / "waitlist_data.csv"
    if not waitlist_path.exists():
        print("[ERROR] waitlist_data.csv not found.")
        return

    df = pd.read_csv(waitlist_path, dtype={"train_no": str})
    X_encoded = pd.get_dummies(df[[c for c in feature_cols if c in df.columns or c in ['train_type', 'travel_class']]])

    for c in feature_cols:
        if c not in X_encoded.columns:
            X_encoded[c] = 0
    X_encoded = X_encoded[feature_cols]

    y = le.transform(df["crowd_severity"])

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    accuracy = np.mean(preds == y_test)

    print(f"Test Set Size: {len(y_test)} samples")
    print(f"Classifier Performance:")
    print(f"   - Accuracy: {accuracy * 100:.2f}%")
    print(f"   - Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_))

    print("Confusion Matrix (Rows = Actual, Columns = Predicted):")
    cm = confusion_matrix(y_test, preds)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)
    print("=" * 70)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    evaluate_delay_model()
    evaluate_crowd_model()
