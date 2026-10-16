"""
Crowd Severity Classification Model
=====================================
Trains a classifier to predict crowd severity (Low/Medium/High/Severe)
for a train/class/date combination, using waitlist growth patterns as
the primary signal.

Run:
    python crowd_model.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "waitlist_data.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "day_of_week", "is_festival", "class_capacity",
    "wl_7day_before", "wl_3day_before", "wl_1day_before",
    "rac_count", "wl_growth_rate_per_day",
]
CATEGORICAL_COLS = ["train_type", "travel_class"]
TARGET_COL = "crowd_severity"


def load_and_prepare():
    df = pd.read_csv(DATA_PATH, dtype={"train_no": str})

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)
    feature_cols = FEATURE_COLS + [c for c in df_encoded.columns
                                    if c.startswith("train_type_") or c.startswith("travel_class_")]

    X = df_encoded[feature_cols]
    y = df_encoded[TARGET_COL]
    return X, y, feature_cols, df


def train_and_evaluate():
    X, y, feature_cols, raw_df = load_and_prepare()

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("=" * 60)
    print("CROWD SEVERITY CLASSIFIER — EVALUATION")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Macro F1:", round(f1_score(y_test, y_pred, average="macro"), 4))
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=le.classes_, columns=le.classes_
    ))

    # Feature importance
    importance = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importance.head(10))

    # Save artifacts
    joblib.dump(clf, os.path.join(MODEL_DIR, "crowd_severity_model.joblib"))
    joblib.dump(le, os.path.join(MODEL_DIR, "crowd_label_encoder.joblib"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "crowd_feature_cols.joblib"))
    print(f"\nModel saved to {MODEL_DIR}/crowd_severity_model.joblib")

    return clf, le, feature_cols


def predict_crowd_severity(clf, le, feature_cols, input_dict):
    """Predict crowd severity for a single train/class/date scenario.
    input_dict should contain raw feature values; categoricals get one-hot
    encoded to match training columns (missing dummy columns filled with 0)."""
    row = pd.DataFrame([input_dict])
    row_encoded = pd.get_dummies(row)
    for col in feature_cols:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[feature_cols]
    pred = clf.predict(row_encoded)
    proba = clf.predict_proba(row_encoded)
    return le.inverse_transform(pred)[0], dict(zip(le.classes_, proba[0].round(3)))


if __name__ == "__main__":
    clf, le, feature_cols = train_and_evaluate()

    # Demo prediction
    sample = {
        "day_of_week": 5, "is_festival": 1, "class_capacity": 250,
        "wl_7day_before": 40, "wl_3day_before": 70, "wl_1day_before": 95,
        "rac_count": 15, "wl_growth_rate_per_day": 7.8,
        "train_type": "Express", "travel_class": "SL",
    }
    severity, proba = predict_crowd_severity(clf, le, feature_cols, sample)
    print(f"\nDemo prediction -> Crowd Severity: {severity}")
    print("Class probabilities:", proba)
