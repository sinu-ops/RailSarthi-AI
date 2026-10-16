"""
Indian Railways AI Dashboard — MVP
=====================================
Interactive Streamlit dashboard that ties together the Delay Prediction
model and the Crowd Severity model. Select a train, date, and class to
see predicted delay at each stop and predicted crowd severity.

Run:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import date

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

st.set_page_config(page_title="IR AI Dashboard — MVP", layout="wide", page_icon="🚆")

# ---------------------------------------------------------------------------
# Load data & models (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_data():
    running = pd.read_csv(os.path.join(DATA_DIR, "train_running_data.csv"), dtype={"train_no": str})
    waitlist = pd.read_csv(os.path.join(DATA_DIR, "waitlist_data.csv"), dtype={"train_no": str})
    return running, waitlist


@st.cache_resource
def load_models():
    delay_model = joblib.load(os.path.join(MODEL_DIR, "delay_model_integrated.joblib"))
    delay_cols = joblib.load(os.path.join(MODEL_DIR, "delay_integrated_feature_cols.joblib"))
    crowd_model = joblib.load(os.path.join(MODEL_DIR, "crowd_severity_model.joblib"))
    crowd_le = joblib.load(os.path.join(MODEL_DIR, "crowd_label_encoder.joblib"))
    crowd_cols = joblib.load(os.path.join(MODEL_DIR, "crowd_feature_cols.joblib"))
    return delay_model, delay_cols, crowd_model, crowd_le, crowd_cols


running_df, waitlist_df = load_data()
delay_model, delay_cols, crowd_model, crowd_le, crowd_cols = load_models()

TRAIN_INFO = running_df[["train_no", "train_name", "train_type", "zone", "single_line"]].drop_duplicates()
TRAIN_STOPS = running_df.groupby("train_no")["station"].apply(list).to_dict()
TRAIN_STOP_SEQ = running_df.groupby("train_no").apply(
    lambda g: g.sort_values("stop_sequence")[["station", "stop_sequence", "distance_covered_pct"]].drop_duplicates()
).reset_index(drop=True)

CROWD_LOOKUP = waitlist_df.groupby("train_no")["wl_to_capacity_ratio"].mean()

# ---------------------------------------------------------------------------
# Sidebar — user inputs
# ---------------------------------------------------------------------------

st.sidebar.title("🚆 Journey Details")

train_choice = st.sidebar.selectbox(
    "Select Train",
    options=TRAIN_INFO["train_no"] + " — " + TRAIN_INFO["train_name"],
)
train_no = train_choice.split(" — ")[0]
train_row = TRAIN_INFO[TRAIN_INFO["train_no"] == train_no].iloc[0]

journey_date = st.sidebar.date_input("Journey Date", value=date(2026, 11, 5))
travel_class = st.sidebar.selectbox("Travel Class", options=["SL", "3A", "2A", "GEN"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "This MVP runs on synthetic data generated to mimic realistic Indian "
    "Railways delay and demand patterns (fog season, festival rush, single-line "
    "congestion, etc.). Swap in real collected data via the same schema to move "
    "from demo to production."
)

# ---------------------------------------------------------------------------
# Derived temporal / seasonal features from the chosen date
# ---------------------------------------------------------------------------

FESTIVAL_WINDOWS = [
    (date(2026, 1, 1), date(2026, 1, 15)),
    (date(2026, 3, 1), date(2026, 3, 6)),
    (date(2026, 10, 15), date(2026, 10, 25)),
    (date(2026, 11, 1), date(2026, 11, 15)),
]


def is_festival_date(d):
    return int(any(start <= d <= end for start, end in FESTIVAL_WINDOWS))


def fog_intensity_for(d):
    return 0.5 if d.month in (12, 1, 2) else 0.0


def rain_intensity_for(d):
    return 0.2 if d.month in (6, 7, 8, 9) else 0.05


def heat_index_for(d):
    return 0.4 if d.month in (4, 5, 6) else 0.1


is_festival = is_festival_date(journey_date)
fog_season = int(journey_date.month in (12, 1, 2))
fog_intensity = fog_intensity_for(journey_date)
rain_intensity = rain_intensity_for(journey_date)
heat_index = heat_index_for(journey_date)
day_of_week = journey_date.weekday()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("AI-Based Train Delay & Crowd Prediction — MVP Dashboard")
st.markdown(
    f"**{train_row['train_name']}** ({train_no}) · {train_row['train_type']} · "
    f"Zone: {train_row['zone']} · Date: {journey_date.strftime('%d %b %Y')} · Class: {travel_class}"
)

col_left, col_right = st.columns([1.4, 1])

# ---------------------------------------------------------------------------
# LEFT: Delay prediction across all stops
# ---------------------------------------------------------------------------

with col_left:
    st.subheader("📍 Predicted Delay by Stop")

    stops = TRAIN_STOP_SEQ[TRAIN_STOP_SEQ["station"].isin(
        running_df[running_df["train_no"] == train_no]["station"].unique()
    )]
    # Get ordered unique stops for this train
    train_specific = running_df[running_df["train_no"] == train_no][
        ["station", "stop_sequence", "distance_covered_pct", "total_stops", "single_line"]
    ].drop_duplicates().sort_values("stop_sequence")

    train_hist_avg = running_df[running_df["train_no"] == train_no]["train_historical_avg_delay"].mean()
    crowd_pressure_val = CROWD_LOOKUP.get(train_no, waitlist_df["wl_to_capacity_ratio"].mean())

    predicted_delays = []
    prev_delay = max(0, np.random.normal(5, 3))  # starting delay assumption at origin

    for _, row in train_specific.iterrows():
        input_row = {
            "single_line": row["single_line"],
            "day_of_week": day_of_week,
            "month": journey_date.month,
            "is_festival": is_festival,
            "fog_season": fog_season,
            "fog_intensity": fog_intensity,
            "rain_intensity": rain_intensity,
            "heat_index": heat_index,
            "stop_sequence": row["stop_sequence"],
            "total_stops": row["total_stops"],
            "distance_covered_pct": row["distance_covered_pct"],
            "prev_stop_delay": prev_delay,
            "train_historical_avg_delay": train_hist_avg,
            "crowd_pressure": crowd_pressure_val,
            "train_type": train_row["train_type"],
            "zone": train_row["zone"],
        }
        input_df = pd.DataFrame([input_row])
        input_encoded = pd.get_dummies(input_df)
        for c in delay_cols:
            if c not in input_encoded.columns:
                input_encoded[c] = 0
        input_encoded = input_encoded[delay_cols]

        pred_delay = max(0, delay_model.predict(input_encoded)[0])
        predicted_delays.append(pred_delay)
        prev_delay = pred_delay

    plot_df = pd.DataFrame({
        "station": train_specific["station"].values,
        "predicted_delay": predicted_delays,
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["station"], y=plot_df["predicted_delay"],
        mode="lines+markers", line=dict(color="#d62728", width=3),
        marker=dict(size=10), name="Predicted Delay (min)"
    ))
    fig.update_layout(
        xaxis_title="Station", yaxis_title="Predicted Delay (minutes)",
        height=420, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    final_delay = plot_df["predicted_delay"].iloc[-1]
    st.metric("Predicted Delay at Final Destination", f"{final_delay:.0f} min")

# ---------------------------------------------------------------------------
# RIGHT: Crowd severity prediction
# ---------------------------------------------------------------------------

with col_right:
    st.subheader("👥 Predicted Crowd Severity")

    wl_hist = waitlist_df[
        (waitlist_df["train_no"] == train_no) & (waitlist_df["travel_class"] == travel_class)
    ]

    if len(wl_hist) > 0:
        wl_7 = wl_hist["wl_7day_before"].mean()
        wl_3 = wl_hist["wl_3day_before"].mean()
        wl_1 = wl_hist["wl_1day_before"].mean()
        rac = wl_hist["rac_count"].mean()
        capacity = wl_hist["class_capacity"].mean()
        growth = wl_hist["wl_growth_rate_per_day"].mean()
    else:
        wl_7, wl_3, wl_1, rac, capacity, growth = 20, 40, 55, 8, 250, 5.0

    # Adjust for chosen date's festival status
    festival_multiplier = 1.6 if is_festival else 1.0
    wl_7 *= festival_multiplier
    wl_3 *= festival_multiplier
    wl_1 *= festival_multiplier
    growth *= festival_multiplier

    crowd_input = {
        "day_of_week": day_of_week, "is_festival": is_festival, "class_capacity": capacity,
        "wl_7day_before": wl_7, "wl_3day_before": wl_3, "wl_1day_before": wl_1,
        "rac_count": rac, "wl_growth_rate_per_day": growth,
        "train_type": train_row["train_type"], "travel_class": travel_class,
    }
    crowd_row = pd.DataFrame([crowd_input])
    crowd_encoded = pd.get_dummies(crowd_row)
    for c in crowd_cols:
        if c not in crowd_encoded.columns:
            crowd_encoded[c] = 0
    crowd_encoded = crowd_encoded[crowd_cols]

    severity_pred = crowd_model.predict(crowd_encoded)[0]
    severity_label = crowd_le.inverse_transform([severity_pred])[0]
    proba = crowd_model.predict_proba(crowd_encoded)[0]

    severity_colors = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728", "Severe": "#8b0000"}
    st.markdown(
        f"<div style='padding:20px;border-radius:10px;background-color:{severity_colors[severity_label]}22;"
        f"border:2px solid {severity_colors[severity_label]};text-align:center;'>"
        f"<h2 style='color:{severity_colors[severity_label]};margin:0;'>{severity_label}</h2>"
        f"<p style='margin:0;'>Predicted Crowd Severity</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Class probabilities:**")
    proba_df = pd.DataFrame({
        "Severity": crowd_le.classes_,
        "Probability": proba,
    }).sort_values("Probability", ascending=True)

    fig2 = go.Figure(go.Bar(
        x=proba_df["Probability"], y=proba_df["Severity"], orientation="h",
        marker_color=[severity_colors[s] for s in proba_df["Severity"]],
    ))
    fig2.update_layout(height=250, margin=dict(t=10, b=10), xaxis_title="Probability")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**Estimated final waitlist:** ~{int(wl_1)} passengers")
    st.markdown(f"**RAC count:** ~{int(rac)}")

st.markdown("---")
st.caption(
    "MVP built on synthetic data. Models: XGBoost Regressor (delay prediction, integrated with "
    "crowd-pressure feature) + Random Forest Classifier (crowd severity). "
    "Replace `data/generate_synthetic_data.py` output with real collected NTES/PNR data "
    "to move this from prototype to production."
)
