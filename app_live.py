"""
RailSarthi AI Dashboard
========================
Combines ML predictions from the demo models with live RailRadar API data.

Data sources:
  - Delay prediction model: XGBoost trained on synthetic data (models/*.joblib)
  - Crowd severity model: Random Forest trained on synthetic data
  - Real-time train status: RailRadar API /trains/{no}/live
  - Real-time train schedule: RailRadar API /trains/{no}

Run:
    streamlit run app_live.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import requests
import plotly.graph_objects as go
from datetime import date, datetime
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

RAILRADAR_KEY = os.getenv("RAILWAY_API_KEY", "")
RAILRADAR_URL = os.getenv("RAILWAY_API_BASE_URL", "https://api.railradar.in/v1")

st.set_page_config(
    page_title="RailSarthi AI - Predict. Plan. Travel Smarter.",
    layout="wide",
    page_icon="🚂",
)

# ---------------------------------------------------------------------------
# API Helper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def fetch_train_details(train_no: str):
    """Fetch train route/schedule from RailRadar (cached 5 min)."""
    headers = {"Authorization": f"Bearer {RAILRADAR_KEY}"}
    try:
        r = requests.get(f"{RAILRADAR_URL}/trains/{train_no}", headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


@st.cache_data(ttl=60)
def fetch_live_status(train_no: str):
    """Fetch live train status from RailRadar (cached 1 min)."""
    headers = {"Authorization": f"Bearer {RAILRADAR_KEY}"}
    try:
        r = requests.get(f"{RAILRADAR_URL}/trains/{train_no}/live", headers=headers, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Load synthetic data & models (MVP prediction backbone)
# ---------------------------------------------------------------------------

@st.cache_data
def load_synthetic_data():
    running_path = os.path.join(BASE_DIR, "train_running_data.csv")
    waitlist_path = os.path.join(BASE_DIR, "waitlist_data.csv")
    if os.path.exists(running_path) and os.path.exists(waitlist_path):
        running = pd.read_csv(running_path, dtype={"train_no": str})
        waitlist = pd.read_csv(waitlist_path, dtype={"train_no": str})
        return running, waitlist
    return None, None


@st.cache_resource
def load_models():
    try:
        delay_model = joblib.load(os.path.join(MODEL_DIR, "delay_model_integrated.joblib"))
        delay_cols = joblib.load(os.path.join(MODEL_DIR, "delay_integrated_feature_cols.joblib"))
        crowd_model = joblib.load(os.path.join(MODEL_DIR, "crowd_severity_model.joblib"))
        crowd_le = joblib.load(os.path.join(MODEL_DIR, "crowd_label_encoder.joblib"))
        crowd_cols = joblib.load(os.path.join(MODEL_DIR, "crowd_feature_cols.joblib"))
        return delay_model, delay_cols, crowd_model, crowd_le, crowd_cols
    except Exception:
        return None, None, None, None, None


models_loaded = False
running_df, waitlist_df = load_synthetic_data()
delay_model, delay_cols, crowd_model, crowd_le, crowd_cols = load_models()
if delay_model is not None:
    models_loaded = True


# ---------------------------------------------------------------------------
# List of popular trains (known to work with RailRadar)
# ---------------------------------------------------------------------------

POPULAR_TRAINS = {
    "12302": "New Delhi - Howrah Rajdhani Express",
    "12301": "Howrah - New Delhi Rajdhani Express",
    "12951": "Mumbai Central - New Delhi Tejas Rajdhani Express",
    "12309": "Rajendra Nagar - New Delhi Tejas Rajdhani Express",
    "12423": "Dibrugarh - New Delhi Rajdhani Express",
    "12429": "New Delhi - Lucknow Swarn Shatabdi Express",
    "12001": "New Delhi - Habibganj Shatabdi Express",
    "12553": "Vaishali Express",
    "12801": "Purushottam Express",
    "12433": "New Delhi - Guwahati Rajdhani Express",
}


# ---------------------------------------------------------------------------
# Temporal features
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("RailSarthi AI")
st.sidebar.caption("Predict. Plan. Travel Smarter.")
st.sidebar.markdown("---")

input_mode = st.sidebar.radio("Train Selection", ["Popular Trains", "Enter Train Number"])

if input_mode == "Popular Trains":
    options = [f"{no} - {name}" for no, name in POPULAR_TRAINS.items()]
    choice = st.sidebar.selectbox("Select Train", options=options)
    train_no = choice.split(" - ")[0].strip()
else:
    train_no = st.sidebar.text_input("Train Number (5 digits)", value="12302", max_chars=5)

journey_date = st.sidebar.date_input("Journey Date", value=date.today())
travel_class = st.sidebar.selectbox("Travel Class", options=["3A", "2A", "1A", "SL", "GEN"])

fetch_btn = st.sidebar.button("Fetch Live Data", type="primary", use_container_width=True)

api_status = "Connected" if RAILRADAR_KEY else "No API Key"
st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Status:** {api_status}")
st.sidebar.markdown(f"**Data Sources:**")
st.sidebar.markdown("- Live: RailRadar API v1")
st.sidebar.markdown("- Predictions: XGBoost + RandomForest (synthetic)")

# ---------------------------------------------------------------------------
# Temporal feature values
# ---------------------------------------------------------------------------

is_festival = is_festival_date(journey_date)
fog_season = int(journey_date.month in (12, 1, 2))
fog_intensity = fog_intensity_for(journey_date)
rain_intensity = rain_intensity_for(journey_date)
heat_index = heat_index_for(journey_date)
day_of_week = journey_date.weekday()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("RailSarthi AI - Railway Journey Intelligence")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

# Fetch train data from API
train_details = fetch_train_details(train_no)
live_status = fetch_live_status(train_no)

if train_details and train_details.get("success"):
    train_meta = train_details["data"]["train"]
    route_stops = train_details["data"].get("route", [])
    halt_stops = [s for s in route_stops if s.get("isHalt")]

    st.markdown(
        f"### {train_meta['name']} ({train_meta['number']})  \n"
        f"**Type:** {train_meta.get('type', 'N/A')} | "
        f"**Distance:** {train_meta.get('distance', 0)} km | "
        f"**Duration:** {train_meta.get('duration', 0)} min | "
        f"**Halts:** {train_meta.get('totalHalts', 0)} | "
        f"**Runs on:** {', '.join(train_meta.get('runDays', []))}"
    )

    # ----- Live Status Section -----
    if live_status and live_status.get("success"):
        live_data = live_status["data"]
        status = live_data.get("status", "unknown")
        delay_min = live_data.get("delayMinutes", 0)
        last_updated = live_data.get("lastUpdatedAt", "")

        col1, col2, col3, col4 = st.columns(4)

        status_color = {"running": "green", "not-started": "blue", "arrived": "gray"}.get(status, "orange")
        col1.metric("Status", status.replace("-", " ").title())
        col2.metric("Current Delay", f"{delay_min} min", delta=f"{delay_min} min" if delay_min > 0 else None, delta_color="inverse")

        next_halt = live_data.get("nextHalt", {})
        if next_halt:
            col3.metric("Next Halt", next_halt.get("name", "N/A"))

        if last_updated:
            try:
                updated_dt = datetime.fromisoformat(last_updated)
                col4.metric("Last Updated", updated_dt.strftime("%H:%M IST"))
            except Exception:
                col4.metric("Last Updated", last_updated[:16])

        # ----- Route Progress -----
        live_route = live_data.get("route", [])
        live_halts = [s for s in live_route if s.get("isHalt")]

        if live_halts:
            st.subheader("Live Route Progress")

            progress_data = []
            for stop in live_halts:
                stn = stop.get("station", {})
                actual_arr = stop.get("actualArrival")
                actual_dep = stop.get("actualDeparture")
                sched_arr = stop.get("scheduledArrival") or stop.get("arrival")
                sched_dep = stop.get("scheduledDeparture") or stop.get("departure")
                stop_delay = stop.get("delayDeparture", stop.get("delayArrival", 0)) or 0
                has_passed = stop.get("hasPassed", False) or stop.get("hasArrived", False)

                progress_data.append({
                    "Station": f"{stn.get('name', 'N/A')} ({stn.get('code', '')})",
                    "Scheduled Arr": sched_arr or "-",
                    "Scheduled Dep": sched_dep or "-",
                    "Actual Arr": actual_arr or "-",
                    "Actual Dep": actual_dep or "-",
                    "Delay (min)": stop_delay,
                    "Passed": "Yes" if has_passed else "No",
                })

            progress_df = pd.DataFrame(progress_data)
            st.dataframe(progress_df, use_container_width=True, hide_index=True)

            # Delay chart along live route
            delays = [d["Delay (min)"] for d in progress_data]
            stations = [d["Station"].split(" (")[0] for d in progress_data]

            if any(d != 0 for d in delays):
                fig_live = go.Figure()
                colors = ["#d62728" if d > 10 else "#ff7f0e" if d > 0 else "#2ca02c" for d in delays]
                fig_live.add_trace(go.Bar(
                    x=stations, y=delays,
                    marker_color=colors,
                    name="Live Delay (min)",
                ))
                fig_live.update_layout(
                    xaxis_title="Station",
                    yaxis_title="Delay (min)",
                    height=350,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_live, use_container_width=True)

    st.markdown("---")

    # ----- ML Prediction Section -----
    st.subheader("AI Delay Prediction (ML Model)")

    if models_loaded and running_df is not None:
        # Check if this train exists in synthetic data
        train_in_synthetic = running_df[running_df["train_no"] == train_no]

        if len(train_in_synthetic) > 0:
            train_specific = train_in_synthetic[
                ["station", "stop_sequence", "distance_covered_pct", "total_stops", "single_line"]
            ].drop_duplicates().sort_values("stop_sequence")

            train_hist_avg = train_in_synthetic["train_historical_avg_delay"].mean()
            crowd_pressure_val = waitlist_df.groupby("train_no")["wl_to_capacity_ratio"].mean().get(
                train_no, waitlist_df["wl_to_capacity_ratio"].mean()
            )

            train_info_row = running_df[running_df["train_no"] == train_no][
                ["train_type", "zone"]
            ].iloc[0]

            predicted_delays = []
            prev_delay = max(0, np.random.normal(5, 3))

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
                    "train_type": train_info_row["train_type"],
                    "zone": train_info_row["zone"],
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

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=plot_df["station"], y=plot_df["predicted_delay"],
                mode="lines+markers", line=dict(color="#d62728", width=3),
                marker=dict(size=10), name="Predicted Delay (min)"
            ))
            fig_pred.update_layout(
                xaxis_title="Station", yaxis_title="Predicted Delay (minutes)",
                height=400, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            final_delay = plot_df["predicted_delay"].iloc[-1]
            st.metric("Predicted Delay at Final Destination", f"{final_delay:.0f} min")

            st.caption(
                "Prediction based on XGBoost model trained on synthetic data encoding "
                "realistic Indian Railways patterns (fog season, festival rush, single-line congestion). "
                "Model will improve as real collected data replaces synthetic training data."
            )
        else:
            st.info(
                f"Train {train_no} is not in the synthetic training dataset. "
                f"ML delay prediction is available for demo trains only. "
                f"Live status from RailRadar is shown above."
            )
    else:
        st.warning(
            "ML models not loaded. Run `python src/delay_model.py` and `python src/crowd_model.py` "
            "to generate model files in the models/ directory."
        )

    # ----- Route Schedule Table -----
    st.markdown("---")
    st.subheader("Full Route Schedule")
    schedule_data = []
    for stop in halt_stops:
        stn = stop.get("station", {})
        schedule_data.append({
            "Stop #": stop.get("sequence"),
            "Station": f"{stn.get('name', '')} ({stn.get('code', '')})",
            "Arrival": stop.get("arrival", "-"),
            "Departure": stop.get("departure", "-"),
            "Distance (km)": stop.get("distance", 0),
            "Platform": stop.get("platform", "-"),
        })

    if schedule_data:
        st.dataframe(pd.DataFrame(schedule_data), use_container_width=True, hide_index=True)

else:
    if not RAILRADAR_KEY:
        st.error("RailRadar API key not configured. Add RAILWAY_API_KEY to your .env file.")
    else:
        st.warning(f"Could not fetch data for train {train_no}. Check the train number and try again.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "**RailSarthi AI** | Built with Streamlit + RailRadar API + XGBoost | "
    "Predict. Plan. Travel Smarter."
)
