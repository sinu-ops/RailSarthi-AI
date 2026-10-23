"""
RailSarthi AI - Intelligent Journey Dashboard
==============================================
Predict. Plan. Travel Smarter.

Features:
- Searchbar with instant local memory search (Train Number or Name).
- Fallback live API fetch if not in local cache (conserving requests).
- Interactive delay prediction across stops with minutes & hours formatting.
- Interactive crowd prediction based on waiting list & class demand.
- Clean, modern, high-contrast UI with Plotly visualizations.
"""

import os
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import date, datetime
from dotenv import load_dotenv

# Import local data manager and API client
from src.data_pipeline.data_manager import TrainDataManager
from src.data_pipeline.api_client import RailRadarClient

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

st.set_page_config(
    page_title="RailSarthi AI — Journey Intelligence",
    layout="wide",
    page_icon="🚆",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Custom Clean CSS Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .memory-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #DEF7EC;
        color: #03543F;
        margin-bottom: 8px;
    }
    .api-badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background-color: #E1EFFE;
        color: #1E429F;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data & Model Loaders (Cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_data_manager():
    return TrainDataManager()

@st.cache_resource
def get_api_client():
    try:
        return RailRadarClient(save_raw=True)
    except Exception:
        return None

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

dm = get_data_manager()
api_client = get_api_client()
delay_model, delay_cols, crowd_model, crowd_le, crowd_cols = load_models()

# ---------------------------------------------------------------------------
# Helper: Format Minutes into Clean "Xh Ym" string
# ---------------------------------------------------------------------------
def format_delay_time(minutes: float) -> str:
    m = max(0, int(round(minutes)))
    if m < 60:
        return f"{m} mins"
    hours = m // 60
    rem_mins = m % 60
    if rem_mins == 0:
        return f"{hours} hr"
    return f"{hours}h {rem_mins}m ({m} mins)"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("<div class='main-header'>🚆 RailSarthi AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict. Plan. Travel Smarter — Journey Intelligence & Crowd Analytics</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Search Bar Section
# ---------------------------------------------------------------------------
search_col1, search_col2, search_col3 = st.columns([3, 1.2, 1.2])

all_trains = dm.get_all_searchable_trains()
train_options = [f"{t['number']} — {t['name']}" for t in all_trains]

with search_col1:
    search_selection = st.selectbox(
        "🔍 Search Train by Number or Name (Saved in Memory):",
        options=train_options,
        index=2 if len(train_options) > 2 else 0, # Default to 12302 Rajdhani if present
        help="Select from locally saved trains or type to search."
    )
    selected_train_no = search_selection.split(" — ")[0].strip()

with search_col2:
    journey_date = st.date_input("📅 Journey Date", value=date.today())

with search_col3:
    travel_class = st.selectbox("💺 Travel Class", options=["3A", "2A", "1A", "SL", "GEN"])

# Optional custom train input for live fetching if not in list
with st.expander("🔎 Search custom train number not in saved list"):
    custom_train = st.text_input("Enter any 5-digit Train Number:", max_chars=5, placeholder="e.g. 12952")
    if custom_train and custom_train.strip().isdigit():
        selected_train_no = custom_train.strip()

# ---------------------------------------------------------------------------
# Fetch Train Data (Cache First -> API Fallback)
# ---------------------------------------------------------------------------
train_data, data_source = dm.get_or_fetch_train(selected_train_no, client=api_client)

if train_data:
    st.markdown("---")

    # Train Info Banner
    banner_col1, banner_col2 = st.columns([3, 1])
    with banner_col1:
        st.subheader(f"{train_data['train_name']} ({train_data['train_no']})")
        st.markdown(
            f"**Route:** {train_data['source_station']} ➔ {train_data['destination_station']} | "
            f"**Type:** {train_data.get('train_type', 'Express')} | "
            f"**Distance:** {train_data.get('distance_km', 0):.0f} km | "
            f"**Halts:** {train_data.get('total_halts', 'N/A')}"
        )
    with banner_col2:
        if data_source == "Saved Memory":
            st.markdown("<div class='memory-badge'>💾 Retrieved from Saved Memory</div>", unsafe_allow_html=True)
            st.caption("Zero API requests consumed")
        else:
            st.markdown("<div class='api-badge'>🌐 Live RailRadar API</div>", unsafe_allow_html=True)
            st.caption("Cached for future searches")

    # Fetch real live observation if available
    live_obs = dm.get_live_observation(selected_train_no, client=api_client)
    live_info = live_obs.get("data", {})
    current_delay = live_info.get("delayMinutes", 0)
    current_status = live_info.get("status", "running").replace("-", " ").title()
    next_stop = live_info.get("nextHalt", {}).get("name", "N/A") if isinstance(live_info.get("nextHalt"), dict) else "N/A"

    # -----------------------------------------------------------------------
    # Top KPI Metrics Row
    # -----------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Live Status", current_status)
    with m2:
        st.metric("Current Delay", format_delay_time(current_delay))
    with m3:
        st.metric("Next Stop", next_stop if next_stop != "N/A" else "En Route")
    with m4:
        st.metric("Travel Class", travel_class)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Two Columns: Left = Delay by Station Graph | Right = Crowd Analytics
    # -----------------------------------------------------------------------
    col_delay, col_crowd = st.columns([1.5, 1])

    # -----------------------------------------------------------------------
    # 1. DELAY PREDICTION & STOP-BY-STOP GRAPH
    # -----------------------------------------------------------------------
    with col_delay:
        st.subheader("📍 Stop-by-Stop Delay Analysis")

        raw_route = train_data.get("route", [])
        # Filter for halts
        halt_stops = [s for s in raw_route if s.get("isHalt", True)]
        if not halt_stops:
            halt_stops = raw_route

        # Build prediction sequence
        stations = []
        distances_pct = []
        delays_predicted = []

        # Start propagation with current known live delay or mild baseline
        prev_delay = float(current_delay) if current_delay > 0 else 5.0

        is_festival = int(journey_date.month in [10, 11, 3, 1])
        fog_season = int(journey_date.month in [12, 1, 2])
        fog_intensity = 0.5 if fog_season else 0.0
        rain_intensity = 0.2 if journey_date.month in [6, 7, 8, 9] else 0.05
        heat_index = 0.4 if journey_date.month in [4, 5, 6] else 0.1
        day_of_week = journey_date.weekday()
        total_halts = len(halt_stops)

        for idx, stop in enumerate(halt_stops):
            stn_name = stop.get("station", {}).get("name") or stop.get("station_name") or f"Stop #{idx+1}"
            dist_pct = float(stop.get("distance_covered_pct", (idx / max(1, total_halts - 1)) * 100))
            single_line = int(stop.get("single_line", 0))

            if delay_model is not None and delay_cols is not None:
                feat_dict = {
                    "single_line": single_line,
                    "day_of_week": day_of_week,
                    "month": journey_date.month,
                    "is_festival": is_festival,
                    "fog_season": fog_season,
                    "fog_intensity": fog_intensity,
                    "rain_intensity": rain_intensity,
                    "heat_index": heat_index,
                    "stop_sequence": idx + 1,
                    "total_stops": total_halts,
                    "distance_covered_pct": dist_pct,
                    "prev_stop_delay": prev_delay,
                    "train_historical_avg_delay": 18.5,
                    "crowd_pressure": 0.45,
                    "train_type": train_data.get("train_type", "Rajdhani"),
                    "zone": "NR"
                }
                feat_df = pd.DataFrame([feat_dict])
                encoded = pd.get_dummies(feat_df)
                for c in delay_cols:
                    if c not in encoded.columns:
                        encoded[c] = 0
                encoded = encoded[delay_cols]
                pred = max(0.0, float(delay_model.predict(encoded)[0]))
            else:
                # Fallback cascade calculation
                pred = prev_delay + np.random.uniform(0.5, 3.0)

            stations.append(stn_name)
            distances_pct.append(dist_pct)
            delays_predicted.append(pred)
            prev_delay = pred

        # Destination delay
        dest_delay = delays_predicted[-1] if delays_predicted else 0
        st.markdown(f"**Estimated Destination Delay:** <span style='font-size:1.25rem; font-weight:700; color:#DC2626;'>{format_delay_time(dest_delay)}</span>", unsafe_allow_html=True)

        # Plotly Interactive Delay Curve
        delay_fig = go.Figure()

        # Add shaded area under curve
        delay_fig.add_trace(go.Scatter(
            x=stations,
            y=delays_predicted,
            mode='lines+markers',
            name='Projected Delay',
            line=dict(color='#DC2626', width=3, shape='spline'),
            marker=dict(size=8, color='#991B1B', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(220, 38, 38, 0.08)',
            hovertemplate="<b>%{x}</b><br>Delay: %{y:.1f} mins (%{customdata})<extra></extra>",
            customdata=[format_delay_time(d) for d in delays_predicted]
        ))

        # Add punctual reference line (15 min IR threshold)
        delay_fig.add_hline(
            y=15, line_dash="dash", line_color="#10B981",
            annotation_text="Punctual Threshold (15m)", annotation_position="bottom right"
        )

        delay_fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=50),
            xaxis=dict(title="Stations along route", tickangle=-35),
            yaxis=dict(title="Delay (Minutes)", gridcolor="#F1F5F9"),
            plot_bgcolor="white",
            hovermode="x unified"
        )
        st.plotly_chart(delay_fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # 2. CROWD & WAITING LIST PREDICTION
    # -----------------------------------------------------------------------
    with col_crowd:
        st.subheader("👥 Crowd & Demand Severity")

        # Query waitlist dataset for typical numbers
        wl_df = dm.waitlist_df
        if wl_df is not None:
            train_wl = wl_df[
                (wl_df["train_no"] == selected_train_no) &
                (wl_df["travel_class"] == travel_class)
            ]
            if not train_wl.empty:
                avg_wl = float(train_wl["wl_1day_before"].mean())
                avg_rac = float(train_wl["rac_count"].mean())
                avg_cap = float(train_wl["class_capacity"].mean())
                growth_rate = float(train_wl["wl_growth_rate_per_day"].mean())
            else:
                avg_wl, avg_rac, avg_cap, growth_rate = 45.0, 12.0, 240.0, 4.5
        else:
            avg_wl, avg_rac, avg_cap, growth_rate = 45.0, 12.0, 240.0, 4.5

        # Festival adjustment
        if is_festival:
            avg_wl *= 1.5
            avg_rac *= 1.3

        # Model Inference
        if crowd_model is not None and crowd_cols is not None:
            crowd_feat = {
                "day_of_week": day_of_week,
                "is_festival": is_festival,
                "class_capacity": avg_cap,
                "wl_7day_before": avg_wl * 0.5,
                "wl_3day_before": avg_wl * 0.8,
                "wl_1day_before": avg_wl,
                "rac_count": avg_rac,
                "wl_growth_rate_per_day": growth_rate,
                "train_type": train_data.get("train_type", "Rajdhani"),
                "travel_class": travel_class
            }
            c_df = pd.DataFrame([crowd_feat])
            c_encoded = pd.get_dummies(c_df)
            for c in crowd_cols:
                if c not in c_encoded.columns:
                    c_encoded[c] = 0
            c_encoded = c_encoded[crowd_cols]

            pred_class = crowd_model.predict(c_encoded)[0]
            severity_label = crowd_le.inverse_transform([pred_class])[0]
            probabilities = crowd_model.predict_proba(c_encoded)[0]
            classes = crowd_le.classes_
        else:
            severity_label = "Medium"
            classes = ["Low", "Medium", "High", "Severe"]
            probabilities = [0.15, 0.60, 0.20, 0.05]

        # Severity Card with Color Accent
        color_map = {
            "Low": "#10B981",       # Green
            "Medium": "#F59E0B",    # Amber
            "High": "#EF4444",      # Red
            "Severe": "#7F1D1D"     # Dark Red
        }
        theme_color = color_map.get(severity_label, "#F59E0B")

        st.markdown(
            f"""
            <div style="background-color: {theme_color}15; border: 2px solid {theme_color}; border-radius: 10px; padding: 18px; text-align: center; margin-bottom: 16px;">
                <span style="font-size: 0.9rem; font-weight: 600; color: #4B5563;">PREDICTED PASSENGER DENSITY</span>
                <h2 style="color: {theme_color}; margin: 4px 0 0 0; font-size: 2rem;">{severity_label} Risk</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Horizontal Probability Distribution Chart
        prob_df = pd.DataFrame({
            "Severity": classes,
            "Probability": [p * 100 for p in probabilities]
        }).sort_values("Probability", ascending=True)

        prob_fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Severity"],
            orientation='h',
            marker=dict(color=[color_map.get(s, "#6B7280") for s in prob_df["Severity"]]),
            hovertemplate="Severity: <b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>"
        ))

        prob_fig.update_layout(
            height=200,
            margin=dict(l=10, r=20, t=10, b=30),
            xaxis=dict(title="Probability (%)", range=[0, 100]),
            yaxis=dict(title=""),
            plot_bgcolor="white"
        )
        st.plotly_chart(prob_fig, use_container_width=True)

        # Waitlist Indicators
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.metric("Avg Waiting List", f"WL {int(avg_wl)}")
        with w_col2:
            st.metric("RAC Allocation", f"RAC {int(avg_rac)}")

    # -----------------------------------------------------------------------
    # Detailed Station Table
    # -----------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Station Halt Details & Timings")

    table_rows = []
    for idx, stop in enumerate(halt_stops):
        stn = stop.get("station", {})
        stn_name = stn.get("name") or stop.get("station_name") or "Station"
        stn_code = stn.get("code") or stop.get("station_code") or ""
        arr = stop.get("arrival") or stop.get("scheduled_arrival") or "—"
        dep = stop.get("departure") or stop.get("scheduled_departure") or "—"
        dist = stop.get("distance", stop.get("distance_km", 0))
        plt = stop.get("platform", "—")
        pred_d = delays_predicted[idx] if idx < len(delays_predicted) else 0

        table_rows.append({
            "Stop #": idx + 1,
            "Station": f"{stn_name} ({stn_code})",
            "Scheduled Arr": arr,
            "Scheduled Dep": dep,
            "Distance": f"{dist:.1f} km" if dist else "0.0 km",
            "Platform": str(plt) if plt else "—",
            "Predicted Delay": format_delay_time(pred_d)
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

else:
    st.error(f"❌ Could not find train '{selected_train_no}' in local memory or through the API.")
    st.info("Tip: Choose from the list of saved trains in memory (e.g. 12302, 12301, 12951, 12423, 12309, 12002).")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("RailSarthi AI · Cache-First Railway Intelligence · Hybrid XGBoost & Random Forest Models")
