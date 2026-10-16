"""
Synthetic Data Generator for Indian Railways Delay & Crowd Prediction MVP
===========================================================================
This script generates REALISTIC SYNTHETIC data that mimics patterns seen in
actual Indian Railways operations (fog-season delays, festival-season
overcrowding, cascading stop-to-stop delay, etc.).

WHY SYNTHETIC DATA: Live scraping of NTES/PNR services requires persistent
infrastructure (a scheduler running for weeks) and careful review of each
service's terms of use -- both are things your team should set up in your
own environment. This generator lets you build, test, and demo the ENTIRE
ML pipeline right now. Swap this module out for your real collected data
later; every downstream script (feature engineering, models, dashboard)
expects the same column schema, so the rest of the pipeline won't need to
change.

Run:
    python generate_synthetic_data.py

Outputs:
    train_running_data.csv   -> stop-by-stop running/delay records
    waitlist_data.csv        -> waitlist snapshots per train/date/class
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Reference data: trains, routes, stations
# ---------------------------------------------------------------------------

TRAINS = [
    {"train_no": "12301", "name": "Howrah Rajdhani", "type": "Rajdhani", "zone": "ER",
     "stops": ["Howrah", "Dhanbad", "Gaya", "Mughalsarai", "Kanpur", "Delhi"], "single_line": False},
    {"train_no": "12951", "name": "Mumbai Rajdhani", "type": "Rajdhani", "zone": "WR",
     "stops": ["Mumbai Central", "Vadodara", "Kota", "Sawai Madhopur", "Delhi"], "single_line": False},
    {"train_no": "12002", "name": "Bhopal Shatabdi", "type": "Shatabdi", "zone": "NR",
     "stops": ["Delhi", "Agra", "Gwalior", "Jhansi", "Bhopal"], "single_line": False},
    {"train_no": "15484", "name": "Danapur Express", "type": "Express", "zone": "ECR",
     "stops": ["Danapur", "Ara", "Buxar", "Mughalsarai", "Varanasi"], "single_line": True},
    {"train_no": "13202", "name": "Rajgir Express", "type": "Express", "zone": "ECR",
     "stops": ["Rajgir", "Bakhtiyarpur", "Patna", "Danapur", "Ara", "Buxar"], "single_line": True},
    {"train_no": "19038", "name": "Avadh Express", "type": "Superfast", "zone": "NR",
     "stops": ["Lucknow", "Ayodhya", "Sultanpur", "Varanasi", "Mughalsarai"], "single_line": True},
    {"train_no": "12392", "name": "Shramjeevi Express", "type": "Superfast", "zone": "ECR",
     "stops": ["Rajgir", "Patna", "Buxar", "Mughalsarai", "Delhi"], "single_line": True},
    {"train_no": "12561", "name": "Swatantrata Senani Express", "type": "Express", "zone": "NCR",
     "stops": ["Danapur", "Ara", "Buxar", "Mughalsarai", "Kanpur", "Delhi"], "single_line": False},
    {"train_no": "22349", "name": "Shaan-e-Bihar Express", "type": "Superfast", "zone": "ECR",
     "stops": ["Patna", "Danapur", "Ara", "Buxar", "Delhi"], "single_line": True},
    {"train_no": "12423", "name": "Dibrugarh Rajdhani", "type": "Rajdhani", "zone": "NFR",
     "stops": ["Dibrugarh", "Guwahati", "Katihar", "Patna", "Mughalsarai", "Delhi"], "single_line": False},
]

# Festival windows in 2026 (approximate, for feature realism)
FESTIVAL_WINDOWS = [
    ("2026-01-01", "2026-01-15"),   # New year / Makar Sankranti rush
    ("2026-03-01", "2026-03-06"),   # Holi
    ("2026-10-15", "2026-10-25"),   # Durga Puja / Dussehra
    ("2026-11-01", "2026-11-15"),   # Diwali / Chhath Puja
]

FOG_MONTHS = {12, 1, 2}  # Dec, Jan, Feb -> North Indian fog season

DATE_RANGE_START = datetime(2025, 10, 1)
DATE_RANGE_END = datetime(2026, 8, 30)


def is_festival(date):
    for start, end in FESTIVAL_WINDOWS:
        if datetime.strptime(start, "%Y-%m-%d") <= date <= datetime.strptime(end, "%Y-%m-%d"):
            return 1
    return 0


def random_dates(n):
    days = (DATE_RANGE_END - DATE_RANGE_START).days
    offsets = RNG.integers(0, days, size=n)
    return [DATE_RANGE_START + timedelta(days=int(o)) for o in offsets]


# ---------------------------------------------------------------------------
# 2. Train running / delay data (stop-by-stop)
# ---------------------------------------------------------------------------

def generate_running_data(n_journeys=1500):
    rows = []
    journey_dates = random_dates(n_journeys)

    for journey_id, journey_date in enumerate(journey_dates):
        train = TRAINS[RNG.integers(0, len(TRAINS))]
        stops = train["stops"]
        month = journey_date.month
        dow = journey_date.weekday()
        festival = is_festival(journey_date)
        fog_season = 1 if month in FOG_MONTHS else 0

        # Base "train personality": some trains chronically run late
        train_reliability = RNG.normal(0, 1)  # negative -> more punctual

        # Weather severity for this journey (fog, rain, heat)
        fog_intensity = 0.0
        if fog_season:
            fog_intensity = max(0, RNG.normal(0.5, 0.3))
        rain_intensity = max(0, RNG.normal(0.15, 0.15)) if month in {6, 7, 8, 9} else max(0, RNG.normal(0.05, 0.05))
        heat_index = max(0, RNG.normal(0.4, 0.2)) if month in {4, 5, 6} else max(0, RNG.normal(0.1, 0.1))

        cumulative_delay = max(0, RNG.normal(5, 5))  # starting delay at origin

        for stop_idx, station in enumerate(stops):
            distance_covered_pct = stop_idx / (len(stops) - 1)

            # Base delay increment per leg
            base_increment = RNG.normal(4, 6)

            # Congestion boost for single-line sections
            single_line_boost = 6 if train["single_line"] else 0
            # Fog boost (much bigger for North Indian winter nights)
            fog_boost = fog_intensity * RNG.normal(25, 8)
            # Rain boost
            rain_boost = rain_intensity * RNG.normal(12, 5)
            # Heat -> speed restrictions
            heat_boost = heat_index * RNG.normal(6, 3)
            # Festival congestion boost (more trains sharing track + longer halts)
            festival_boost = festival * RNG.normal(10, 4)
            # Weekend effect (slightly higher leisure travel congestion)
            weekend_boost = RNG.normal(2, 1) if dow >= 5 else 0
            # Reliability effect compounds with distance
            reliability_effect = train_reliability * RNG.normal(3, 1) * (1 + distance_covered_pct)

            increment = max(
                -5,  # trains can recover a little time
                base_increment + single_line_boost * RNG.uniform(0.5, 1.0)
                + fog_boost + rain_boost + heat_boost + festival_boost
                + weekend_boost + reliability_effect
            )

            cumulative_delay = max(0, cumulative_delay + increment)

            rows.append({
                "journey_id": journey_id,
                "train_no": train["train_no"],
                "train_name": train["name"],
                "train_type": train["type"],
                "zone": train["zone"],
                "single_line": int(train["single_line"]),
                "journey_date": journey_date.strftime("%Y-%m-%d"),
                "day_of_week": dow,
                "month": month,
                "is_festival": festival,
                "fog_season": fog_season,
                "fog_intensity": round(fog_intensity, 3),
                "rain_intensity": round(rain_intensity, 3),
                "heat_index": round(heat_index, 3),
                "station": station,
                "stop_sequence": stop_idx,
                "total_stops": len(stops),
                "distance_covered_pct": round(distance_covered_pct, 3),
                "delay_minutes": round(cumulative_delay, 1),
            })

    df = pd.DataFrame(rows)

    # Add lag features (delay at previous stop) computed per journey
    df["prev_stop_delay"] = df.groupby("journey_id")["delay_minutes"].shift(1).fillna(0)

    # Rolling average historical delay per train_no (simulating "this train's typical delay")
    train_avg_delay = df.groupby("train_no")["delay_minutes"].transform("mean")
    df["train_historical_avg_delay"] = train_avg_delay.round(1)

    return df


# ---------------------------------------------------------------------------
# 3. Waitlist / crowd data
# ---------------------------------------------------------------------------

CLASSES = ["SL", "3A", "2A", "GEN"]
CLASS_CAPACITY = {"SL": 72, "3A": 64, "2A": 46, "GEN": 90}  # per coach, rough approx


def generate_waitlist_data(n_records=2000):
    rows = []
    dates = random_dates(n_records)

    for i, journey_date in enumerate(dates):
        train = TRAINS[RNG.integers(0, len(TRAINS))]
        travel_class = CLASSES[RNG.integers(0, len(CLASSES))]
        festival = is_festival(journey_date)
        dow = journey_date.weekday()

        capacity = CLASS_CAPACITY[travel_class] * RNG.integers(2, 6)  # a few coaches of this class

        # Base demand pressure -- higher for festival, weekend, General class
        demand_base = RNG.normal(0.3, 0.15)
        if festival:
            demand_base += RNG.normal(0.4, 0.15)
        if dow >= 5:
            demand_base += RNG.normal(0.1, 0.05)
        if travel_class == "GEN":
            demand_base += RNG.normal(0.25, 0.1)  # General class chronically over-demanded
        if train["type"] in ("Rajdhani", "Shatabdi"):
            demand_base += RNG.normal(0.1, 0.05)  # premium trains fill fast

        demand_base = max(0.05, demand_base)

        # Waitlist snapshots at different lead times before departure
        wl_7day = max(0, int(RNG.normal(demand_base * capacity * 0.3, 5)))
        wl_3day = max(wl_7day, int(wl_7day + RNG.normal(demand_base * capacity * 0.35, 6)))
        wl_1day = max(wl_3day, int(wl_3day + RNG.normal(demand_base * capacity * 0.25, 5)))
        wl_final = max(wl_1day, int(wl_1day + RNG.normal(demand_base * capacity * 0.15, 4)))

        rac_count = max(0, int(RNG.normal(0.15 * wl_final, 3)))

        wl_growth_rate = (wl_final - wl_7day) / 7.0

        # Crowd severity label derived from final WL relative to capacity
        ratio = wl_final / max(capacity, 1)
        if ratio < 0.15:
            severity = "Low"
        elif ratio < 0.4:
            severity = "Medium"
        elif ratio < 0.8:
            severity = "High"
        else:
            severity = "Severe"

        rows.append({
            "record_id": i,
            "train_no": train["train_no"],
            "train_name": train["name"],
            "train_type": train["type"],
            "travel_class": travel_class,
            "journey_date": journey_date.strftime("%Y-%m-%d"),
            "day_of_week": dow,
            "is_festival": festival,
            "class_capacity": capacity,
            "wl_7day_before": wl_7day,
            "wl_3day_before": wl_3day,
            "wl_1day_before": wl_1day,
            "wl_final_chart": wl_final,
            "rac_count": rac_count,
            "wl_growth_rate_per_day": round(wl_growth_rate, 2),
            "wl_to_capacity_ratio": round(ratio, 3),
            "crowd_severity": severity,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic train running data...")
    running_df = generate_running_data(n_journeys=1500)
    running_df.to_csv("train_running_data.csv", index=False)
    print(f"  -> {len(running_df)} rows written to train_running_data.csv")

    print("Generating synthetic waitlist data...")
    waitlist_df = generate_waitlist_data(n_records=2000)
    waitlist_df.to_csv("waitlist_data.csv", index=False)
    print(f"  -> {len(waitlist_df)} rows written to waitlist_data.csv")

    print("\nCrowd severity class balance:")
    print(waitlist_df["crowd_severity"].value_counts())
    print("\nDone.")
