"""
Data Parser & Transformer
=========================
Parses raw JSON responses from RailRadar API into clean, structured Parquet tables
matching the RailSarthi AI schema requirements.
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RailwayDataParser:
    """Parses raw JSON train data into structured Parquet format."""

    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def parse_train_schedules(self) -> pd.DataFrame:
        """
        Parses all train schedule/route JSON files into a clean `train_routes` DataFrame
        and saves as Parquet.
        """
        pattern = str(self.raw_dir / "train_*.json")
        files = glob.glob(pattern)

        if not files:
            logger.warning("No train schedule JSON files found in %s", self.raw_dir)
            return pd.DataFrame()

        route_rows = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                if not payload.get("success"):
                    continue

                data = payload.get("data", {})
                train_meta = data.get("train", {})
                train_no = str(train_meta.get("number"))
                train_name = train_meta.get("name")
                train_type = train_meta.get("type")
                zone = train_meta.get("zone", "NR") # Default fallback
                distance_total = train_meta.get("distance", 0.0)

                route_stops = data.get("route", [])

                # Filter only halts or record all stops with sequence
                for idx, stop in enumerate(route_stops):
                    station = stop.get("station", {})
                    station_code = station.get("code")
                    station_name = station.get("name")
                    is_halt = stop.get("isHalt", False)

                    # Only keep halts for primary train route table, or keep all if needed.
                    # Halts represent actual passenger stops.
                    if not is_halt and len(route_stops) > 50:
                        continue # Skip non-halt minor waypoints to keep model features clean

                    arrival = stop.get("arrival")
                    departure = stop.get("departure")
                    distance_km = stop.get("distance", 0.0)
                    platform = stop.get("platform")

                    route_rows.append({
                        "train_no": train_no,
                        "train_name": train_name,
                        "train_type": train_type,
                        "zone": zone,
                        "source_station": train_meta.get("source", {}).get("code"),
                        "destination_station": train_meta.get("destination", {}).get("code"),
                        "stop_sequence": stop.get("sequence"),
                        "station_code": station_code,
                        "station_name": station_name,
                        "is_halt": is_halt,
                        "scheduled_arrival": arrival,
                        "scheduled_departure": departure,
                        "distance_km": distance_km,
                        "distance_covered_pct": (distance_km / distance_total * 100) if distance_total > 0 else 0.0,
                        "platform": platform
                    })

            except Exception as e:
                logger.error("Failed to parse file %s: %s", file_path, e)

        df = pd.DataFrame(route_rows)
        if not df.empty:
            out_path = self.processed_dir / "train_routes.parquet"
            df.to_parquet(out_path, index=False)
            logger.info("Saved %d route stops to %s", len(df), out_path)

        return df

    def parse_live_observations(self) -> pd.DataFrame:
        """
        Parses live status JSON files into `train_observations` DataFrame
        and saves as Parquet.
        """
        pattern = str(self.raw_dir / "live_*.json")
        files = glob.glob(pattern)

        if not files:
            logger.warning("No live status JSON files found in %s", self.raw_dir)
            return pd.DataFrame()

        obs_rows = []

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                if not payload.get("success"):
                    continue

                data = payload.get("data", {})
                train_no = str(data.get("trainNumber"))
                train_name = data.get("trainName")
                journey_date = data.get("startDate")
                status = data.get("status")
                delay_minutes = data.get("delayMinutes", 0)
                last_updated = data.get("lastUpdatedAt")

                current_loc = data.get("currentLocation", {})
                current_station = current_loc.get("stationCode") or current_loc.get("name")

                next_halt = data.get("nextHalt", {})
                next_station = next_halt.get("code") or next_halt.get("name")

                obs_rows.append({
                    "train_no": train_no,
                    "train_name": train_name,
                    "journey_date": journey_date,
                    "collected_at": last_updated or datetime.now().isoformat(),
                    "current_station": current_station,
                    "delay_minutes": delay_minutes,
                    "next_station": next_station,
                    "status": status,
                    "source": "rail_radar_live"
                })

            except Exception as e:
                logger.error("Failed to parse live observation file %s: %s", file_path, e)

        df = pd.DataFrame(obs_rows)
        if not df.empty:
            out_path = self.processed_dir / "train_observations.parquet"
            df.to_parquet(out_path, index=False)
            logger.info("Saved %d live observations to %s", len(df), out_path)

        return df


if __name__ == "__main__":
    parser = RailwayDataParser()
    print("Parsing train schedules into Parquet...")
    routes_df = parser.parse_train_schedules()
    print(f"Routes DataFrame shape: {routes_df.shape}")
    if not routes_df.empty:
        print(routes_df.head(3))

    print("\nParsing live observations into Parquet...")
    obs_df = parser.parse_live_observations()
    print(f"Observations DataFrame shape: {obs_df.shape}")
    if not obs_df.empty:
        print(obs_df.head(3))
