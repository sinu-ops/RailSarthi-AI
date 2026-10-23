"""
Local Data Manager & Cache Engine
==================================
Handles local search across cached raw JSON files and CSV databases.
Implements the cache-first pattern:
1. Search local memory (JSON files in data/raw + train_running_data.csv)
2. If train not found locally, fetch from RailRadar API and save to local memory.
"""

import os
import glob
import json
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR

class TrainDataManager:
    def __init__(self):
        self.raw_dir = RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._load_csv_indices()

    def _load_csv_indices(self):
        running_path = DATA_DIR / "train_running_data.csv"
        waitlist_path = DATA_DIR / "waitlist_data.csv"

        self.running_df = None
        self.waitlist_df = None

        if running_path.exists():
            self.running_df = pd.read_csv(running_path, dtype={"train_no": str})
        if waitlist_path.exists():
            self.waitlist_df = pd.read_csv(waitlist_path, dtype={"train_no": str})

    def get_all_searchable_trains(self) -> List[Dict[str, str]]:
        """Returns list of train dicts with train_no and train_name for autocomplete/search."""
        trains_dict = {}

        # 1. From JSON cache in data/raw
        for f in glob.glob(str(self.raw_dir / "train_*.json")):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    d = json.load(file)
                train = d.get("data", {}).get("train", {})
                num = str(train.get("number", "")).strip()
                name = train.get("name", "").strip()
                if num and num not in trains_dict:
                    trains_dict[num] = name
            except Exception:
                continue

        # 2. From CSV dataset
        if self.running_df is not None:
            csv_trains = self.running_df[["train_no", "train_name"]].drop_duplicates()
            for _, row in csv_trains.iterrows():
                num = str(row["train_no"]).strip()
                name = str(row["train_name"]).strip()
                if num and num not in trains_dict:
                    trains_dict[num] = name

        return [{"number": k, "name": v} for k, v in sorted(trains_dict.items())]

    def search_local_train(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search for train by 5-digit number or name substring in local cache.
        Returns parsed route & info dict if found locally, else None.
        """
        query_str = str(query).strip().lower()
        if not query_str:
            return None

        # Check JSON files first (has richest coordinates, platforms, halts)
        for f in sorted(glob.glob(str(self.raw_dir / "train_*.json")), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    d = json.load(file)
                data = d.get("data", {})
                train = data.get("train", {})
                num = str(train.get("number", "")).lower()
                name = str(train.get("name", "")).lower()

                if query_str == num or query_str in name or query_str in num:
                    return {
                        "source": "local_cache",
                        "train_no": str(train.get("number")),
                        "train_name": train.get("name"),
                        "train_type": train.get("type", "Express"),
                        "source_station": train.get("source", {}).get("name", "Origin"),
                        "destination_station": train.get("destination", {}).get("name", "Destination"),
                        "distance_km": train.get("distance", 0),
                        "total_halts": train.get("totalHalts", len(data.get("route", []))),
                        "classes": train.get("availableClasses", ["1A", "2A", "3A", "SL"]),
                        "route": data.get("route", []),
                        "raw_data": data
                    }
            except Exception:
                continue

        # Check CSV running_df if not in JSON
        if self.running_df is not None:
            matches = self.running_df[
                (self.running_df["train_no"].str.lower() == query_str) |
                (self.running_df["train_name"].str.lower().str.contains(query_str, na=False))
            ]
            if not matches.empty:
                first_row = matches.iloc[0]
                train_no = str(first_row["train_no"])
                unique_stops = matches[
                    ["station", "stop_sequence", "distance_covered_pct", "total_stops", "single_line"]
                ].drop_duplicates().sort_values("stop_sequence")

                route_formatted = []
                for _, stn in unique_stops.iterrows():
                    route_formatted.append({
                        "sequence": int(stn["stop_sequence"]),
                        "station": {"name": stn["station"], "code": stn["station"][:4].upper()},
                        "isHalt": True,
                        "distance": float(stn["distance_covered_pct"]),
                        "distance_covered_pct": float(stn["distance_covered_pct"]),
                        "single_line": int(stn["single_line"]),
                        "total_stops": int(stn["total_stops"])
                    })

                return {
                    "source": "local_dataset",
                    "train_no": train_no,
                    "train_name": str(first_row["train_name"]),
                    "train_type": str(first_row["train_type"]),
                    "source_station": route_formatted[0]["station"]["name"],
                    "destination_station": route_formatted[-1]["station"]["name"],
                    "distance_km": float(matches["distance_covered_pct"].max()),
                    "total_halts": len(route_formatted),
                    "classes": ["SL", "3A", "2A", "GEN"],
                    "route": route_formatted,
                    "raw_data": None
                }

        return None

    def get_or_fetch_train(self, query: str, client=None) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Cache-first strategy:
        1. Checks local memory.
        2. If missing and client provided, fetches from RailRadar API and caches locally.
        Returns (train_data, source_type) where source_type is 'local_cache', 'api_fetch', or 'not_found'.
        """
        local_data = self.search_local_train(query)
        if local_data:
            return local_data, "Saved Memory"

        # Not found locally: fetch from API if client is available
        if client is not None and str(query).strip().isdigit() and len(str(query).strip()) == 5:
            try:
                res = client.get_train_details(str(query).strip())
                if res and res.get("success"):
                    # Re-search local train now that client saved it to data/raw/
                    fetched = self.search_local_train(query)
                    if fetched:
                        return fetched, "Live RailRadar API"
            except Exception as e:
                logger.error("Live fetch failed for %s: %s", query, e)

        return None, "Not Found"

    def get_live_observation(self, train_no: str, client=None) -> Dict[str, Any]:
        """Gets live status from API if client available, else looks in local cache."""
        # Check local live JSON first
        for f in sorted(glob.glob(str(self.raw_dir / f"live_{train_no}_*.json")), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as file:
                    d = json.load(file)
                if d.get("success"):
                    return {"source": "Saved Memory", "data": d.get("data", {})}
            except Exception:
                continue

        # If not in cache and client is available, fetch live
        if client is not None:
            try:
                res = client.get_live_status(train_no)
                if res and res.get("success"):
                    return {"source": "Live RailRadar API", "data": res.get("data", {})}
            except Exception:
                pass

        return {"source": "Unavailable", "data": {}}
