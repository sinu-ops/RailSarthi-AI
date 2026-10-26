"""
Dummy RailRadar Client for Testing and Quota Protection
=========================================================
Provides a mock implementation of RailRadarClient that never makes
real HTTP requests. Used with FastAPI dependency overrides to protect
API quotas during development, testing, and demonstration.
"""

from typing import Dict, Any, Optional
from src.data_pipeline.api_client import RailRadarClient


class DummyRailRadarClient:
    """
    Mock client matching the RailRadarClient interface.
    Returns simulated data consistent with real RailRadar response schemas,
    allowing full endpoint and dashboard testing without consuming API quota.
    """

    def __init__(self, save_raw: bool = False):
        self.save_raw = save_raw
        print("[DUMMY] Dummy RailRadar Client initialized (zero API quota consumption)")

    def _make_train_name(self, number: str) -> str:
        # Return realistic Indian Railway names for popular train numbers
        names = {
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
        return names.get(str(number).strip(), f"Train {number}")

    def get_train_details(self, train_no: str) -> Dict[str, Any]:
        """Simulate /trains/{train_no} response."""
        number = str(train_no).strip()
        train_name = self._make_train_name(number)
        return {
            "success": True,
            "data": {
                "train": {
                    "number": number,
                    "name": train_name,
                    "type": "Rajdhani Express" if "Rajdhani" in train_name or "Shatabdi" in train_name else "Express",
                    "category": "Premium",
                    "source": {"code": "NDLS", "name": "New Delhi"},
                    "destination": {"code": "HWH", "name": "Howrah Jn"},
                    "runDays": ["mon", "tue", "wed", "thu", "sat", "sun"],
                    "distance": 1452.3,
                    "duration": 1025,
                    "avgSpeed": 85,
                    "maxSpeed": 94.3,
                    "totalHalts": 9,
                    "returnTrain": "12301",
                    "classes": ["1A", "2A", "3A"],
                    "isPrsBookable": True,
                },
                "route": [
                    {"sequence": 1, "station": {"code": "NDLS", "name": "New Delhi", "lat": 28.64177, "lng": 77.22027},
                     "isHalt": True, "platform": "14", "departure": "16:50", "distance": 0},
                    {"sequence": 2, "station": {"code": "CNB", "name": "Kanpur Central", "lat": 26.44992, "lng": 80.33187},
                     "isHalt": True, "arrival": "21:30", "departure": "21:35", "platform": "6", "distance": 440.1},
                    {"sequence": 3, "station": {"code": "PRYJ", "name": "Prayagraj Jn", "lat": 25.4358, "lng": 81.8463},
                     "isHalt": True, "arrival": "23:41", "departure": "23:43", "platform": "4", "distance": 634.5},
                    {"sequence": 4, "station": {"code": "DDU", "name": "Pt. Deen Dayal Upadhyaya Jn", "lat": 25.15201, "lng": 83.00808},
                     "isHalt": True, "arrival": "01:33", "departure": "01:40", "platform": "1", "distance": 787.3},
                    {"sequence": 5, "station": {"code": "HWH", "name": "Howrah Jn", "lat": 22.58382, "lng": 88.34221},
                     "isHalt": True, "arrival": "03:15", "departure": None, "distance": 1452.3},
                ],
            },
            "meta": {"timestamp": "2026-10-23T15:29:37+05:30", "source": "dummy", "executionTime": 0, "quota_consumed": 0},
        }

    def get_live_status(self, train_no: str) -> Dict[str, Any]:
        """Simulate /trains/{train_no}/live response."""
        number = str(train_no).strip()
        status_options = ["running", "not-started", "arrived"]
        # Make deterministic based on train number
        status_idx = int(number[-1]) % 3
        status = status_options[status_idx]
        delay = int(number[-2:]) % 25  # Deterministic delay between 0-24 min

        return {
            "success": True,
            "data": {
                "trainNumber": number,
                "trainName": self._make_train_name(number),
                "startDate": "2026-10-23",
                "lastUpdatedAt": "2026-10-23T15:29:37+05:30",
                "status": status,
                "train": {"number": number, "name": self._make_train_name(number)},
                "isLive": True,
                "trackingMode": "realtime",
                "nextHalt": {"name": "Kanpur Central", "code": "CNB"},
                "delayMinutes": delay,
                "currentLocation": {"name": "New Delhi", "code": "NDLS", "lat": 28.64177, "lng": 77.22027},
                "route": [
                    {"sequence": 1, "station": {"code": "NDLS", "name": "New Delhi"},
                     "arrival": None, "departure": "16:50", "distance": 0, "isReversal": False},
                ],
            },
            "meta": {"timestamp": "2026-10-23T15:29:37+05:30", "source": "dummy", "quota_consumed": 0},
        }


# Helper function for FastAPI Depends override (see Lesson 1 teaching notes)
def get_dummy_client():
    """Returns the Dummy Client for testing / quota protection."""
    return DummyRailRadarClient(save_raw=False)
