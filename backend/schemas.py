"""
Pydantic API Request and Response Schemas
=========================================
Defines input validation and output types for all FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime


# --- Train Info & Schedule ---
class StationStopResponse(BaseModel):
    sequence: int
    station_code: str
    station_name: str
    is_halt: bool = True
    scheduled_arrival: Optional[str] = None
    scheduled_departure: Optional[str] = None
    distance_km: float = 0.0
    distance_covered_pct: float = 0.0
    platform: Optional[str] = None


class TrainScheduleResponse(BaseModel):
    train_no: str
    train_name: str
    train_type: str
    source_station: str
    destination_station: str
    distance_km: float
    total_halts: int
    data_source: str
    stops: List[StationStopResponse]


# --- Live Status ---
class LiveStatusResponse(BaseModel):
    train_no: str
    train_name: str
    status: str
    delay_minutes: int
    delay_formatted: str
    current_station: Optional[str] = None
    next_halt: Optional[str] = None
    last_updated: Optional[str] = None
    data_source: str


# --- Delay Prediction ---
class DelayPredictionRequest(BaseModel):
    train_no: str = Field(..., pattern=r"^\d{5}$", example="12302")
    journey_date: Optional[date] = Field(default_factory=date.today)
    initial_delay_minutes: float = Field(default=0.0, ge=0.0, le=1440.0)


class StopDelayPrediction(BaseModel):
    stop_sequence: int
    station_code: str
    station_name: str
    distance_km: float
    predicted_delay_minutes: float
    predicted_delay_formatted: str


class DelayPredictionResponse(BaseModel):
    train_no: str
    train_name: str
    journey_date: date
    destination_station: str
    final_predicted_delay_minutes: float
    final_predicted_delay_formatted: str
    is_punctual: bool  # Delay <= 15 min
    model_version: str = "XGBoost-Integrated-v1.0"
    stops: List[StopDelayPrediction]


# --- Route Reliability ---
class RouteReliabilityResponse(BaseModel):
    train_no: str
    train_name: str
    route_distance_km: float
    total_halts: int
    average_historical_delay_minutes: float
    median_historical_delay_minutes: float
    on_time_reliability_score: float  # Percentage 0-100%
    severe_delay_risk: str  # Low, Medium, High
    data_source: str


# --- Health Check ---
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    service: str
    models_loaded: bool
    cached_trains_count: int
