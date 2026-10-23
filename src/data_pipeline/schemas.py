"""
Pydantic Schemas for Data Validation
=====================================
Validates API responses and ensures data quality
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from typing import Optional, List
import re


class StationStop(BaseModel):
    """Single station stop in a train's route"""
    station_code: str = Field(..., min_length=2, max_length=10)
    station_name: str
    arrival: Optional[str] = None  # HH:MM format or None for first station
    departure: Optional[str] = None  # HH:MM format or None for last station
    stop_number: int = Field(..., ge=1)
    distance_km: float = Field(..., ge=0)
    platform: Optional[str] = None

    @validator('station_code')
    def validate_station_code(cls, v):
        if not v.isupper():
            raise ValueError('Station code must be uppercase')
        return v


class TrainSchedule(BaseModel):
    """Complete train schedule/route information"""
    train_no: str = Field(..., pattern=r'^\d{5}$')
    train_name: str
    train_type: Optional[str] = None  # Rajdhani, Shatabdi, Express, etc.
    source_station: str
    destination_station: str
    stations: List[StationStop]
    distance_km: float = Field(..., ge=0)
    duration_hours: Optional[float] = None
    zone: Optional[str] = None  # NR, ER, WR, etc.

    @validator('train_no')
    def validate_train_no(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError('Train number must be 5 digits')
        return v


class LiveTrainStatus(BaseModel):
    """Live running status of a train"""
    train_no: str = Field(..., pattern=r'^\d{5}$')
    train_name: str
    journey_date: date
    current_station: Optional[str] = None
    current_station_name: Optional[str] = None
    delay_minutes: int = Field(..., ge=0, le=1440)  # Max 24 hours
    next_station: Optional[str] = None
    next_station_name: Optional[str] = None
    status: str  # 'running', 'arrived', 'not_started', 'cancelled'
    last_updated: datetime
    collected_at: datetime = Field(default_factory=datetime.now)

    @validator('status')
    def validate_status(cls, v):
        valid_statuses = {'running', 'arrived', 'not_started', 'cancelled', 'delayed', 'terminated'}
        if v.lower() not in valid_statuses:
            raise ValueError(f'Status must be one of {valid_statuses}')
        return v.lower()


class TrainInfo(BaseModel):
    """Basic train metadata"""
    train_no: str = Field(..., pattern=r'^\d{5}$')
    train_name: str
    train_type: Optional[str] = None
    source_station: str
    destination_station: str
    runs_on_days: Optional[List[str]] = None  # ['Mon', 'Wed', 'Fri']
    zone: Optional[str] = None
    is_active: bool = True


class DataCollectionRecord(BaseModel):
    """Record of data collection attempt"""
    train_no: str
    collection_type: str  # 'schedule', 'live_status', 'info'
    timestamp: datetime = Field(default_factory=datetime.now)
    success: bool
    error_message: Optional[str] = None
    response_time_ms: Optional[float] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# Export all schemas
__all__ = [
    'StationStop',
    'TrainSchedule',
    'LiveTrainStatus',
    'TrainInfo',
    'DataCollectionRecord'
]
