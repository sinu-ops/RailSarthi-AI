"""
FastAPI Routers for RailSarthi AI Backend.
"""

from fastapi import APIRouter

api_router = APIRouter(prefix="/api/v1", tags=["RailSarthi API"])

# Endpoints will be added here as we develop the backend module-by-module
# For now, the primary app assembly lives in backend/main.py with core endpoints.

__all__ = ["api_router"]
