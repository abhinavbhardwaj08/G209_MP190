"""
Pydantic validation schemas for MP190 prediction service.

Enforces client business rules via Field constraints:
1. temp_c <= 70 (temperatures above 70 C rejected with 422)
2. wind_kmph >= 0 (wind speeds below 0 rejected with 422)
"""

from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Input schema for POST /predict.
    Must contain exactly these eight fields and no others.
    """
    pm25_today: float = Field(..., description="Average PM2.5 level today in ug/m3")
    wind_kmph: float = Field(..., ge=0.0, description="Average wind speed in km/h (must be >= 0)")
    humidity_pct: float = Field(..., description="Relative humidity percentage")
    temp_c: float = Field(..., le=70.0, description="Average temperature in Celsius (must be <= 70)")
    traffic_index: float = Field(..., description="Traffic volume count index")
    stubble_fires: float = Field(..., description="Satellite farm fire detections within 200 km")
    day_type: str = Field(..., description="Type of day (weekday, weekend, festival)")
    station_zone: str = Field(..., description="Station location zone (Central, Residential, Industrial)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pm25_today": 45.2,
                "wind_kmph": 12.5,
                "humidity_pct": 78.0,
                "temp_c": 29.5,
                "traffic_index": 120.0,
                "stubble_fires": 3.0,
                "day_type": "weekday",
                "station_zone": "Central",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response schema for POST /predict."""
    prediction: float = Field(..., description="Predicted next-day PM2.5 in ug/m3")
    status: str = Field(default="success", description="Prediction status indicator")
    model_version: Optional[str] = Field(default="1.0", description="Loaded model version")


class HealthResponse(BaseModel):
    """Response schema for GET /health."""
    status: str = Field(default="ok", description="Service health indicator")
    service: str = Field(default="mp190-api", description="Service identifier")
