"""
API endpoint tests using FastAPI TestClient.

Tests:
- GET /health returns 200 and {"status": "ok"}
- POST /predict with valid schema returns 200
- Client Rule 1: temp_c > 70 is rejected with HTTP 422
- Client Rule 2: wind_kmph < 0 is rejected with HTTP 422
- Missing required fields is rejected with HTTP 422
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client instance."""
    return TestClient(app)


@pytest.fixture
def valid_payload():
    """Return a valid prediction request payload."""
    return {
        "pm25_today": 45.2,
        "wind_kmph": 12.5,
        "humidity_pct": 78.0,
        "temp_c": 29.5,
        "traffic_index": 120.0,
        "stubble_fires": 3.0,
        "day_type": "weekday",
        "station_zone": "Central",
    }


def test_health_endpoint(client):
    """GET /health must return status 200 and {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_valid_request(client, valid_payload):
    """POST /predict with valid input returns 200 and numerical prediction."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))
    assert data["status"] == "success"


def test_predict_rejects_temperature_above_70(client, valid_payload):
    """Rule 1: temp_c above 70 must be rejected with 422 Unprocessable Entity."""
    bad_payload = valid_payload.copy()
    bad_payload["temp_c"] = 71.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_negative_wind_speed(client, valid_payload):
    """Rule 2: wind_kmph below 0 must be rejected with 422 Unprocessable Entity."""
    bad_payload = valid_payload.copy()
    bad_payload["wind_kmph"] = -1.0
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, valid_payload):
    """Missing any required field must be rejected with 422."""
    bad_payload = valid_payload.copy()
    del bad_payload["pm25_today"]
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
