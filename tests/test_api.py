"""Endpoint-level tests using FastAPI's TestClient."""
import os

# Block Mongo from connecting during tests — keeps tests fast and offline.
os.environ.pop("MONGO_URI", None)

from fastapi.testclient import TestClient  # noqa: E402
from noshow_iq.api import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "Gender": "F",
    "Age": 35,
    "Neighbourhood": "JARDIM CAMBURI",
    "Scholarship": 0,
    "Hipertension": 0,
    "Diabetes": 0,
    "Alcoholism": 0,
    "Handcap": 0,
    "SMS_received": 1,
    "ScheduledDay": "2016-04-29T18:38:08Z",
    "AppointmentDay": "2016-05-03T00:00:00Z",
}


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_required_keys():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) >= {"risk_level", "probability", "recommendation"}


def test_predict_probability_within_range():
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
