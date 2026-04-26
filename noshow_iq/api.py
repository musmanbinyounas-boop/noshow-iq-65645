"""FastAPI service for NoShowIQ — predicts patient no-show risk."""
import os
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

from . import model as M

load_dotenv(".env")

app = FastAPI(title="NoShowIQ", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Mongo (optional for local dev / tests) ---
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI) if MONGO_URI else None
db = client["noshowiq"] if client is not None else None
predictions = db["predictions"] if db is not None else None
training_runs = db["training_runs"] if db is not None else None

# --- Model load ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
PIPE = M.load(MODEL_PATH)

# Feature order must match training. We'll capture this from the model itself.
EXPECTED_FEATURES = [
    "Age", "Scholarship", "hypertension", "Diabetes", "Alcoholism",
    "handicap", "SMS_received", "days_in_advance", "is_same_day",
    "scheduled_dow", "is_weekend_appointment", "Gender_M", "nbh_freq",
]


class Appointment(BaseModel):
    """One appointment record as the clinic would send it."""
    Gender: str
    Age: int
    Neighbourhood: str
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int
    ScheduledDay: str
    AppointmentDay: str


def _featurize(raw: dict) -> dict:
    """Apply the same transforms as training to a single dict."""
    sched = pd.to_datetime(raw["ScheduledDay"], errors="coerce")
    appt = pd.to_datetime(raw["AppointmentDay"], errors="coerce")
    days_in_advance = max(0, (appt - sched).days)
    return {
        "Age": int(raw["Age"]),
        "Scholarship": int(raw["Scholarship"]),
        "hypertension": int(raw["Hipertension"]),
        "Diabetes": int(raw["Diabetes"]),
        "Alcoholism": int(raw["Alcoholism"]),
        "handicap": int(raw["Handcap"]),
        "SMS_received": int(raw["SMS_received"]),
        "days_in_advance": int(days_in_advance),
        "is_same_day": int(days_in_advance == 0),
        "scheduled_dow": int(sched.dayofweek),
        "is_weekend_appointment": int(appt.dayofweek in (5, 6)),
        "Gender_M": bool(raw["Gender"].upper() == "M"),
        # Default freq for unseen neighbourhoods. Fine for a demo.
        "nbh_freq": 0.05,
    }


def _risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


def _recommend(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH risk — call patient and send SMS reminder"
    if prob >= 0.4:
        return "MEDIUM risk — send SMS reminder"
    return "LOW risk — standard reminder"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(appt: Appointment):
    raw = appt.dict()
    features = _featurize(raw)
    X = pd.DataFrame([features])[EXPECTED_FEATURES]
    proba = float(PIPE.predict_proba(X)[0, 1])
    risk = _risk_level(proba)
    out = {
        "risk_level": risk,
        "probability": round(proba, 4),
        "recommendation": _recommend(proba),
    }
    if predictions is not None:
        predictions.insert_one({
            "timestamp": datetime.now(timezone.utc),
            "raw": raw,
            "features": features,
            "risk_level": out["risk_level"],
            "probability": out["probability"],
            "recommendation": out["recommendation"],
        })
    return out


@app.get("/history")
def history():
    if predictions is None:
        return []
    docs = list(predictions.find().sort("timestamp", -1).limit(20))
    for d in docs:
        d["_id"] = str(d["_id"])
        if "timestamp" in d and hasattr(d["timestamp"], "isoformat"):
            d["timestamp"] = d["timestamp"].isoformat()
    return docs


@app.get("/stats")
def stats():
    if predictions is None:
        raise HTTPException(503, "MongoDB not configured")
    pipeline = [
        {"$group": {
            "_id": None,
            "total_predictions": {"$sum": 1},
            "high_risk_count": {
                "$sum": {"$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]}
            },
            "medium_risk_count": {
                "$sum": {"$cond": [{"$eq": ["$risk_level", "MEDIUM"]}, 1, 0]}
            },
            "low_risk_count": {
                "$sum": {"$cond": [{"$eq": ["$risk_level", "LOW"]}, 1, 0]}
            },
            "average_probability": {"$avg": "$probability"},
        }},
        {"$project": {"_id": 0}},
    ]
    agg = list(predictions.aggregate(pipeline))
    out = agg[0] if agg else {
        "total_predictions": 0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0,
    }
    last = (
        training_runs.find_one(sort=[("timestamp", -1)])
        if training_runs is not None else None
    )
    if last and "timestamp" in last:
        ts = last["timestamp"]
        out["last_trained"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    else:
        out["last_trained"] = None
    return out
