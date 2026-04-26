"""Log the most recent training run to MongoDB's training_runs collection."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(".env")
METRICS_PATH = Path("models/metrics.json")


def main():
    if not METRICS_PATH.exists():
        raise SystemExit(f"{METRICS_PATH} not found — run scripts/train.py first")

    metrics_data = json.loads(METRICS_PATH.read_text())

    uri = os.getenv("MONGO_URI")
    if not uri:
        raise SystemExit("MONGO_URI not set in environment")

    client = MongoClient(uri)
    runs = client["noshowiq"]["training_runs"]

    doc = {
        "timestamp": datetime.now(timezone.utc),
        "training_size": metrics_data["training_size"],
        "test_size": metrics_data["test_size"],
        "model": metrics_data["model"],
        "imbalance_technique": metrics_data["imbalance_technique"],
        "metrics": {
            "0": {
                "precision": metrics_data["metrics"]["0"]["precision"],
                "recall": metrics_data["metrics"]["0"]["recall"],
                "f1": metrics_data["metrics"]["0"]["f1-score"],
            },
            "1": {
                "precision": metrics_data["metrics"]["1"]["precision"],
                "recall": metrics_data["metrics"]["1"]["recall"],
                "f1": metrics_data["metrics"]["1"]["f1-score"],
            },
        },
        "accuracy": metrics_data["metrics"]["accuracy"],
    }
    result = runs.insert_one(doc)
    print(f"Logged training run: {result.inserted_id}")
    print(f"  Class 0 F1: {doc['metrics']['0']['f1']:.3f}")
    print(f"  Class 1 F1: {doc['metrics']['1']['f1']:.3f}")
    print(f"  Imbalance: {doc['imbalance_technique']}")


if __name__ == "__main__":
    main()
