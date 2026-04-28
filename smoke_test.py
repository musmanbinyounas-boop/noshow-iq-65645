"""Smoke test for the live NoShowIQ deployment.

Hits /health, /predict, /stats and prints PASS or FAIL for each.

Usage:
    python smoke_test.py https://musmanbinyounas-noshow-iq-65645.hf.space
"""
import json
import sys

import requests

if len(sys.argv) != 2:
    print("Usage: python smoke_test.py <BASE_URL>")
    sys.exit(1)

URL = sys.argv[1].rstrip("/")

SAMPLE = {
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

results = []


def check(name: str, ok: bool, body=None) -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}")
    results.append(ok)
    if not ok and body is not None:
        snippet = json.dumps(body, indent=2)[:300] if isinstance(body, dict) else str(body)[:300]
        print(f"       {snippet}")


# 1. /health
try:
    r = requests.get(f"{URL}/health", timeout=15)
    check("/health returns 200 and status ok",
          r.status_code == 200 and r.json().get("status") == "ok",
          r.json() if r.ok else r.text)
except Exception as e:
    check("/health returns 200 and status ok", False, {"error": str(e)})

# 2. /predict
try:
    r = requests.post(f"{URL}/predict", json=SAMPLE, timeout=20)
    body = r.json() if r.ok else r.text
    keys_ok = (
        r.status_code == 200
        and isinstance(body, dict)
        and {"risk_level", "probability", "recommendation"}.issubset(body.keys())
        and 0.0 <= body.get("probability", -1) <= 1.0
    )
    check("/predict returns risk_level, probability, recommendation", keys_ok, body)
except Exception as e:
    check("/predict returns risk_level, probability, recommendation", False, {"error": str(e)})

# 3. /stats
try:
    r = requests.get(f"{URL}/stats", timeout=15)
    body = r.json() if r.ok else r.text
    ok = r.status_code == 200 and isinstance(body, dict) and "total_predictions" in body
    check("/stats returns aggregation with total_predictions", ok, body)
except Exception as e:
    check("/stats returns aggregation with total_predictions", False, {"error": str(e)})

print()
print(f"Result: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
