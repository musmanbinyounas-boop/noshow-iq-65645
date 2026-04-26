"""Train the no-show model and persist artifacts to models/."""
import json
from pathlib import Path

from noshow_iq.preprocess import full_pipeline
from noshow_iq.model import train, save

DATA_PATH = "data/KaggleV2-May-2016.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "model.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"


def main():
    print(f"Loading and preprocessing {DATA_PATH} ...")
    X, y = full_pipeline(DATA_PATH)
    print(f"  X shape: {X.shape}, y distribution: {dict(y.value_counts())}")

    print("Training Logistic Regression with class_weight=balanced ...")
    pipe, metrics, (X_train, X_test, y_train, y_test) = train(X, y)

    print("\n=== Classification Report (test set) ===")
    for cls in ("0", "1"):
        m = metrics[cls]
        print(f"  Class {cls}: precision={m['precision']:.3f}  "
              f"recall={m['recall']:.3f}  f1={m['f1-score']:.3f}  "
              f"support={int(m['support'])}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")

    save(pipe, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    out = {
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "model": "LogisticRegression",
        "imbalance_technique": "class_weight=balanced",
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()