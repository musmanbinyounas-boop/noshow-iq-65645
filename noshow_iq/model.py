"""Train, predict, evaluate, and persist the no-show classifier."""
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def build_pipeline() -> Pipeline:
    """Logistic Regression with balanced class weights to address the ~80/20 imbalance."""
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            n_jobs=-1,
            random_state=42,
        )),
    ])


def train(X, y):
    """Train on stratified 80/20 split. Returns (pipeline, metrics, splits)."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    metrics = evaluate(pipe, X_test, y_test)
    return pipe, metrics, (X_train, X_test, y_train, y_test)


def evaluate(pipe, X, y) -> dict:
    """Return per-class precision/recall/F1 (full classification_report dict)."""
    return classification_report(y, pipe.predict(X), output_dict=True)


def predict(pipe, row_dict: dict):
    """Predict on a single appointment dict. Returns (label, probability)."""
    X = pd.DataFrame([row_dict])
    proba = float(pipe.predict_proba(X)[0, 1])
    label = int(proba >= 0.5)
    return label, proba


def save(pipe, path: str) -> None:
    joblib.dump(pipe, path)


def load(path: str):
    return joblib.load(path)