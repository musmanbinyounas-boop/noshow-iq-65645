"""Cleaning and feature engineering for the medical no-show dataset."""
import pandas as pd

ID_COLS = ["PatientId", "AppointmentID"]
DATE_COLS = ["ScheduledDay", "AppointmentDay"]
TARGET_RAW = "No-show"


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        TARGET_RAW: "no_show",
        "Hipertension": "hypertension",
        "Handcap": "handicap",
    })
    df.columns = [c.strip() for c in df.columns]
    return df


def filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Age"] >= 0].copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["days_in_advance"] = (df["AppointmentDay"] - df["ScheduledDay"]).dt.days.clip(lower=0)
    df["is_same_day"] = (df["days_in_advance"] == 0).astype(int)
    df["scheduled_dow"] = df["ScheduledDay"].dt.dayofweek
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)
    freq = df["Neighbourhood"].value_counts(normalize=True)
    df["nbh_freq"] = df["Neighbourhood"].map(freq)
    df = df.drop(columns=ID_COLS + DATE_COLS + ["Neighbourhood"])
    return df


def split_xy(df: pd.DataFrame):
    df["no_show"] = (df["no_show"].str.lower() == "yes").astype(int)
    y = df["no_show"]
    X = df.drop(columns=["no_show"])
    return X, y


def full_pipeline(path: str):
    df = load_csv(path)
    df = clean_columns(df)
    df = filter_invalid(df)
    df = engineer_features(df)
    df = encode(df)
    return split_xy(df)
