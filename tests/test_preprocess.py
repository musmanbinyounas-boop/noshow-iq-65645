import pandas as pd
from noshow_iq.preprocess import (
    clean_columns, filter_invalid, engineer_features
)


def _sample():
    return pd.DataFrame({
        "PatientId": [1, 2, 3],
        "AppointmentID": [10, 20, 30],
        "Gender": ["F", "M", "F"],
        "ScheduledDay": ["2016-04-29T18:38:08Z", "2016-04-29T16:08:27Z", "2016-04-29T16:19:04Z"],
        "AppointmentDay": ["2016-04-29T00:00:00Z", "2016-04-29T00:00:00Z", "2016-05-03T00:00:00Z"],
        "Age": [62, -1, 8],
        "Neighbourhood": ["A", "B", "A"],
        "Scholarship": [0, 0, 0], "Hipertension": [1, 0, 0], "Diabetes": [0, 0, 0],
        "Alcoholism": [0, 0, 0], "Handcap": [0, 0, 0], "SMS_received": [0, 0, 0],
        "No-show": ["No", "Yes", "No"],
    })


def test_clean_columns_renames_target():
    df = clean_columns(_sample())
    assert "no_show" in df.columns
    assert "No-show" not in df.columns


def test_filter_invalid_drops_negative_age():
    df = filter_invalid(_sample())
    assert (df["Age"] >= 0).all()
    assert len(df) == 2


def test_engineer_days_in_advance_non_negative():
    df = engineer_features(clean_columns(_sample()))
    assert (df["days_in_advance"] >= 0).all()
    assert "is_same_day" in df.columns