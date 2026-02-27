"""Tests for ingest.preprocessing — S2CoolDataPreprocessor pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ingest.preprocessing import S2CoolDataPreprocessor


def _build_sample_df() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01 00:00", periods=90, freq="1h", tz="UTC")
    rows: list[dict[str, object]] = []

    for city_offset, city in enumerate(["Islamabad", "Lahore"]):
        for idx, ts in enumerate(timestamps):
            hour = ts.hour
            ghi = max(0.0, 780.0 * np.sin(np.pi * (hour - 6) / 12))
            temp = 10.0 + city_offset * 2.0 + (idx * 0.15)
            rows.append(
                {
                    "Timestamp": ts,
                    "City": city,
                    "GHI": float(ghi),
                    "Ambient_Temp": float(temp),
                }
            )

    df = pd.DataFrame(rows)

    # Short gap (2 hours) for linear interpolation.
    short_gap_mask = (
        (df["City"] == "Islamabad")
        & (df["Timestamp"] >= pd.Timestamp("2026-01-01 10:00", tz="UTC"))
        & (df["Timestamp"] <= pd.Timestamp("2026-01-01 11:00", tz="UTC"))
    )
    df.loc[short_gap_mask, "GHI"] = np.nan

    # Long gap (5 hours) for fallback imputation.
    long_gap_mask = (
        (df["City"] == "Lahore")
        & (df["Timestamp"] >= pd.Timestamp("2026-01-02 09:00", tz="UTC"))
        & (df["Timestamp"] <= pd.Timestamp("2026-01-02 13:00", tz="UTC"))
    )
    df.loc[long_gap_mask, "Ambient_Temp"] = np.nan

    # Duplicate timestamp for aggregation check.
    duplicate_row = {
        "Timestamp": pd.Timestamp("2026-01-02 00:00", tz="UTC"),
        "City": "Islamabad",
        "GHI": 1000.0,
        "Ambient_Temp": 40.0,
    }
    df = pd.concat([df, pd.DataFrame([duplicate_row])], ignore_index=True)

    return df


def test_process_pipeline_generates_required_outputs(tmp_path: Path) -> None:
    preprocessor = S2CoolDataPreprocessor(scaler_path=tmp_path / "s2cool_scaler.joblib")

    processed = preprocessor.process_pipeline(_build_sample_df())

    assert processed.index.tz is not None
    assert str(processed.index.tz) == "Asia/Karachi"

    required_columns = {
        "City",
        "GHI",
        "Ambient_Temp",
        "day_of_year",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "GHI_lag_1",
        "GHI_lag_2",
        "GHI_lag_3",
        "GHI_lag_24",
        "GHI_lag_48",
        "Ambient_Temp_lag_1",
        "Ambient_Temp_lag_2",
        "Ambient_Temp_lag_3",
        "Ambient_Temp_lag_24",
        "Ambient_Temp_lag_48",
        "GHI_rolling_mean_3h",
        "GHI_rolling_std_3h",
        "Temp_rolling_mean_6h",
        "Target_GHI_next_1h",
        "Target_Temp_next_1h",
    }
    assert required_columns.issubset(processed.columns)
    assert "hour" not in processed.columns
    assert "month" not in processed.columns

    assert preprocessor.scaler_path.exists()


def test_imputation_and_nighttime_zeroing() -> None:
    preprocessor = S2CoolDataPreprocessor(scaler_path="artifacts/test_scaler.joblib")
    processed = preprocessor.process_pipeline(_build_sample_df())

    # GHI must be exactly zero at night hours 20:00-05:00 in PKT.
    night_mask = (processed.index.hour >= 20) | (processed.index.hour <= 5)
    assert (processed.loc[night_mask, "GHI"] == 0.0).all()

    # The two injected gap regions should be filled by the imputation stack.
    assert processed["GHI"].isna().sum() == 0
    assert processed["Ambient_Temp"].isna().sum() == 0


def test_duplicate_timestamp_is_aggregated_in_step1_before_scaling(tmp_path: Path) -> None:
    df = _build_sample_df()
    preprocessor = S2CoolDataPreprocessor(scaler_path=tmp_path / "scaler.joblib")

    normalized = preprocessor._normalize_and_resample(df)
    target_ts = pd.Timestamp("2026-01-02 05:00", tz="Asia/Karachi")
    city_slice = normalized[(normalized["City"] == "Islamabad") & (normalized.index == target_ts)]

    # Original value and duplicate are averaged during resample.
    assert len(city_slice) == 1
    assert city_slice["GHI"].iloc[0] == 500.0
