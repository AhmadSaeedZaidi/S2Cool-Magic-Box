"""Data cleaning and feature engineering pipeline for S2Cool."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class S2CoolDataPreprocessor:
    """Build a production-grade feature matrix from raw hourly weather data.

    The pipeline enforces temporal consistency, imputes missing values with a
    tiered strategy, generates cyclical and autoregressive features, builds next-hour
    targets, and persists a fitted scaler for serving-time reuse.
    """

    REQUIRED_COLUMNS: tuple[str, str, str, str] = ("Timestamp", "City", "GHI", "Ambient_Temp")
    BASE_FEATURES: tuple[str, str] = ("GHI", "Ambient_Temp")
    LAG_HOURS: tuple[int, int, int, int, int] = (1, 2, 3, 24, 48)

    def __init__(self, scaler_path: str | Path = "artifacts/s2cool_standard_scaler.joblib") -> None:
        """Initialize preprocessor configuration.

        Args:
            scaler_path: Filesystem location where the fitted ``StandardScaler``
                is serialized via ``joblib``.
        """
        self.scaler_path = Path(scaler_path)
        self.scaler: StandardScaler | None = None
        self.scaled_columns: list[str] = []

    def process_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full 7-step preprocessing pipeline in strict order.

        Args:
            df: Raw hourly observations with columns ``Timestamp``, ``City``,
                ``GHI`` and ``Ambient_Temp``.

        Returns:
            A model-ready feature matrix with engineered predictors and
            next-hour targets.

        Raises:
            ValueError: If required columns are missing or timestamps cannot be parsed.
        """
        self._validate_input(df)
        logger.info("Preprocessing start shape: %s", df.shape)

        processed = self._normalize_and_resample(df)
        processed, imputed_count = self._apply_advanced_imputation(processed)
        processed = self._apply_cyclical_encoding(processed)
        processed = self._generate_lag_features(processed)
        processed = self._generate_rolling_features(processed)
        processed = self._generate_targets(processed)
        processed = self._scale_continuous_features(processed)

        logger.info("NaNs imputed during pipeline: %d", imputed_count)
        logger.info("Preprocessing end shape: %s", processed.shape)
        return processed

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate that mandatory columns required by the pipeline are present.

        Args:
            df: Input data frame to validate.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _normalize_and_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamps to PKT and aggregate to a strict hourly grid.

        Math notes:
            Duplicate rows within a ``(City, Timestamp)`` bin are reduced with the
            arithmetic mean: ``x̄ = (1/n) * Σ xᵢ``.

        Args:
            df: Raw input dataframe.

        Returns:
            Hourly, city-partitioned dataframe indexed by PKT timestamp.

        Raises:
            ValueError: If timestamp parsing fails for any row.
        """
        working = df.loc[:, list(self.REQUIRED_COLUMNS)].copy()
        timestamps = pd.to_datetime(working["Timestamp"], errors="coerce", utc=True)

        if timestamps.isna().any():
            invalid_count = int(timestamps.isna().sum())
            raise ValueError(f"Unable to parse {invalid_count} timestamp value(s).")

        working["Timestamp"] = timestamps.dt.tz_convert("Asia/Karachi")
        working = working.set_index("Timestamp").sort_index()

        resampled = (
            working.groupby("City")
            .resample("1h")
            .mean(numeric_only=True)
            .reset_index()
            .sort_values(["City", "Timestamp"])
            .set_index("Timestamp")
        )
        return resampled

    def _apply_advanced_imputation(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Apply short-gap interpolation, long-gap history fills, and night masking.

        Imputation logic:
            1) Short gaps (≤3 hours): linear interpolation.
            2) Long gaps (>3 hours): mean of same hour from prior 7 days.
            3) Fallback: monthly mean at matching hour.
            4) Physical constraint: force ``GHI=0`` for hours 20:00-05:00.

        Args:
            df: Hourly dataframe indexed by timestamp.

        Returns:
            Tuple of imputed dataframe and number of NaN values filled.
        """
        before_missing = int(df.loc[:, list(self.BASE_FEATURES)].isna().sum().sum())
        city_frames: list[pd.DataFrame] = []

        for _, city_frame in df.groupby("City", sort=False):
            city_copy = city_frame.copy()
            for feature in self.BASE_FEATURES:
                city_copy[feature] = self._impute_feature_series(city_copy[feature])

            night_mask = (city_copy.index.hour >= 20) | (city_copy.index.hour <= 5)
            city_copy.loc[night_mask, "GHI"] = 0.0
            city_frames.append(city_copy)

        imputed = pd.concat(city_frames)
        imputed = (
            imputed.reset_index().sort_values(["City", "Timestamp"]).set_index("Timestamp")
        )

        after_missing = int(imputed.loc[:, list(self.BASE_FEATURES)].isna().sum().sum())
        return imputed, before_missing - after_missing

    def _impute_feature_series(self, series: pd.Series) -> pd.Series:
        """Impute one city-level hourly feature using a three-tier hierarchy.

        Math notes:
            Linear interpolation estimates a missing point ``x_t`` via line segment
            interpolation between nearest observed neighbors. Long-gap fallback uses:
            ``x_t = mean(x_{t-24}, x_{t-48}, ..., x_{t-168})`` ignoring missing terms.

        Args:
            series: Hourly feature values for a single city.

        Returns:
            Imputed feature series aligned to the original index.
        """
        missing = series.isna()
        run_id = (missing != missing.shift(fill_value=False)).cumsum()
        run_length = missing.groupby(run_id).transform("sum")
        long_gap_mask = missing & (run_length > 3)

        interpolated = series.interpolate(method="linear", limit=3, limit_area="inside")
        interpolated.loc[long_gap_mask] = np.nan

        long_gap_history = pd.concat(
            [interpolated.shift(24 * offset) for offset in range(1, 8)], axis=1
        )
        same_hour_seven_day_mean = long_gap_history.mean(axis=1, skipna=True)
        interpolated = interpolated.fillna(same_hour_seven_day_mean)

        monthly_hour_mean = interpolated.groupby(
            [interpolated.index.month, interpolated.index.hour]
        ).transform("mean")
        interpolated = interpolated.fillna(monthly_hour_mean)

        return interpolated

    def _apply_cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode periodic time variables using sine/cosine projections.

        For a period ``P`` and value ``v``, cyclical projection is:
        ``sin(2πv/P)``, ``cos(2πv/P)``.

        Args:
            df: Imputed hourly dataframe.

        Returns:
            Dataframe with cyclical features and without raw hour/month columns.
        """
        encoded = df.copy()
        encoded["hour"] = encoded.index.hour
        encoded["day_of_year"] = encoded.index.dayofyear
        encoded["month"] = encoded.index.month

        encoded["hour_sin"] = np.sin(2 * np.pi * encoded["hour"] / 24)
        encoded["hour_cos"] = np.cos(2 * np.pi * encoded["hour"] / 24)
        encoded["month_sin"] = np.sin(2 * np.pi * encoded["month"] / 12)
        encoded["month_cos"] = np.cos(2 * np.pi * encoded["month"] / 12)

        encoded = encoded.drop(columns=["hour", "month"])
        return encoded

    def _generate_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate autoregressive lag features grouped by city.

        Args:
            df: Time-indexed dataframe with base weather features.

        Returns:
            Dataframe with lag columns for ``GHI`` and ``Ambient_Temp``.
        """
        lagged = df.copy()
        grouped = lagged.groupby("City")

        for feature in self.BASE_FEATURES:
            for lag_hour in self.LAG_HOURS:
                lagged[f"{feature}_lag_{lag_hour}"] = grouped[feature].shift(lag_hour)

        return lagged

    def _generate_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create leakage-safe rolling statistics that only use past observations.

        Shift-by-one ensures the window excludes the current timestamp, avoiding
        lookahead leakage in supervised forecasting.

        Args:
            df: Lag-enriched dataframe.

        Returns:
            Dataframe with rolling means and volatility statistics.
        """
        rolling = df.copy()
        grouped = rolling.groupby("City", group_keys=False)

        rolling["GHI_rolling_mean_3h"] = grouped["GHI"].transform(
            lambda series: series.shift(1).rolling(window=3, min_periods=1).mean()
        )
        rolling["GHI_rolling_std_3h"] = grouped["GHI"].transform(
            lambda series: series.shift(1).rolling(window=3, min_periods=2).std()
        )
        rolling["Temp_rolling_mean_6h"] = grouped["Ambient_Temp"].transform(
            lambda series: series.shift(1).rolling(window=6, min_periods=1).mean()
        )

        return rolling

    def _generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create one-hour-ahead forecasting targets and trim terminal NaNs.

        Args:
            df: Feature dataframe before target generation.

        Returns:
            Dataframe with ``Target_GHI_next_1h`` and
            ``Target_Temp_next_1h`` columns.
        """
        target_df = df.copy()
        grouped = target_df.groupby("City")

        target_df["Target_GHI_next_1h"] = grouped["GHI"].shift(-1)
        target_df["Target_Temp_next_1h"] = grouped["Ambient_Temp"].shift(-1)

        target_df = target_df.dropna(subset=["Target_GHI_next_1h", "Target_Temp_next_1h"])
        return target_df

    def _scale_continuous_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize continuous predictors and persist fitted scaler.

        Standardization transforms each feature with:
        ``z = (x - μ) / σ``.

        Scaled columns include raw ``Ambient_Temp``, all lag columns, and rolling
        statistics. Cyclical encodings and target columns are left untouched.

        Args:
            df: Post-target dataframe.

        Returns:
            Dataframe with standardized continuous predictor columns.
        """
        scaled_df = df.copy()

        continuous_columns = [
            "Ambient_Temp",
            *[col for col in scaled_df.columns if col.startswith("GHI_lag_")],
            *[col for col in scaled_df.columns if col.startswith("Ambient_Temp_lag_")],
            "GHI_rolling_mean_3h",
            "GHI_rolling_std_3h",
            "Temp_rolling_mean_6h",
        ]
        self.scaled_columns = [col for col in continuous_columns if col in scaled_df.columns]

        self.scaler = StandardScaler()
        scaled_df.loc[:, self.scaled_columns] = self.scaler.fit_transform(
            scaled_df.loc[:, self.scaled_columns]
        )

        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)

        return scaled_df
