"""Automated EDA visualizations and strict chronological time-series data splits."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FEATURES_CSV_PATH = Path("data/processed/s2cool_features_ready.csv")
VISUALIZATION_DIR = Path("visualizations")
SPLITS_DIR = Path("data/splits")
TARGET_COLUMNS: tuple[str, str] = ("Target_GHI_next_1h", "Target_Temp_next_1h")


def configure_logging() -> None:
    """Configure consistent logging for EDA and split generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_feature_matrix(csv_path: Path = FEATURES_CSV_PATH) -> pd.DataFrame:
    """Load and validate the processed feature matrix.

    Args:
        csv_path: Path to preprocessed features CSV.

    Returns:
        Loaded dataframe sorted chronologically by timestamp.

    Raises:
        FileNotFoundError: If processed CSV does not exist.
        ValueError: If the file is empty or timestamp parsing fails.
        KeyError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed feature file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty:
        raise ValueError("Processed feature CSV is empty.")

    if "Timestamp" not in dataframe.columns:
        raise KeyError("Processed feature CSV must include a 'Timestamp' column.")

    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"], errors="coerce", utc=True)
    if dataframe["Timestamp"].isna().any():
        raise ValueError("Failed to parse one or more Timestamp values from processed CSV.")

    dataframe = dataframe.sort_values("Timestamp").reset_index(drop=True)
    return dataframe


def create_output_directories() -> None:
    """Create visualization and split directories if absent."""
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create and save target-focused correlation heatmap.

    Args:
        df: Processed feature dataframe.

    Raises:
        KeyError: If any required heatmap column is missing.
    """
    columns = [
        "Target_GHI_next_1h",
        "Ambient_Temp",
        "GHI_lag_1",
        "GHI_lag_24",
        "GHI_rolling_mean_3h",
    ]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for correlation heatmap: {missing}")

    corr = df.loc[:, columns].corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix (Target_GHI_next_1h Focus)")
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "corr_heatmap.png", dpi=300)
    plt.close()


def plot_thermal_inertia_slice(df: pd.DataFrame, city: str = "Lahore") -> None:
    """Generate dual-axis 7-day thermal inertia chart for a city.

    The slice is fixed to the first week of July 2025 to expose expected lag
    between solar irradiance and ambient temperature.

    Args:
        df: Processed feature dataframe.
        city: City name to chart.

    Raises:
        ValueError: If no records exist for the requested city and period.
        KeyError: If required plotting columns are missing.
    """
    required = ["City", "Timestamp", "GHI", "Ambient_Temp"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for thermal inertia plot: {missing}")

    start = pd.Timestamp("2025-07-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-07-07 23:00:00", tz="UTC")

    city_slice = df[(df["City"] == city) & (df["Timestamp"] >= start) & (df["Timestamp"] <= end)].copy()
    if city_slice.empty:
        raise ValueError(
            f"No rows available for city='{city}' in window {start.isoformat()} to {end.isoformat()}."
        )

    fig, ax_left = plt.subplots(figsize=(12, 6))
    ax_right = ax_left.twinx()

    ax_left.plot(city_slice["Timestamp"], city_slice["GHI"], color="goldenrod", label="GHI")
    ax_right.plot(
        city_slice["Timestamp"],
        city_slice["Ambient_Temp"],
        color="crimson",
        label="Ambient_Temp",
    )

    ax_left.set_xlabel("Timestamp")
    ax_left.set_ylabel("GHI (W/m²)", color="goldenrod")
    ax_right.set_ylabel("Ambient Temp (°C)", color="crimson")
    plt.title(f"7-Day Thermal Inertia Slice — {city} (July 2025)")

    left_lines, left_labels = ax_left.get_legend_handles_labels()
    right_lines, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_lines + right_lines, left_labels + right_labels, loc="upper right")

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "7_day_thermal_slice.png", dpi=300)
    plt.close(fig)


def plot_cyclical_time_proof(df: pd.DataFrame) -> None:
    """Generate scatter plot verifying cyclical hour encoding lies on a circle.

    Args:
        df: Processed feature dataframe.

    Raises:
        KeyError: If cyclical columns are missing.
    """
    required = ["hour_sin", "hour_cos"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for cyclical proof: {missing}")

    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=df["hour_sin"], y=df["hour_cos"], s=30, alpha=0.65)
    plt.title("Cyclical Time Encoding Integrity (hour_sin vs hour_cos)")
    plt.xlabel("hour_sin")
    plt.ylabel("hour_cos")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / "cyclical_time_proof.png", dpi=300)
    plt.close()


def split_time_series(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into 80% train and 20% test sets.

    Args:
        df: Processed feature dataframe sorted by ``Timestamp`` ascending.

    Returns:
        Tuple of ``X_train``, ``X_test``, ``y_train``, ``y_test`` dataframes.

    Raises:
        KeyError: If target columns are missing.
        ValueError: If not enough rows for a meaningful split.
    """
    missing_targets = [column for column in TARGET_COLUMNS if column not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target column(s): {missing_targets}")

    if len(df) < 10:
        raise ValueError("Insufficient rows for robust 80/20 time-series split.")

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    y_train = train_df.loc[:, list(TARGET_COLUMNS)].copy()
    y_test = test_df.loc[:, list(TARGET_COLUMNS)].copy()

    x_train = train_df.drop(columns=list(TARGET_COLUMNS))
    x_test = test_df.drop(columns=list(TARGET_COLUMNS))

    return x_train, x_test, y_train, y_test


def save_splits(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> None:
    """Persist train/test feature and target datasets as CSV files."""
    x_train.to_csv(SPLITS_DIR / "X_train.csv", index=False)
    x_test.to_csv(SPLITS_DIR / "X_test.csv", index=False)
    y_train.to_csv(SPLITS_DIR / "y_train.csv", index=False)
    y_test.to_csv(SPLITS_DIR / "y_test.csv", index=False)


def main() -> None:
    """Run end-to-end EDA plotting and chronological split export."""
    configure_logging()
    create_output_directories()

    try:
        dataframe = load_feature_matrix()
        logger.info("Loaded processed matrix with %d rows.", len(dataframe))

        plot_correlation_heatmap(dataframe)
        plot_thermal_inertia_slice(dataframe, city="Lahore")
        plot_cyclical_time_proof(dataframe)
        logger.info("Saved EDA visualizations to %s", VISUALIZATION_DIR)

        x_train, x_test, y_train, y_test = split_time_series(dataframe)
        save_splits(x_train, x_test, y_train, y_test)
        logger.info(
            "Saved splits to %s | X_train=%d, X_test=%d, y_train=%d, y_test=%d",
            SPLITS_DIR,
            len(x_train),
            len(x_test),
            len(y_train),
            len(y_test),
        )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        logger.exception("eda_and_splitting pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
