"""Train S2Cool GHI models from preprocessed features (no database access)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

logger = logging.getLogger(__name__)

PROCESSED_CSV_PATH = Path("data/s2cool_features_ready.csv")
ARTIFACTS_DIR = Path("artifacts")
METRICS_PATH = ARTIFACTS_DIR / "metrics_latest.json"
ALLOW_RECENT_TRAINING = os.getenv("ALLOW_RECENT_TRAINING", "false").lower() == "true"


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""

    model_name: str
    mae: float
    rmse: float
    model_path: str


@dataclass
class TrainOutput:
    """Aggregated output metadata for one training run."""

    trained_at_utc: str
    train_rows: int
    test_rows: int
    xgboost: ModelMetrics
    lstm: ModelMetrics


def configure_logging() -> None:
    """Configure console logging for the training workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_processed_dataframe(csv_path: Path = PROCESSED_CSV_PATH) -> pd.DataFrame:
    """Load processed feature matrix from CSV.

    Args:
        csv_path: Path to processed features CSV.

    Returns:
        Parsed and chronologically sorted dataframe.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file is empty or timestamp parsing fails.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    if "recent" in csv_path.name.lower() and not ALLOW_RECENT_TRAINING:
        raise ValueError(
            "Refusing to train from recent-serving slice. "
            "Use stable file 'data/s2cool_features_ready.csv' or set "
            "ALLOW_RECENT_TRAINING=true to override."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Processed CSV is empty: {csv_path}")

    if "Timestamp" not in df.columns:
        raise KeyError("Processed CSV must include a 'Timestamp' column.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce", utc=True)
    if df["Timestamp"].isna().any():
        raise ValueError("Timestamp parsing failed in processed CSV.")

    df = df.sort_values("Timestamp").reset_index(drop=True)
    logger.info("Loaded processed dataframe shape: %s", df.shape)
    return df


def build_train_test_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create strict chronological train/test matrices for GHI prediction.

    Args:
        df: Processed dataframe.

    Returns:
        X_train, X_test, y_train_ghi, y_test_ghi.
    """
    target_columns = ["Target_GHI_next_1h", "Target_Temp_next_1h"]
    missing_targets = [column for column in target_columns if column not in df.columns]
    if missing_targets:
        raise KeyError(f"Missing target column(s): {missing_targets}")

    x_df = df.drop(columns=target_columns).copy()
    y_df = df[target_columns].copy()

    if "Timestamp" in x_df.columns:
        x_df["Timestamp"] = pd.to_datetime(x_df["Timestamp"], utc=True).astype("int64") // 10**9
    if "City" in x_df.columns:
        x_df["City"] = x_df["City"].astype("category")
        x_df = pd.get_dummies(x_df, columns=["City"], dtype=float)

    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    split_index = int(len(x_df) * 0.8)
    x_train = x_df.iloc[:split_index].copy()
    x_test = x_df.iloc[split_index:].copy()
    y_train_ghi = y_df.iloc[:split_index]["Target_GHI_next_1h"].copy()
    y_test_ghi = y_df.iloc[split_index:]["Target_GHI_next_1h"].copy()

    logger.info("X_train shape=%s | X_test shape=%s", x_train.shape, x_test.shape)
    logger.info("y_train_ghi shape=%s | y_test_ghi shape=%s", y_train_ghi.shape, y_test_ghi.shape)
    return x_train, x_test, y_train_ghi, y_test_ghi


def train_xgboost(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_ghi: pd.Series,
    y_test_ghi: pd.Series,
    artifacts_dir: Path,
) -> ModelMetrics:
    """Train and evaluate XGBoost baseline for GHI forecast."""
    model = xgb.XGBRegressor(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=200,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train_ghi.to_numpy())

    preds = model.predict(x_test)
    mae = float(mean_absolute_error(y_test_ghi, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test_ghi, preds)))

    model_path = artifacts_dir / "xgboost_ghi_model.pkl"
    joblib.dump(model, model_path)
    logger.info("XGBoost MAE=%.4f RMSE=%.4f saved=%s", mae, rmse, model_path)

    return ModelMetrics(model_name="xgboost", mae=mae, rmse=rmse, model_path=str(model_path))


def train_lstm(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_ghi: pd.Series,
    y_test_ghi: pd.Series,
    artifacts_dir: Path,
) -> ModelMetrics:
    """Train and evaluate lightweight LSTM for GHI forecast."""
    x_train_lstm = x_train.to_numpy(dtype=np.float32).reshape((x_train.shape[0], 1, x_train.shape[1]))
    x_test_lstm = x_test.to_numpy(dtype=np.float32).reshape((x_test.shape[0], 1, x_test.shape[1]))
    y_train_lstm = y_train_ghi.to_numpy(dtype=np.float32)
    y_test_lstm = y_test_ghi.to_numpy(dtype=np.float32)

    logger.info(
        "LSTM arrays: X_train=%s X_test=%s y_train=%s y_test=%s",
        x_train_lstm.shape,
        x_test_lstm.shape,
        y_train_lstm.shape,
        y_test_lstm.shape,
    )

    model = Sequential(
        [
            LSTM(32, return_sequences=False, input_shape=(1, x_train.shape[1])),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        x_train_lstm,
        y_train_lstm,
        validation_data=(x_test_lstm, y_test_lstm),
        epochs=10,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    preds = model.predict(x_test_lstm, verbose=0).reshape(-1)
    mae = float(mean_absolute_error(y_test_lstm, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test_lstm, preds)))

    model_path = artifacts_dir / "lstm_ghi_model.keras"
    model.save(model_path)
    logger.info("LSTM MAE=%.4f RMSE=%.4f saved=%s", mae, rmse, model_path)

    return ModelMetrics(model_name="lstm", mae=mae, rmse=rmse, model_path=str(model_path))


def save_metrics(output: TrainOutput, metrics_path: Path = METRICS_PATH) -> None:
    """Persist latest training metrics as JSON."""
    payload = {
        "trained_at_utc": output.trained_at_utc,
        "train_rows": output.train_rows,
        "test_rows": output.test_rows,
        "xgboost": asdict(output.xgboost),
        "lstm": asdict(output.lstm),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved metrics to %s", metrics_path)


def main() -> None:
    """Run end-to-end training for XGBoost and LSTM from processed CSV."""
    configure_logging()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = load_processed_dataframe()
    x_train, x_test, y_train_ghi, y_test_ghi = build_train_test_matrices(dataframe)

    xgboost_metrics = train_xgboost(x_train, x_test, y_train_ghi, y_test_ghi, ARTIFACTS_DIR)
    lstm_metrics = train_lstm(x_train, x_test, y_train_ghi, y_test_ghi, ARTIFACTS_DIR)

    output = TrainOutput(
        trained_at_utc=datetime.now(UTC).isoformat(),
        train_rows=len(x_train),
        test_rows=len(x_test),
        xgboost=xgboost_metrics,
        lstm=lstm_metrics,
    )
    save_metrics(output)


if __name__ == "__main__":
    main()
