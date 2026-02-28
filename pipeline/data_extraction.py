"""Secure raw-data extraction from Neon, preprocessing, and feature artifact export."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from preprocessing import S2CoolDataPreprocessor

logger = logging.getLogger(__name__)

RAW_TABLE_NAME = "solar_weather_data"
TRAINING_OUTPUT_CSV_PATH = Path("data/processed/s2cool_features_ready.csv")
RECENT_OUTPUT_CSV_PATH = Path("data/processed/s2cool_features_recent.csv")
STABLE_LAG_DAYS = int(os.getenv("STABLE_LAG_DAYS", "2"))
COUNT_RAW_SQL = text(
    """
    SELECT COUNT(*) AS total_rows
    FROM solar_weather_data;
    """
)
SELECT_RAW_SQL = text(
    """
    SELECT
        timestamp AS "Timestamp",
        city_name AS "City",
        shortwave_radiation AS "GHI",
        temperature_2m AS "Ambient_Temp"
    FROM solar_weather_data
    ORDER BY city_name ASC, timestamp ASC;
    """
)


def configure_logging() -> None:
    """Configure structured console logging for extraction workflows."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_database_url() -> str:
    """Load the Neon database URL from environment variables.

    Returns:
        Postgres SQLAlchemy connection URL.

    Raises:
        RuntimeError: If ``NEON_DB_URL`` is missing.
    """
    load_dotenv()
    database_url = os.getenv("NEON_DB_URL")
    if not database_url:
        raise RuntimeError("Missing required environment variable: NEON_DB_URL")
    return database_url


def create_db_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine with health-check enabled.

    Args:
        database_url: SQLAlchemy-compatible Postgres URL.

    Returns:
        Initialized SQLAlchemy engine.
    """
    return create_engine(database_url, pool_pre_ping=True)


def extract_raw_weather_data(engine: Engine) -> pd.DataFrame:
    """Extract raw weather records ordered by city then timestamp.

    Args:
        engine: SQLAlchemy engine connected to Neon.

    Returns:
        Raw weather dataframe with columns expected by the preprocessor.

    Raises:
        RuntimeError: If extraction fails or query returns no rows.
    """
    try:
        with engine.connect() as connection:
            total_rows_result = connection.execute(COUNT_RAW_SQL).scalar_one()
            total_rows = int(total_rows_result)

            if total_rows == 0:
                raise RuntimeError(f"Query returned no data from table '{RAW_TABLE_NAME}'.")

            logger.info("[extract   0%%] Starting row fetch from %s (%d rows)", RAW_TABLE_NAME, total_rows)

            chunks: list[pd.DataFrame] = []
            fetched_rows = 0
            for chunk in pd.read_sql_query(SELECT_RAW_SQL, con=connection, chunksize=25_000):
                chunks.append(chunk)
                fetched_rows += len(chunk)
                progress = min(100, int((fetched_rows / total_rows) * 100))
                logger.info(
                    "[extract %3d%%] Fetched %d/%d rows",
                    progress,
                    fetched_rows,
                    total_rows,
                )

            dataframe = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    except SQLAlchemyError as exc:
        raise RuntimeError("Failed to query Neon raw weather data.") from exc

    if dataframe.empty:
        raise RuntimeError(f"Query returned no data from table '{RAW_TABLE_NAME}'.")

    return dataframe


def run_preprocessing(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the S2Cool preprocessing pipeline on raw extracted data.

    Args:
        raw_df: Raw weather dataframe.

    Returns:
        Processed feature matrix dataframe.
    """
    processor = S2CoolDataPreprocessor(progress_enabled=False)
    return processor.process_pipeline(raw_df)


def split_stable_and_recent(
    processed_df: pd.DataFrame,
    stable_lag_days: int = STABLE_LAG_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split processed data into stable (training) and recent (serving) windows.

    Args:
        processed_df: Fully processed feature matrix indexed by timestamp.
        stable_lag_days: Number of trailing days to reserve as recent data.

    Returns:
        Tuple of (stable_df, recent_df, cutoff_timestamp_utc).
    """
    cutoff_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=stable_lag_days)
    stable_df = processed_df.loc[processed_df.index <= cutoff_utc].copy()
    recent_df = processed_df.loc[processed_df.index > cutoff_utc].copy()
    return stable_df, recent_df, cutoff_utc


def save_processed_features(processed_df: pd.DataFrame, output_path: Path) -> None:
    """Persist a processed dataframe to CSV.

    Args:
        processed_df: Processed feature matrix slice.
        output_path: Destination CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_save = processed_df.reset_index().rename(columns={"index": "Timestamp"})
    to_save.to_csv(output_path, index=False)


def main() -> None:
    """Execute end-to-end extraction, preprocessing, and artifact saving."""
    configure_logging()

    try:
        db_url = load_database_url()
        engine = create_db_engine(db_url)
        raw_df = extract_raw_weather_data(engine)
        logger.info("Raw extraction complete. Rows before processing: %d", len(raw_df))

        processed_df = run_preprocessing(raw_df)
        logger.info("Preprocessing complete. Rows after processing: %d", len(processed_df))

        stable_df, recent_df, cutoff_utc = split_stable_and_recent(processed_df)
        logger.info(
            "Time contract cutoff (UTC): %s | stable_rows=%d | recent_rows=%d",
            cutoff_utc.isoformat(),
            len(stable_df),
            len(recent_df),
        )

        if stable_df.empty:
            raise RuntimeError(
                "Stable training slice is empty after cutoff. Reduce STABLE_LAG_DAYS or backfill more data."
            )

        save_processed_features(stable_df, TRAINING_OUTPUT_CSV_PATH)
        logger.info("Saved stable training features to %s", TRAINING_OUTPUT_CSV_PATH)

        save_processed_features(recent_df, RECENT_OUTPUT_CSV_PATH)
        logger.info("Saved recent serving features to %s", RECENT_OUTPUT_CSV_PATH)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("data_extraction pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
