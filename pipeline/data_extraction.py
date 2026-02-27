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
OUTPUT_CSV_PATH = Path("data/processed/s2cool_features_ready.csv")
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
            dataframe = pd.read_sql_query(SELECT_RAW_SQL, con=connection)
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
    processor = S2CoolDataPreprocessor()
    return processor.process_pipeline(raw_df)


def save_processed_features(processed_df: pd.DataFrame, output_path: Path = OUTPUT_CSV_PATH) -> None:
    """Persist processed feature matrix to CSV.

    Args:
        processed_df: Fully processed feature matrix.
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

        save_processed_features(processed_df)
        logger.info("Saved processed features to %s", OUTPUT_CSV_PATH)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("data_extraction pipeline failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
