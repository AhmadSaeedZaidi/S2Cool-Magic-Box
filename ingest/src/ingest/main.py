"""Pipeline CLI entrypoint (current / historic modes)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from .api import (
    HISTORIC_DEFAULT_LOOKBACK_DAYS,
    PAKISTAN_CITIES,
    fetch_all_cities_historic,
    fetch_city_weather,
)
from .db import init_db, upsert_records

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    """Return env-var value or ``sys.exit(1)`` with a clear message."""
    value = os.environ.get(name)
    if not value:
        logger.error("Required environment variable '%s' is not set.", name)
        sys.exit(1)
    return value


def _parse_date_arg(value: str, label: str) -> date:
    """Parse *YYYY-MM-DD* into a ``date``; exits on bad format."""
    try:
        return date.fromisoformat(value)
    except ValueError:
        logger.error("Invalid date for %s: '%s'. Expected YYYY-MM-DD.", label, value)
        sys.exit(1)


def _city_loop(database_url: str, fetch_fn, mode_label: str) -> tuple[int, int]:
    """Fetch + upsert for every city. Returns ``(successes, failures)``."""
    successes = failures = 0
    for city in PAKISTAN_CITIES:
        logger.info("[%s] Processing %s", mode_label, city.name)
        try:
            records = fetch_fn(city)
        except Exception:  # noqa: BLE001
            logger.exception("[%s] Fetch failed, skipping", city.name)
            failures += 1
            continue
        if not records:
            logger.warning("[%s] 0 records returned, skipping upsert", city.name)
            failures += 1
            continue
        try:
            upsert_records(database_url, records, city_name=city.name)
            successes += 1
        except Exception:  # noqa: BLE001
            logger.exception("[%s] Upsert failed", city.name)
            failures += 1
    return successes, failures


def _finish(successes: int, failures: int, total: int) -> None:
    """Log summary and set exit code."""
    logger.info("Pipeline finished: %d/%d cities succeeded", successes, total)
    if successes == 0:
        sys.exit(2)
    if failures > 0:
        logger.warning("%d city/cities had errors (partial success).", failures)


# --- Mode handlers ----------------------------------------------------------


def run_current(database_url: str) -> None:
    """Run the current-data pipeline (forecast endpoint)."""
    logger.info("Mode: CURRENT")
    s, f = _city_loop(database_url, fetch_city_weather, "current")
    _finish(s, f, len(PAKISTAN_CITIES))


def run_historic(database_url: str, start_date: date, end_date: date) -> None:
    """Run the historic backfill pipeline (archive endpoint).

    Args:
        database_url: Postgres DSN.
        start_date: First date (inclusive).
        end_date: Last date (inclusive).
    """
    logger.info("Mode: HISTORIC  %s → %s", start_date, end_date)
    if end_date < start_date:
        logger.error("--end-date (%s) is before --start-date (%s).", end_date, start_date)
        sys.exit(1)

    all_results = fetch_all_cities_historic(start_date=start_date, end_date=end_date)
    successes = failures = 0
    for city in PAKISTAN_CITIES:
        records = all_results.get(city.name, [])
        if not records:
            logger.warning("[historic] [%s] no records, skipping upsert", city.name)
            failures += 1
            continue
        try:
            upsert_records(database_url, records, city_name=city.name)
            successes += 1
        except Exception:  # noqa: BLE001
            logger.exception("[historic] [%s] upsert failed", city.name)
            failures += 1
    _finish(successes, failures, len(PAKISTAN_CITIES))


# --- CLI --------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ingest", description="Open-Meteo → Neon Postgres.")
    p.add_argument("--mode", choices=["current", "historic"], required=True)
    p.add_argument("--start-date", default=None, metavar="YYYY-MM-DD",
                   help="[historic] First date (inclusive). Env: HISTORIC_START_DATE")
    p.add_argument("--end-date", default=None, metavar="YYYY-MM-DD",
                   help="[historic] Last date (inclusive). Env: HISTORIC_END_DATE")
    return p


def main() -> None:
    """Parse CLI args and dispatch to the appropriate mode handler."""
    args = _build_parser().parse_args()
    database_url = _require_env("DATABASE_URL")

    logger.info("Cities: %s", ", ".join(c.name for c in PAKISTAN_CITIES))

    try:
        init_db(database_url)
    except Exception as exc:
        logger.exception("Schema init failed: %s", exc)
        sys.exit(1)

    if args.mode == "current":
        run_current(database_url)
    elif args.mode == "historic":
        start_date = (
            _parse_date_arg(args.start_date, "--start-date")
            if args.start_date
            else _parse_date_arg(os.environ["HISTORIC_START_DATE"], "HISTORIC_START_DATE")
            if os.environ.get("HISTORIC_START_DATE")
            else date.today() - timedelta(days=HISTORIC_DEFAULT_LOOKBACK_DAYS)
        )
        end_date = (
            _parse_date_arg(args.end_date, "--end-date")
            if args.end_date
            else _parse_date_arg(os.environ["HISTORIC_END_DATE"], "HISTORIC_END_DATE")
            if os.environ.get("HISTORIC_END_DATE")
            else date.today() - timedelta(days=2)
        )
        run_historic(database_url, start_date, end_date)


if __name__ == "__main__":
    main()
