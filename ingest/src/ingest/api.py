"""Open-Meteo API integration (current forecast & historical archive)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, NamedTuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 15

HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]

_RETRY_CONFIG = Retry(
    total=3,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

CURRENT_API_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT_PAST_HOURS = 3
CURRENT_FORECAST_HOURS = 1

HISTORIC_API_URL = "https://archive-api.open-meteo.com/v1/archive"
HISTORIC_CHUNK_DAYS = 90
HISTORIC_INTER_REQUEST_SLEEP = 2.0
HISTORIC_DEFAULT_LOOKBACK_DAYS = 365


class CityConfig(NamedTuple):
    """Coordinates and label for a tracked city."""

    name: str
    latitude: float
    longitude: float


PAKISTAN_CITIES: list[CityConfig] = [
    CityConfig("Islamabad", 33.6844, 73.0479),
    CityConfig("Lahore", 31.5204, 74.3587),
    CityConfig("Karachi", 24.8607, 67.0011),
    CityConfig("Peshawar", 34.0151, 71.5249),
]


@dataclass
class SolarWeatherRecord:
    """Single hourly observation for one city."""

    city_name: str
    timestamp: datetime
    temperature_2m: float | None
    relative_humidity_2m: float | None
    wind_speed_10m: float | None
    cloud_cover: float | None
    shortwave_radiation: float | None
    direct_radiation: float | None
    direct_normal_irradiance: float | None
    diffuse_radiation: float | None


# --- Internal helpers -------------------------------------------------------


def _build_session() -> requests.Session:
    """Create a ``requests.Session`` with retry/backoff on both schemes."""
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY_CONFIG)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _parse_hourly(city_name: str, hourly: dict[str, Any]) -> list[SolarWeatherRecord]:
    """Convert Open-Meteo parallel-array ``hourly`` block into records.

    Unparseable timestamps are skipped with a warning.
    """
    times: list[str] = hourly.get("time") or []
    if not times:
        logger.warning("[%s] No hourly time entries in response.", city_name)
        return []

    def _col(key: str) -> list[Any]:
        values = hourly.get(key)
        return values if values is not None else [None] * len(times)

    records: list[SolarWeatherRecord] = []
    for i, ts_str in enumerate(times):
        try:
            ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("[%s] Skipping unparseable timestamp '%s'.", city_name, ts_str)
            continue
        records.append(
            SolarWeatherRecord(
                city_name=city_name,
                timestamp=ts,
                temperature_2m=_col("temperature_2m")[i],
                relative_humidity_2m=_col("relative_humidity_2m")[i],
                wind_speed_10m=_col("wind_speed_10m")[i],
                cloud_cover=_col("cloud_cover")[i],
                shortwave_radiation=_col("shortwave_radiation")[i],
                direct_radiation=_col("direct_radiation")[i],
                direct_normal_irradiance=_col("direct_normal_irradiance")[i],
                diffuse_radiation=_col("diffuse_radiation")[i],
            )
        )
    return records


def _get_json(
    session: requests.Session, url: str, params: dict[str, Any], label: str
) -> dict[str, Any]:
    """GET *url* with *params* and return parsed JSON.

    Raises:
        requests.HTTPError: Non-2xx after retries.
        requests.exceptions.Timeout: Request exceeded deadline.
    """
    try:
        response = session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("[%s] Request timed out after %s s.", label, REQUEST_TIMEOUT_SECONDS)
        raise
    except requests.exceptions.RetryError as exc:
        logger.error("[%s] All retries exhausted: %s", label, exc)
        raise
    return response.json()  # type: ignore[return-value]


# --- Current-mode API -------------------------------------------------------


def fetch_city_weather(city: CityConfig) -> list[SolarWeatherRecord]:
    """Fetch recent hourly data from the forecast endpoint for *city*.

    Args:
        city: Target city coordinates.

    Returns:
        Hourly ``SolarWeatherRecord`` list covering the last few hours.

    Raises:
        requests.HTTPError: Non-2xx after retries.
        ValueError: Missing ``hourly`` key in response payload.
    """
    params: dict[str, Any] = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "past_hours": CURRENT_PAST_HOURS,
        "forecast_hours": CURRENT_FORECAST_HOURS,
    }
    logger.info(
        "[current] %s — past %dh + forecast %dh",
        city.name, CURRENT_PAST_HOURS, CURRENT_FORECAST_HOURS,
    )
    payload = _get_json(_build_session(), CURRENT_API_URL, params, city.name)
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(f"[{city.name}] 'hourly' key missing in forecast response")
    records = _parse_hourly(city.name, hourly)
    logger.info("[current] [%s] parsed %d records.", city.name, len(records))
    return records


def fetch_all_cities_current(
    cities: list[CityConfig] | None = None,
) -> dict[str, list[SolarWeatherRecord]]:
    """Fetch current data for every city, skipping failures."""
    cities = cities or PAKISTAN_CITIES
    results: dict[str, list[SolarWeatherRecord]] = {}
    for city in cities:
        try:
            results[city.name] = fetch_city_weather(city)
        except Exception:  # noqa: BLE001
            logger.exception("[current] [%s] skipping due to error", city.name)
    return results


fetch_all_cities = fetch_all_cities_current  # backward-compat alias


# --- Historic-mode API ------------------------------------------------------


def fetch_city_weather_historic(
    city: CityConfig,
    start_date: date,
    end_date: date,
    chunk_days: int = HISTORIC_CHUNK_DAYS,
    sleep_seconds: float = HISTORIC_INTER_REQUEST_SLEEP,
) -> list[SolarWeatherRecord]:
    """Fetch hourly archive data for *city* in chunked requests.

    Args:
        city: Target city.
        start_date: First date (inclusive).
        end_date: Last date (inclusive, must be ≥2 days before today).
        chunk_days: Days per API call.
        sleep_seconds: Pause between requests for rate-limit compliance.

    Returns:
        All ``SolarWeatherRecord`` across the date range.
    """
    all_records: list[SolarWeatherRecord] = []
    session = _build_session()
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days - 1), end_date)
        params: dict[str, Any] = {
            "latitude": city.latitude,
            "longitude": city.longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "UTC",
            "start_date": chunk_start.isoformat(),
            "end_date": chunk_end.isoformat(),
        }
        logger.info(
            "[historic] [%s] %s → %s",
            city.name, chunk_start.isoformat(), chunk_end.isoformat(),
        )
        try:
            payload = _get_json(session, HISTORIC_API_URL, params, city.name)
        except Exception:  # noqa: BLE001
            logger.exception(
                "[historic] [%s] chunk %s→%s failed, skipping",
                city.name, chunk_start, chunk_end,
            )
            chunk_start = chunk_end + timedelta(days=1)
            continue

        hourly = payload.get("hourly")
        if isinstance(hourly, dict):
            chunk_records = _parse_hourly(city.name, hourly)
            logger.info(
                "[historic] [%s] chunk %s→%s — %d records",
                city.name, chunk_start, chunk_end, len(chunk_records),
            )
            all_records.extend(chunk_records)
        else:
            logger.warning(
                "[historic] [%s] 'hourly' missing for %s→%s, skipping",
                city.name, chunk_start, chunk_end,
            )

        chunk_start = chunk_end + timedelta(days=1)
        if chunk_start <= end_date:
            time.sleep(sleep_seconds)

    logger.info("[historic] [%s] total records: %d", city.name, len(all_records))
    return all_records


def fetch_all_cities_historic(
    start_date: date,
    end_date: date,
    cities: list[CityConfig] | None = None,
    chunk_days: int = HISTORIC_CHUNK_DAYS,
    sleep_seconds: float = HISTORIC_INTER_REQUEST_SLEEP,
) -> dict[str, list[SolarWeatherRecord]]:
    """Fetch historic data for every city, skipping failures.

    Args:
        start_date: First date (inclusive).
        end_date: Last date (inclusive).
        cities: City list; defaults to ``PAKISTAN_CITIES``.
        chunk_days: Days per API call.
        sleep_seconds: Pause between chunk requests.
    """
    cities = cities or PAKISTAN_CITIES
    results: dict[str, list[SolarWeatherRecord]] = {}
    for city in cities:
        try:
            results[city.name] = fetch_city_weather_historic(
                city,
                start_date=start_date,
                end_date=end_date,
                chunk_days=chunk_days,
                sleep_seconds=sleep_seconds,
            )
        except Exception:  # noqa: BLE001
            logger.exception("[historic] [%s] skipping city due to error", city.name)
    return results
