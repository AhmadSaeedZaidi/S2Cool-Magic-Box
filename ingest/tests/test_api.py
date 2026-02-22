"""Tests for ingest.api — Open-Meteo integration (mocked, no network calls)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ingest.api import (
    CURRENT_API_URL,
    HISTORIC_API_URL,
    HOURLY_VARIABLES,
    CityConfig,
    SolarWeatherRecord,
    _parse_hourly,
    fetch_all_cities_current,
    fetch_city_weather,
    fetch_city_weather_historic,
)

ISLAMABAD = CityConfig("Islamabad", 33.6844, 73.0479)


def _make_hourly_payload(
    times: list[str],
    fill: float = 1.0,
) -> dict:
    """Build a minimal Open-Meteo-shaped ``hourly`` dict."""
    hourly = {"time": times}
    for var in HOURLY_VARIABLES:
        hourly[var] = [fill] * len(times)
    return hourly


class TestParseHourly:
    def test_empty_time_array(self):
        assert _parse_hourly("X", {"time": []}) == []

    def test_missing_time_key(self):
        assert _parse_hourly("X", {}) == []

    def test_valid_records(self):
        hourly = _make_hourly_payload(["2025-06-01T00:00", "2025-06-01T01:00"])
        records = _parse_hourly("Lahore", hourly)
        assert len(records) == 2
        assert all(isinstance(r, SolarWeatherRecord) for r in records)
        assert records[0].city_name == "Lahore"
        assert records[0].timestamp == datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
        assert records[0].temperature_2m == 1.0

    def test_unparseable_timestamp_skipped(self):
        hourly = _make_hourly_payload(["NOT-A-DATE", "2025-06-01T01:00"])
        records = _parse_hourly("X", hourly)
        assert len(records) == 1

    def test_missing_variable_column_yields_none(self):
        hourly = {"time": ["2025-06-01T00:00"]}
        records = _parse_hourly("X", hourly)
        assert len(records) == 1
        assert records[0].temperature_2m is None
        assert records[0].diffuse_radiation is None


class TestFetchCityWeather:
    @patch("ingest.api._build_session")
    def test_success(self, mock_build):
        hourly = _make_hourly_payload(["2025-06-01T00:00"], fill=25.5)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": hourly}
        mock_resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = mock_resp
        mock_build.return_value = session

        records = fetch_city_weather(ISLAMABAD)
        assert len(records) == 1
        assert records[0].temperature_2m == 25.5
        session.get.assert_called_once()

    @patch("ingest.api._build_session")
    def test_missing_hourly_raises(self, mock_build):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"latitude": 33.68}
        mock_resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = mock_resp
        mock_build.return_value = session

        with pytest.raises(ValueError, match="hourly"):
            fetch_city_weather(ISLAMABAD)


class TestFetchAllCitiesCurrent:
    @patch("ingest.api.fetch_city_weather")
    def test_skips_failing_city(self, mock_fetch):
        ok_record = SolarWeatherRecord(
            city_name="Lahore",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            temperature_2m=30.0,
            relative_humidity_2m=50.0,
            wind_speed_10m=10.0,
            cloud_cover=20.0,
            shortwave_radiation=500.0,
            direct_radiation=300.0,
            direct_normal_irradiance=400.0,
            diffuse_radiation=200.0,
        )
        mock_fetch.side_effect = [
            RuntimeError("network"),  # Islamabad fails
            [ok_record],  # Lahore ok
            RuntimeError("network"),  # Karachi fails
            [ok_record],  # Peshawar ok
        ]
        result = fetch_all_cities_current()
        assert "Lahore" in result
        assert "Islamabad" not in result


class TestFetchCityWeatherHistoric:
    @patch("ingest.api.time.sleep")
    @patch("ingest.api._build_session")
    def test_chunking(self, mock_build, mock_sleep):
        """A 100-day range with chunk_days=50 should produce 2 API calls."""
        hourly = _make_hourly_payload(["2025-01-01T00:00"])
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hourly": hourly}
        mock_resp.raise_for_status = MagicMock()

        session = MagicMock()
        session.get.return_value = mock_resp
        mock_build.return_value = session

        records = fetch_city_weather_historic(
            ISLAMABAD,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 4, 10),
            chunk_days=50,
            sleep_seconds=0,
        )
        assert session.get.call_count == 2
        assert len(records) == 2  # 1 record per chunk response
        mock_sleep.assert_called_once()

    @patch("ingest.api.time.sleep")
    @patch("ingest.api._build_session")
    def test_failed_chunk_skips_gracefully(self, mock_build, mock_sleep):
        session = MagicMock()
        session.get.side_effect = TimeoutError("boom")
        mock_build.return_value = session

        records = fetch_city_weather_historic(
            ISLAMABAD,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 10),
            chunk_days=90,
            sleep_seconds=0,
        )
        assert records == []
