"""Tests for ingest.db — UPSERT query formation (mocked, no real DB)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

from ingest.api import SolarWeatherRecord
from ingest.db import CREATE_TABLE_SQL, UPSERT_SQL, init_db, upsert_records


def _record(city: str = "Lahore", temp: float = 30.0) -> SolarWeatherRecord:
    return SolarWeatherRecord(
        city_name=city,
        timestamp=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc),
        temperature_2m=temp,
        relative_humidity_2m=55.0,
        wind_speed_10m=12.0,
        cloud_cover=20.0,
        shortwave_radiation=600.0,
        direct_radiation=400.0,
        direct_normal_irradiance=500.0,
        diffuse_radiation=200.0,
    )


class TestSqlTemplates:
    def test_create_table_contains_primary_key(self):
        assert "PRIMARY KEY (city_name, timestamp)" in CREATE_TABLE_SQL

    def test_upsert_contains_on_conflict(self):
        assert "ON CONFLICT (city_name, timestamp)" in UPSERT_SQL

    def test_upsert_updates_all_metric_columns(self):
        for col in [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "cloud_cover",
            "shortwave_radiation",
            "direct_radiation",
            "direct_normal_irradiance",
            "diffuse_radiation",
        ]:
            assert f"EXCLUDED.{col}" in UPSERT_SQL


class TestInitDb:
    @patch("ingest.db.get_connection")
    def test_executes_create_table(self, mock_conn_cm):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn_cm.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.return_value.__exit__ = MagicMock(return_value=False)

        init_db("postgresql://test")
        mock_cursor.execute.assert_called_once_with(CREATE_TABLE_SQL)


class TestUpsertRecords:
    @patch("ingest.db.get_connection")
    def test_empty_records_skips(self, mock_conn_cm):
        upsert_records("postgresql://test", [], city_name="Lahore")
        mock_conn_cm.assert_not_called()

    @patch("ingest.db.psycopg2.extras.execute_values")
    @patch("ingest.db.get_connection")
    def test_calls_execute_values_with_correct_tuples(self, mock_conn_cm, mock_exec):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.statusmessage = "INSERT 0 1"
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn_cm.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.return_value.__exit__ = MagicMock(return_value=False)

        rec = _record()
        upsert_records("postgresql://test", [rec])

        mock_exec.assert_called_once()
        args = mock_exec.call_args
        sql_arg = args[0][1]
        rows_arg = args[0][2]
        assert sql_arg == UPSERT_SQL
        assert len(rows_arg) == 1
        assert rows_arg[0][0] == "Lahore"
        assert rows_arg[0][1] == rec.timestamp
        assert rows_arg[0][2] == 30.0  # temperature_2m

    @patch("ingest.db.psycopg2.extras.execute_values")
    @patch("ingest.db.get_connection")
    def test_batch_multiple_records(self, mock_conn_cm, mock_exec):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.statusmessage = "INSERT 0 3"
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn_cm.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn_cm.return_value.__exit__ = MagicMock(return_value=False)

        records = [_record(temp=i) for i in range(3)]
        upsert_records("postgresql://test", records)

        rows_arg = mock_exec.call_args[0][2]
        assert len(rows_arg) == 3
