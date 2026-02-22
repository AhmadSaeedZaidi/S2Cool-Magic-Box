# s2cool-ingest

Automated Open-Meteo solar & weather data ingestion for Pakistani cities into Neon Postgres.

## Setup
Built with Python 3.11+ and Poetry.

```bash
make install
```

## Usage
Run the current forecast ingestion:
```bash
make run-current
```

Run the historic backfill:
```bash
make run-historic
```