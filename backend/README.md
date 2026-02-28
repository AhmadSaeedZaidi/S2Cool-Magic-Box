# S2Cool FastAPI Backend (Math Model First)

This backend exposes a deterministic "math model" aligned to project constraints:
- Cooling capacity: 20 kW thermal
- Ideal COP: 10
- Electrical load: 2 kW
- Rule gates:
  - cool ambient -> `NO_COOLING_NEEDED`
  - solar >= load -> `RUN_ON_SOLAR`
  - solar < load -> `RUN_ON_GRID`

## Run

```bash
pip install fastapi uvicorn pydantic
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

- `GET /health`
- `POST /v1/predict/math`
- `POST /v1/psh`
- `POST /v1/simulate/day`

## Example payload for `/v1/predict/math`

```json
{
  "city": "Lahore",
  "timestamp_utc": "2026-03-01T12:00:00Z",
  "predicted_ghi_wm2": 780,
  "predicted_ambient_temp_c": 34,
  "panel_count": 10,
  "panel_watt_rating": 640,
  "operating_hours_enabled": true
}
```
