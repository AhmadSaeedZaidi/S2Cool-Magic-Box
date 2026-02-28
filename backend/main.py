"""FastAPI backend exposing S2Cool math-model decision endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from .schemas import (
    DailySimulationRequest,
    DailySimulationResponse,
    DecisionRequest,
    DecisionResponse,
    PshRequest,
    PshResponse,
)
from .services.math_model import MathDecisionEngine

app = FastAPI(title="S2Cool Backend API", version="0.1.0")
engine = MathDecisionEngine()
STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/health")
def health() -> dict[str, str]:
    """Service health endpoint."""
    return {"status": "ok"}


@app.get("/")
def frontend() -> FileResponse:
    """Serve ultra-basic demo frontend."""
    return FileResponse(STATIC_INDEX)


@app.post("/v1/predict/math", response_model=DecisionResponse)
def predict_math_decision(request: DecisionRequest) -> DecisionResponse:
    """Return deterministic source-mode decision for one timestamp."""
    return engine.make_decision(
        city=request.city,
        timestamp_utc=request.timestamp_utc,
        predicted_ghi_wm2=request.predicted_ghi_wm2,
        predicted_ambient_temp_c=request.predicted_ambient_temp_c,
        panel_count=request.panel_count,
        panel_watt_rating=request.panel_watt_rating,
        operating_hours_enabled=request.operating_hours_enabled,
    )


@app.post("/v1/psh", response_model=PshResponse)
def compute_psh(request: PshRequest) -> PshResponse:
    """Calculate raw and adjusted PSH from hourly GHI predictions."""
    psh_raw, psh_adjusted, monthly_factor = engine.calculate_psh(
        hourly_ghi_wm2=request.hourly_ghi_wm2,
        month=request.month,
    )
    return PshResponse(
        month=request.month,
        psh_raw=psh_raw,
        psh_adjusted=psh_adjusted,
        monthly_factor=monthly_factor,
    )


@app.post("/v1/simulate/day", response_model=DailySimulationResponse)
def simulate_day(request: DailySimulationRequest) -> DailySimulationResponse:
    """Run day simulation over hourly predicted values using logic gates."""
    no_cooling_hours = 0
    solar_hours = 0
    grid_hours = 0
    solar_energy_kwh = 0.0
    grid_energy_kwh = 0.0

    month = request.hours[0].timestamp_utc.month
    psh_raw, psh_adjusted, _ = engine.calculate_psh(
        hourly_ghi_wm2=[hour.predicted_ghi_wm2 for hour in request.hours],
        month=month,
    )

    for hour in request.hours:
        decision = engine.make_decision(
            city=request.city,
            timestamp_utc=hour.timestamp_utc,
            predicted_ghi_wm2=hour.predicted_ghi_wm2,
            predicted_ambient_temp_c=hour.predicted_ambient_temp_c,
            panel_count=request.panel_count,
            panel_watt_rating=request.panel_watt_rating,
            operating_hours_enabled=True,
        )

        if decision.mode == "NO_COOLING_NEEDED":
            no_cooling_hours += 1
            continue
        if decision.mode == "RUN_ON_SOLAR":
            solar_hours += 1
            solar_energy_kwh += decision.electrical_load_kw
        else:
            grid_hours += 1
            grid_energy_kwh += decision.electrical_load_kw

    return DailySimulationResponse(
        city=request.city,
        total_hours=len(request.hours),
        no_cooling_hours=no_cooling_hours,
        solar_hours=solar_hours,
        grid_hours=grid_hours,
        solar_energy_kwh=round(solar_energy_kwh, 4),
        grid_energy_kwh=round(grid_energy_kwh, 4),
        psh_adjusted=psh_adjusted,
    )
