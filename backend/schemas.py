"""Pydantic request/response schemas for S2Cool backend endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    """Single-timestep hybrid cooling decision input."""

    city: str = Field(..., examples=["Lahore"])
    timestamp_utc: datetime
    predicted_ghi_wm2: float = Field(..., ge=0.0)
    predicted_ambient_temp_c: float
    panel_count: int = Field(default=10, ge=1)
    panel_watt_rating: float = Field(default=640.0, gt=0.0)
    operating_hours_enabled: bool = True


class DecisionResponse(BaseModel):
    """Single-timestep cooling source decision output."""

    city: str
    timestamp_utc: datetime
    mode: str
    cooling_needed: bool
    no_cooling_needed_banner: str | None = None
    predicted_ghi_wm2: float
    predicted_ambient_temp_c: float
    solar_generation_kw: float
    electrical_load_kw: float
    cooling_capacity_kw_thermal: float
    cop_ideal: float


class PshRequest(BaseModel):
    """Daily PSH request using predicted hourly GHI values."""

    month: int = Field(..., ge=1, le=12)
    hourly_ghi_wm2: list[float] = Field(..., min_length=1)


class PshResponse(BaseModel):
    """Peak Sun Hour output with monthly adjustment."""

    month: int
    psh_raw: float
    psh_adjusted: float
    monthly_factor: float


class DailyHourInput(BaseModel):
    """One hourly forecast point for daily simulation."""

    timestamp_utc: datetime
    predicted_ghi_wm2: float = Field(..., ge=0.0)
    predicted_ambient_temp_c: float


class DailySimulationRequest(BaseModel):
    """Daily simulation request over hourly predicted values."""

    city: str
    panel_count: int = Field(default=10, ge=1)
    panel_watt_rating: float = Field(default=640.0, gt=0.0)
    hours: list[DailyHourInput] = Field(..., min_length=1)


class DailySimulationResponse(BaseModel):
    """Daily simulation summary output."""

    city: str
    total_hours: int
    no_cooling_hours: int
    solar_hours: int
    grid_hours: int
    solar_energy_kwh: float
    grid_energy_kwh: float
    psh_adjusted: float
