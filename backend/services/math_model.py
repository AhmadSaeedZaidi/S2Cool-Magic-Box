"""Deterministic math/rules model for hybrid cooling control decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import CoolingConfig, DEFAULT_CONFIG
from ..schemas import DecisionResponse


@dataclass
class MathDecisionEngine:
    """Rule-based decision engine using project hard constraints."""

    config: CoolingConfig = DEFAULT_CONFIG

    MONTHLY_PSH_FACTORS: dict[int, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.MONTHLY_PSH_FACTORS is None:
            self.MONTHLY_PSH_FACTORS = {
                1: 0.78,
                2: 0.82,
                3: 0.90,
                4: 1.00,
                5: 1.08,
                6: 1.12,
                7: 1.10,
                8: 1.04,
                9: 0.96,
                10: 0.88,
                11: 0.82,
                12: 0.76,
            }

    def estimate_solar_generation_kw(
        self,
        ghi_wm2: float,
        panel_count: int,
        panel_watt_rating: float,
    ) -> float:
        """Estimate instantaneous PV generation from GHI and panel capacity.

        Uses linear irradiance scaling against STC 1000 W/m².
        """
        irradiance_factor = max(0.0, ghi_wm2) / 1000.0
        dc_kw = panel_count * panel_watt_rating / 1000.0
        return dc_kw * irradiance_factor

    def needs_cooling(self, ambient_temp_c: float, operating_hours_enabled: bool = True) -> bool:
        """Determine if cooling is required by threshold and schedule gate."""
        if not operating_hours_enabled:
            return False
        return ambient_temp_c >= self.config.cooling_temp_threshold_c

    def make_decision(
        self,
        *,
        city: str,
        timestamp_utc: datetime,
        predicted_ghi_wm2: float,
        predicted_ambient_temp_c: float,
        panel_count: int,
        panel_watt_rating: float,
        operating_hours_enabled: bool,
    ) -> DecisionResponse:
        """Apply project logic gates to determine operating source mode."""
        load_kw = self.config.electrical_load_kw
        solar_kw = self.estimate_solar_generation_kw(predicted_ghi_wm2, panel_count, panel_watt_rating)
        cooling_needed = self.needs_cooling(predicted_ambient_temp_c, operating_hours_enabled)

        if not cooling_needed:
            mode = "NO_COOLING_NEEDED"
            banner = "NO COOLING NEEDED"
        elif solar_kw >= load_kw:
            mode = "RUN_ON_SOLAR"
            banner = None
        else:
            mode = "RUN_ON_GRID"
            banner = None

        return DecisionResponse(
            city=city,
            timestamp_utc=timestamp_utc,
            mode=mode,
            cooling_needed=cooling_needed,
            no_cooling_needed_banner=banner,
            predicted_ghi_wm2=predicted_ghi_wm2,
            predicted_ambient_temp_c=predicted_ambient_temp_c,
            solar_generation_kw=round(solar_kw, 4),
            electrical_load_kw=round(load_kw, 4),
            cooling_capacity_kw_thermal=self.config.cooling_capacity_kw_thermal,
            cop_ideal=self.config.cop_ideal,
        )

    def calculate_psh(self, hourly_ghi_wm2: list[float], month: int) -> tuple[float, float, float]:
        """Compute raw and month-adjusted peak sun hours."""
        psh_raw = sum(max(0.0, value) for value in hourly_ghi_wm2) / 1000.0
        monthly_factor = self.MONTHLY_PSH_FACTORS.get(month, 1.0)
        psh_adjusted = psh_raw * monthly_factor
        return round(psh_raw, 4), round(psh_adjusted, 4), monthly_factor
