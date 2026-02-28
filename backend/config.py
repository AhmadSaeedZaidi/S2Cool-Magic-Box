"""Configuration for S2Cool backend domain defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoolingConfig:
    """Business constants for hybrid cooling logic."""

    cooling_capacity_kw_thermal: float = 20.0
    cop_ideal: float = 10.0
    default_panel_count: int = 10
    panel_watt_rating: float = 640.0
    cooling_temp_threshold_c: float = 24.0

    @property
    def electrical_load_kw(self) -> float:
        """Required electrical power from idealized COP contract."""
        return self.cooling_capacity_kw_thermal / self.cop_ideal


DEFAULT_CONFIG = CoolingConfig()
