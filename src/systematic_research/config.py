"""Typed configuration loading for reproducible research runs."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ResearchConfig(BaseModel):
    """Core assumptions shared by research and portfolio construction."""

    model_config = ConfigDict(extra="forbid")

    base_currency: str = "USD"
    timezone: str = "UTC"
    annualization_days: int = Field(default=252, gt=0)
    random_seed: int = 42
    signal_lag_periods: int = Field(default=1, ge=1)


class CostConfig(BaseModel):
    """Simple, explicit one-way implementation-cost assumptions."""

    model_config = ConfigDict(extra="forbid")

    commission_bps: float = Field(default=0.0, ge=0.0)
    spread_bps: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)
    roll_bps: float = Field(default=0.0, ge=0.0)

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.spread_bps + self.slippage_bps + self.roll_bps


class PortfolioConfig(BaseModel):
    """Portfolio-level limits used by construction and monitoring."""

    model_config = ConfigDict(extra="forbid")

    target_volatility: float = Field(default=0.10, gt=0.0)
    maximum_gross_leverage: float = Field(default=2.0, gt=0.0)
    maximum_asset_weight: float = Field(default=0.25, gt=0.0, le=1.0)


class WorkspaceConfig(BaseModel):
    """Validated top-level workspace configuration."""

    model_config = ConfigDict(extra="forbid")

    research: ResearchConfig
    costs: CostConfig
    portfolio: PortfolioConfig


def load_config(path: str | Path) -> WorkspaceConfig:
    """Load YAML and fail fast on missing or invalid assumptions."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return WorkspaceConfig.model_validate(payload)
