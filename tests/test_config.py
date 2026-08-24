from __future__ import annotations

from systematic_research.config import CostConfig, load_config


def test_costs_sum_in_basis_points() -> None:
    costs = CostConfig(commission_bps=0.5, spread_bps=1.0, slippage_bps=1.5)
    assert costs.total_bps == 3.0


def test_workspace_config_loads() -> None:
    config = load_config("config/research.yaml")
    assert config.research.signal_lag_periods >= 1
    assert config.costs.total_bps == 3.5
    assert config.portfolio.maximum_gross_leverage == 2.0
