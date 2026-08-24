from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.broad_strategy_search import (
    StrategySpec,
    build_signal,
    classify_research_tier,
    strategy_specs,
)


def test_registry_has_ten_strategies_in_each_theme() -> None:
    specs = strategy_specs()
    counts = pd.Series([spec.theme for spec in specs]).value_counts()
    assert len(specs) == 60
    assert counts.to_dict() == {
        "momentum": 10,
        "trend": 10,
        "mean_reversion": 10,
        "breakout": 10,
        "relative_value": 10,
        "lead_lag": 10,
    }


def test_signal_scaling_uses_only_prior_observations() -> None:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    features = pd.DataFrame(
        {
            "trading_date": dates,
            "symbol": "ES",
            "momentum_2s": np.arange(1.0, 81.0),
        }
    )
    spec = StrategySpec("test", "momentum", "momentum_2s")
    baseline = build_signal(features, spec)
    changed = features.copy()
    changed.loc[changed.index[-1], "momentum_2s"] = 1_000_000.0
    revised = build_signal(changed, spec)
    pd.testing.assert_frame_equal(baseline.iloc[:-1], revised.iloc[:-1])


def test_research_tiers_report_results_without_hard_rejection() -> None:
    assert classify_research_tier(2.1, 4) == "A_promising"
    assert classify_research_tier(1.0, 2) == "B_candidate"
    assert classify_research_tier(0.2, 1) == "C_exploratory_positive"
    assert classify_research_tier(-0.5, 0) == "D_exploratory_negative"
