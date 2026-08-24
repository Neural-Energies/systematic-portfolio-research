"""Tests for the combined development candidate portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from systematic_research.candidate_portfolio import (
    build_breakout_forecast,
    build_candidate_portfolios,
    classify_candidate,
    group_mean_reversion_families,
    select_candidate_sleeves,
)


def test_breakout_forecast_is_directional_and_bounded() -> None:
    features = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "symbol": ["CL", "ZN"],
            "close_percentile_20s": [0.90, 0.10],
            "close_percentile_60s": [0.80, 0.20],
            "close_percentile_120s": [0.75, 0.25],
            "range_percentile_60s": [0.80, 0.80],
            "volume_percentile_60s": [0.70, 0.70],
        }
    )

    forecast = build_breakout_forecast(features)

    assert forecast.loc[forecast["symbol"].eq("CL"), "forecast"].item() > 0.0
    assert forecast.loc[forecast["symbol"].eq("ZN"), "forecast"].item() < 0.0
    assert forecast["forecast"].between(-1.0, 1.0).all()


def test_mean_reversion_variants_are_equal_weighted_within_family() -> None:
    index = pd.date_range("2024-01-02", periods=2, freq="B")
    strategy_returns = pd.DataFrame(
        {
            "a_one": [0.01, 0.03],
            "a_two": [0.03, 0.01],
            "b_one": [-0.02, 0.02],
        },
        index=index,
    )
    registry = pd.DataFrame(
        {
            "name": ["a_one", "a_two", "b_one"],
            "family": ["family_a", "family_a", "family_b"],
        }
    )

    grouped = group_mean_reversion_families(strategy_returns, registry)

    expected = pd.DataFrame(
        {
            "session_mr_family_a": [0.02, 0.02],
            "session_mr_family_b": [-0.02, 0.02],
        },
        index=index,
    )
    assert_frame_equal(grouped, expected)


def test_causal_portfolio_does_not_change_before_future_revision() -> None:
    index = pd.date_range("2024-01-02", periods=45, freq="B")
    base = pd.DataFrame(
        {
            "trend": np.sin(np.arange(45)) / 100.0,
            "reversal": np.cos(np.arange(45)) / 120.0,
        },
        index=index,
    )
    revised = base.copy()
    revised.iloc[-5:] = revised.iloc[-5:] * 100.0

    original_portfolios = build_candidate_portfolios(base)
    revised_portfolios = build_candidate_portfolios(revised)

    assert_frame_equal(original_portfolios.iloc[:-5], revised_portfolios.iloc[:-5])


def test_candidate_gate_is_deliberately_a_research_gate() -> None:
    assert classify_candidate(annualized_return=0.01, positive_folds=2) == "candidate"
    assert classify_candidate(annualized_return=0.01, positive_folds=1) == "research_further"
    assert classify_candidate(annualized_return=-0.01, positive_folds=2) == "research_further"
    assert classify_candidate(annualized_return=-0.01, positive_folds=1) == "reject"


def test_candidate_portfolio_funds_only_sleeves_that_clear_the_gate() -> None:
    returns = pd.DataFrame(
        {"accepted": [0.01, 0.02], "observe": [0.03, -0.02], "rejected": [-0.01, -0.02]}
    )
    summary = pd.DataFrame(
        {
            "strategy": ["accepted", "observe", "rejected"],
            "research_status": ["candidate", "research_further", "reject"],
        }
    )

    selected = select_candidate_sleeves(returns, summary)

    assert list(selected.columns) == ["accepted"]
