import numpy as np
import pandas as pd

from systematic_research.trend import (
    TREND_FEATURES,
    _balance_long_short,
    apply_regime_risk_overlay,
    build_trend_forecast,
    simulate_with_drawdown_controls,
)


def test_trend_forecast_preserves_direction_and_is_bounded() -> None:
    rows = 80
    frame = pd.DataFrame({"trading_date": pd.date_range("2020-01-01", periods=rows), "symbol": "A"})
    for feature in TREND_FEATURES:
        frame[feature] = np.linspace(0.01, 1.0, rows)
    result = build_trend_forecast(frame)
    assert result["forecast"].iloc[-1] > 0.0
    assert result["forecast"].abs().max() <= 1.0


def test_long_short_balance_has_unit_gross_and_bounded_net() -> None:
    weights = _balance_long_short(pd.Series({"A": 3.0, "B": 1.0, "C": -1.0, "D": -2.0}), 0.25)
    assert np.isclose(weights.abs().sum(), 1.0)
    assert abs(weights.sum()) <= 0.25 + 1e-12
    assert (weights > 0).any() and (weights < 0).any()


def test_simulation_lags_positions_and_charges_costs() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    returns = pd.DataFrame({"A": [0.10, 0.10, 0.10]}, index=dates)
    targets = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    net, implemented = simulate_with_drawdown_controls(returns, targets, cost_bps=10.0)
    assert implemented.iloc[0, 0] == 0.0
    assert np.isclose(net.iloc[0], 0.0)
    # Turnover is one-way: opening 100% exposure is 50% turnover under this convention.
    assert np.isclose(net.iloc[1], 0.0995)


def test_uncertain_regime_does_not_change_risk() -> None:
    date = pd.Timestamp("2020-01-01")
    weights = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=[date])
    regime = pd.DataFrame(
        {
            "trading_date": [date],
            "ensemble_low_volatility_trend": [0.25],
            "ensemble_high_volatility_trend": [0.25],
            "ensemble_stress_transition": [0.25],
            "ensemble_confidence": [0.0],
        }
    )
    result = apply_regime_risk_overlay(weights, regime)
    pd.testing.assert_frame_equal(result, weights)
