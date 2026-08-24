from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.portfolio import (
    inverse_volatility_weights,
    minimum_variance_weights,
    net_returns,
)


def test_net_returns_lag_signal_and_charge_turnover() -> None:
    index = pd.date_range("2025-01-01", periods=3, tz="UTC")
    returns = pd.DataFrame({"asset": [0.10, 0.02, -0.01]}, index=index)
    targets = pd.DataFrame({"asset": [1.0, 1.0, 0.0]}, index=index)
    result = net_returns(returns, targets, cost_bps=10.0)
    np.testing.assert_allclose(result, [0.0, 0.0195, -0.01])


def test_inverse_volatility_respects_cap() -> None:
    volatility = pd.Series({"a": 0.01, "b": 0.20, "c": 0.20})
    weights = inverse_volatility_weights(volatility, maximum_weight=0.50)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.50


def test_minimum_variance_prefers_lower_risk_asset() -> None:
    covariance = pd.DataFrame([[0.01, 0.0], [0.0, 0.04]], index=["a", "b"], columns=["a", "b"])
    weights = minimum_variance_weights(covariance)
    assert np.isclose(weights.sum(), 1.0)
    assert weights["a"] > weights["b"]
