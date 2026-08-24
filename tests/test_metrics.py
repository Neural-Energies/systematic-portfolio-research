from __future__ import annotations

import math

import pandas as pd

from systematic_research.metrics import expected_shortfall, maximum_drawdown


def test_maximum_drawdown_uses_compounded_wealth() -> None:
    returns = pd.Series([0.10, -0.20, 0.05])
    assert math.isclose(maximum_drawdown(returns), -0.20)


def test_expected_shortfall_reports_positive_loss() -> None:
    returns = pd.Series([-0.10, -0.05, 0.01, 0.02])
    assert math.isclose(expected_shortfall(returns, confidence=0.75), 0.10)
