"""Portfolio performance and tail-risk metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized sample volatility."""
    return float(returns.dropna().std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> float:
    """Annualized arithmetic Sharpe ratio with an annual risk-free rate."""
    clean = returns.dropna()
    volatility = clean.std(ddof=1)
    if volatility == 0.0:
        return float("nan")
    period_risk_free = risk_free_rate / periods_per_year
    return float((clean.mean() - period_risk_free) / volatility * np.sqrt(periods_per_year))


def drawdown(returns: pd.Series) -> pd.Series:
    """Drawdown series from compounded simple returns."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return wealth / wealth.cummax() - 1.0


def maximum_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough loss."""
    return float(drawdown(returns).min())


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical expected loss conditional on exceeding value-at-risk."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    clean = returns.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return float("nan")
    cutoff = np.quantile(clean, 1.0 - confidence)
    return float(-clean[clean <= cutoff].mean())
