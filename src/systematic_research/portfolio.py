"""Bias-aware portfolio construction and implementation accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize


def lag_positions(target_positions: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Apply positions no earlier than a later observation to prevent look-ahead bias."""
    if periods < 1:
        raise ValueError("positions must be lagged by at least one period")
    return target_positions.shift(periods).fillna(0.0)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """One-way portfolio turnover: half the absolute change in weights."""
    return weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0


def net_returns(
    asset_returns: pd.DataFrame,
    target_weights: pd.DataFrame,
    cost_bps: float = 0.0,
    signal_lag_periods: int = 1,
) -> pd.Series:
    """Portfolio returns after lagging positions and charging linear trading costs."""
    if cost_bps < 0.0:
        raise ValueError("cost_bps cannot be negative")
    returns, weights = asset_returns.align(target_weights, join="inner", axis=0)
    returns, weights = returns.align(weights, join="inner", axis=1)
    implemented = lag_positions(weights, signal_lag_periods)
    gross = (implemented * returns).sum(axis=1)
    costs = turnover(implemented) * cost_bps / 10_000.0
    return gross - costs


def inverse_volatility_weights(volatility: pd.Series, maximum_weight: float = 1.0) -> pd.Series:
    """Long-only inverse-volatility weights with an iterative per-asset cap."""
    if volatility.empty or (volatility <= 0.0).any():
        raise ValueError("volatilities must be non-empty and positive")
    if not 0.0 < maximum_weight <= 1.0:
        raise ValueError("maximum_weight must be in (0, 1]")
    if maximum_weight * len(volatility) < 1.0:
        raise ValueError("maximum_weight is infeasible for the number of assets")

    scores = 1.0 / volatility.astype(float)
    weights = pd.Series(0.0, index=volatility.index)
    remaining = pd.Series(True, index=volatility.index)
    budget = 1.0
    while remaining.any():
        proposed = scores[remaining] / scores[remaining].sum() * budget
        capped = proposed > maximum_weight
        if not capped.any():
            weights.loc[remaining] = proposed
            break
        capped_index = proposed[capped].index
        weights.loc[capped_index] = maximum_weight
        remaining.loc[capped_index] = False
        budget = 1.0 - float(weights.sum())
    return weights


def minimum_variance_weights(covariance: pd.DataFrame, maximum_weight: float = 1.0) -> pd.Series:
    """Long-only, fully invested minimum-variance portfolio."""
    if covariance.empty or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a non-empty square matrix")
    if not covariance.index.equals(covariance.columns):
        raise ValueError("covariance index and columns must match")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("covariance must be symmetric")
    asset_count = len(covariance)
    if not 0.0 < maximum_weight <= 1.0 or maximum_weight * asset_count < 1.0:
        raise ValueError("maximum_weight is infeasible")

    matrix = covariance.to_numpy(dtype=float)
    initial = np.full(asset_count, 1.0 / asset_count)

    def variance(weights: NDArray[np.float64]) -> float:
        return float(weights @ matrix @ weights)

    result = minimize(
        variance,
        initial,
        method="SLSQP",
        bounds=Bounds(np.zeros(asset_count), np.full(asset_count, maximum_weight)),
        constraints=LinearConstraint(np.ones((1, asset_count)), 1.0, 1.0),
    )
    if not result.success:
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return pd.Series(result.x, index=covariance.index, name="weight")
