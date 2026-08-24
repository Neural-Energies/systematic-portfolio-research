"""Development-only, risk-controlled trend portfolio baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.portfolio import lag_positions, turnover

TREND_FEATURES = (
    "standardized_return_20s",
    "standardized_return_60s",
    "price_to_sma_20s",
    "price_to_sma_60s",
    "ema_gap_5_20s",
    "ema_gap_10_60s",
    "trend_slope_20s",
    "trend_slope_60s",
)


def build_trend_forecast(features: pd.DataFrame) -> pd.DataFrame:
    """Create a bounded, equal-weighted directional trend forecast."""
    missing = sorted(set(TREND_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"missing trend features: {missing}")
    values = features.loc[:, TREND_FEATURES].astype(float)
    scale = values.abs().groupby(features["symbol"]).rolling(60, min_periods=20).median()
    scale = scale.reset_index(level=0, drop=True).reindex(values.index).replace(0.0, np.nan)
    scaled = values.div(scale)
    bounded = pd.DataFrame(
        np.tanh(scaled.clip(-5.0, 5.0).to_numpy()), index=scaled.index, columns=scaled.columns
    )
    forecast = bounded.mean(axis=1, skipna=False)
    result = features[["trading_date", "symbol"]].copy()
    result["forecast"] = forecast
    return result


def _shrunk_covariance(history: pd.DataFrame, half_life: int = 20) -> pd.DataFrame:
    covariance = pd.DataFrame(
        history.ewm(halflife=half_life, min_periods=20).cov().loc[history.index[-1]]
    )
    diagonal = pd.DataFrame(
        np.diag(np.diag(covariance)), index=covariance.index, columns=covariance.columns
    )
    return pd.DataFrame(covariance * 0.8 + diagonal * 0.2)


def _balance_long_short(scores: pd.Series, net_limit: float) -> pd.Series:
    working = scores.copy()
    if (working > 0.0).sum() == 0 or (working < 0.0).sum() == 0:
        # Keep both books active when every absolute forecast has the same sign.
        working = working - working.median()
    long = working.clip(lower=0.0)
    short = -working.clip(upper=0.0)
    if long.sum() <= 0.0 or short.sum() <= 0.0:
        return pd.Series(0.0, index=scores.index)
    raw_net = float((long.sum() - short.sum()) / (long.sum() + short.sum()))
    target_net = float(np.clip(raw_net, -net_limit, net_limit))
    long_budget = (1.0 + target_net) / 2.0
    short_budget = (1.0 - target_net) / 2.0
    weights = long_budget * long / long.sum() - short_budget * short / short.sum()
    if abs(float(weights.sum())) > net_limit + 1e-12:
        raise RuntimeError("net exposure constraint failed")
    return pd.Series(weights, index=scores.index, dtype=float)


def construct_target_weights(
    forecasts: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    target_volatility: float = 0.10,
    maximum_gross: float = 2.0,
    net_limit: float = 0.25,
    lookback: int = 60,
    half_life: int = 20,
    minimum_assets: int = 3,
) -> pd.DataFrame:
    """Size trend forecasts with trailing shrunk covariance and mandate constraints."""
    forecast_matrix = forecasts.pivot(index="trading_date", columns="symbol", values="forecast")
    forecast_matrix.index = pd.to_datetime(forecast_matrix.index)
    forecast_matrix = forecast_matrix.reindex(returns.index)
    output = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for position in range(lookback - 1, len(returns)):
        date = returns.index[position]
        history = returns.iloc[position - lookback + 1 : position + 1].dropna()
        scores = forecast_matrix.loc[date].dropna()
        available = scores.index.intersection(history.columns)
        if len(available) < minimum_assets:
            continue
        covariance = _shrunk_covariance(history[available], half_life)
        if len(available) == 1:
            variance = float(cast(Any, covariance.iloc[0, 0]))
            asset_volatility = float(np.sqrt(variance) * np.sqrt(252))
            if asset_volatility > 0.0:
                exposure = min(target_volatility / asset_volatility, maximum_gross, net_limit)
                score = float(cast(Any, scores.iloc[0]))
                output.loc[date, available[0]] = float(np.sign(score) * exposure)
            continue
        volatility = np.sqrt(np.diag(covariance)).clip(min=1e-8)
        risk_scores = scores[available] / volatility
        weights = _balance_long_short(risk_scores, net_limit)
        vector = weights.to_numpy(dtype=float)
        matrix = covariance.to_numpy(dtype=float)
        portfolio_volatility = float(np.sqrt(vector @ matrix @ vector) * np.sqrt(252))
        if portfolio_volatility <= 0.0:
            continue
        leverage = min(target_volatility / portfolio_volatility, maximum_gross)
        scaled = weights * leverage
        scaled_net = float(scaled.sum())
        if abs(scaled_net) > net_limit:
            gross = float(scaled.abs().sum())
            target_net = float(np.clip(scaled_net, -net_limit, net_limit))
            positive = scaled.clip(lower=0.0)
            negative = -scaled.clip(upper=0.0)
            scaled = (
                positive / positive.sum() * (gross + target_net) / 2.0
                - negative / negative.sum() * (gross - target_net) / 2.0
            )
        output.loc[date, available] = scaled
    return output


def simulate_with_drawdown_controls(
    returns: pd.DataFrame, target_weights: pd.DataFrame, *, cost_bps: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Lag positions, charge costs, and apply sequential emergency drawdown scaling."""
    implemented = lag_positions(target_weights, 1)
    controlled = implemented.copy()
    net_returns: list[float] = []
    wealth = peak = 1.0
    previous = pd.Series(0.0, index=implemented.columns)
    for raw_date, row in implemented.iterrows():
        date = pd.Timestamp(cast(Any, raw_date))
        drawdown = 1.0 - wealth / peak
        if drawdown >= 0.15:
            scale = 0.15
        elif drawdown >= 0.10:
            scale = 0.50
        elif drawdown >= 0.075:
            scale = 0.75
        else:
            scale = 1.0
        current = row * scale
        controlled.loc[date] = current
        trade = float((current - previous).abs().sum() / 2.0)
        result = float((current * returns.loc[date]).sum() - trade * cost_bps / 10_000.0)
        net_returns.append(result)
        wealth *= 1.0 + result
        peak = max(peak, wealth)
        previous = current
    return pd.Series(net_returns, index=implemented.index, name="net_return"), controlled


def apply_regime_risk_overlay(
    target_weights: pd.DataFrame, regime_probabilities: pd.DataFrame
) -> pd.DataFrame:
    """Adjust gross risk gently; uncertain regime estimates revert toward no adjustment."""
    regime = regime_probabilities.copy()
    regime["trading_date"] = pd.to_datetime(regime["trading_date"])
    regime = regime.set_index("trading_date").reindex(target_weights.index)
    trend_probability = (
        regime["ensemble_low_volatility_trend"] + regime["ensemble_high_volatility_trend"]
    )
    raw_multiplier = (
        0.65 + 0.70 * trend_probability - 0.35 * regime["ensemble_stress_transition"]
    ).clip(0.35, 1.15)
    confidence = regime["ensemble_confidence"].fillna(0.0).clip(0.0, 1.0)
    multiplier = 1.0 + confidence * (raw_multiplier - 1.0)
    return target_weights.mul(multiplier.fillna(1.0), axis=0)


def summarize_folds(net: pd.Series, weights: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Report only the predefined walk-forward validation windows."""
    rows: list[dict[str, float | int]] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        sample = net.loc[start:end]
        sample_weights = weights.loc[start:end]
        rows.append(
            {
                "fold": int(fold["fold"]),
                "sessions": len(sample),
                "annualized_return": float(sample.mean() * 252),
                "annualized_volatility": annualized_volatility(sample),
                "sharpe": sharpe_ratio(sample),
                "maximum_drawdown": maximum_drawdown(sample),
                "average_gross": float(sample_weights.abs().sum(axis=1).mean()),
                "average_net": float(sample_weights.sum(axis=1).mean()),
                "average_daily_turnover": float(turnover(sample_weights).mean()),
            }
        )
    return pd.DataFrame(rows)


def run_baseline(project_root: Path) -> None:
    """Build and save the development-only trend baseline artifacts."""
    feature_path = project_root / "data/processed/features/development_session_features.parquet"
    panel_path = project_root / "data/processed/research/development_session_panel.parquet"
    regime_path = project_root / "data/processed/regimes/development_regime_probabilities.parquet"
    folds_path = project_root / "outputs/development_walk_forward_folds.csv"
    output = project_root / "data/processed/portfolios"
    output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(feature_path)
    panel = pd.read_parquet(panel_path)
    returns = panel.pivot(index="trading_date", columns="symbol", values="close_to_close_return")
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    forecast = build_trend_forecast(features.sort_values(["symbol", "trading_date"]))
    weights = construct_target_weights(forecast, returns)
    net, implemented = simulate_with_drawdown_controls(returns, weights, cost_bps=3.5)
    regime_weights = apply_regime_risk_overlay(weights, pd.read_parquet(regime_path))
    regime_net, regime_implemented = simulate_with_drawdown_controls(
        returns, regime_weights, cost_bps=3.5
    )
    folds = pd.read_csv(folds_path)
    plain_summary = summarize_folds(net, implemented, folds).assign(strategy="plain_trend")
    regime_summary = summarize_folds(regime_net, regime_implemented, folds).assign(
        strategy="regime_aware_trend"
    )
    summary = pd.concat([plain_summary, regime_summary], ignore_index=True)
    forecast.to_parquet(output / "development_trend_forecasts.parquet", index=False)
    weights.to_parquet(output / "development_trend_target_weights.parquet")
    pd.DataFrame({"net_return": net}).to_parquet(output / "development_trend_returns.parquet")
    regime_weights.to_parquet(output / "development_regime_trend_target_weights.parquet")
    pd.DataFrame({"net_return": regime_net}).to_parquet(
        output / "development_regime_trend_returns.parquet"
    )
    summary.to_csv(output / "development_trend_walk_forward_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_baseline(args.project_root.resolve())


if __name__ == "__main__":
    main()
