"""Combine development-stage strategy sleeves without performance cherry-picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.trend import construct_target_weights, simulate_with_drawdown_controls

BREAKOUT_FEATURES = (
    "close_percentile_20s",
    "close_percentile_60s",
    "close_percentile_120s",
    "range_percentile_60s",
    "volume_percentile_60s",
)


def build_breakout_forecast(features: pd.DataFrame) -> pd.DataFrame:
    """Create a bounded continuation score from price location and market activity."""
    missing = sorted(set(BREAKOUT_FEATURES) - set(features.columns))
    if missing:
        raise ValueError(f"missing breakout features: {missing}")
    price_location = features.loc[
        :, ["close_percentile_20s", "close_percentile_60s", "close_percentile_120s"]
    ].astype(float)
    direction = np.tanh((price_location.mean(axis=1, skipna=False) - 0.5) * 3.0)
    activity = (
        0.50
        + 0.25 * features["range_percentile_60s"].astype(float)
        + 0.25 * features["volume_percentile_60s"].astype(float)
    )
    result = features[["trading_date", "symbol"]].copy()
    result["forecast"] = (direction * activity).clip(-1.0, 1.0)
    return result


def group_mean_reversion_families(
    strategy_returns: pd.DataFrame, registry: pd.DataFrame
) -> pd.DataFrame:
    """Equal-weight variants inside each family so parameter counts cannot dominate."""
    required = {"name", "family"}
    if not required.issubset(registry.columns):
        raise ValueError(f"registry must contain columns: {sorted(required)}")
    output: dict[str, pd.Series] = {}
    for raw_family, rows in registry.groupby("family", sort=True):
        family = str(raw_family)
        members = [str(name) for name in rows["name"] if str(name) in strategy_returns.columns]
        if not members:
            raise ValueError(f"no return series found for mean-reversion family: {family}")
        output[f"session_mr_{family}"] = strategy_returns[members].mean(axis=1)
    return pd.DataFrame(output, index=strategy_returns.index)


def build_candidate_portfolios(
    sleeve_returns: pd.DataFrame,
    *,
    target_sleeve_volatility: float = 0.10,
    target_portfolio_volatility: float = 0.10,
    lookback: int = 60,
    minimum_history: int = 20,
) -> pd.DataFrame:
    """Build equal-weight and causal volatility-balanced combinations of net returns."""
    if sleeve_returns.empty:
        raise ValueError("at least one sleeve return series is required")
    returns = sleeve_returns.astype(float).sort_index()
    equal_weight = returns.mean(axis=1, skipna=True)

    prior_sleeve_volatility = returns.rolling(lookback, min_periods=minimum_history).std(
        ddof=1
    ).shift(1) * np.sqrt(252)
    sleeve_scale = (target_sleeve_volatility / prior_sleeve_volatility).clip(0.25, 2.0)
    sleeve_scale = sleeve_scale.where(prior_sleeve_volatility.gt(0.0), 1.0).fillna(1.0)
    risk_balanced = (returns * sleeve_scale).mean(axis=1, skipna=True)

    prior_portfolio_volatility = risk_balanced.rolling(lookback, min_periods=minimum_history).std(
        ddof=1
    ).shift(1) * np.sqrt(252)
    portfolio_scale = (target_portfolio_volatility / prior_portfolio_volatility).clip(0.25, 2.0)
    portfolio_scale = portfolio_scale.where(prior_portfolio_volatility.gt(0.0), 1.0).fillna(1.0)
    causal_volatility_balanced = risk_balanced * portfolio_scale
    return pd.DataFrame(
        {
            "equal_weight": equal_weight,
            "causal_volatility_balanced": causal_volatility_balanced,
        },
        index=returns.index,
    )


def classify_candidate(*, annualized_return: float, positive_folds: int) -> str:
    """Apply a permissive discovery gate; candidate does not mean live-ready."""
    if annualized_return > 0.0 and positive_folds >= 2:
        return "candidate"
    if annualized_return > 0.0 or positive_folds >= 2:
        return "research_further"
    return "reject"


def select_candidate_sleeves(
    sleeve_returns: pd.DataFrame, aggregate_summary: pd.DataFrame
) -> pd.DataFrame:
    """Fund only sleeves that pass the declared development candidate gate."""
    required = {"strategy", "research_status"}
    if not required.issubset(aggregate_summary.columns):
        raise ValueError(f"aggregate summary must contain columns: {sorted(required)}")
    candidate_names = set(
        aggregate_summary.loc[
            aggregate_summary["research_status"].eq("candidate"), "strategy"
        ].astype(str)
    )
    selected = [str(name) for name in sleeve_returns.columns if str(name) in candidate_names]
    if not selected:
        raise ValueError("no sleeves cleared the development candidate gate")
    return sleeve_returns.loc[:, selected].copy()


def _validation_mask(index: pd.DatetimeIndex, folds: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        mask.loc[start:end] = True
    return mask


def summarize_returns(
    returns: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize every sleeve on the same predefined outer validation folds."""
    details: list[dict[str, float | int | str]] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        for raw_name in returns.columns:
            name = str(raw_name)
            sample = returns.loc[start:end, name].dropna()
            details.append(
                {
                    "strategy": name,
                    "fold": int(fold["fold"]),
                    "sessions": len(sample),
                    "annualized_return": float(sample.mean() * 252),
                    "annualized_volatility": annualized_volatility(sample),
                    "sharpe": sharpe_ratio(sample),
                    "maximum_drawdown": maximum_drawdown(sample),
                }
            )
    detail = pd.DataFrame(details)
    mask = _validation_mask(pd.DatetimeIndex(returns.index), folds)
    aggregates: list[dict[str, float | int | str]] = []
    for raw_name in returns.columns:
        name = str(raw_name)
        sample = returns.loc[mask, name].dropna()
        folds_for_strategy = detail.loc[detail["strategy"].eq(name)]
        positive_folds = int((folds_for_strategy["annualized_return"] > 0.0).sum())
        annualized_return = float(sample.mean() * 252)
        aggregates.append(
            {
                "strategy": name,
                "sessions": len(sample),
                "annualized_return": annualized_return,
                "annualized_volatility": annualized_volatility(sample),
                "sharpe": sharpe_ratio(sample),
                "maximum_drawdown": maximum_drawdown(sample),
                "positive_folds": positive_folds,
                "worst_fold_sharpe": float(folds_for_strategy["sharpe"].min()),
                "research_status": classify_candidate(
                    annualized_return=annualized_return, positive_folds=positive_folds
                ),
            }
        )
    aggregate = pd.DataFrame(aggregates).sort_values("sharpe", ascending=False)
    return detail, aggregate


def _read_return(path: Path, column: str = "net_return") -> pd.Series:
    frame = pd.read_parquet(path)
    if column not in frame.columns:
        raise ValueError(f"{path} does not contain {column}")
    series = frame[column].astype(float).copy()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def run_candidate_portfolio(project_root: Path) -> None:
    """Build the development candidate portfolio from costed strategy artifacts."""
    portfolio_root = project_root / "data/processed/portfolios"
    output_root = project_root / "data/processed/candidate_portfolio"
    output_root.mkdir(parents=True, exist_ok=True)
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")

    features = pd.read_parquet(
        project_root / "data/processed/features/development_session_features.parquet"
    )
    panel = pd.read_parquet(
        project_root / "data/processed/research/development_session_panel.parquet"
    )
    market_returns = panel.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    market_returns.index = pd.to_datetime(market_returns.index)
    market_returns = market_returns.sort_index()
    breakout_forecast = build_breakout_forecast(features.sort_values(["symbol", "trading_date"]))
    breakout_weights = construct_target_weights(breakout_forecast, market_returns)
    breakout_returns, breakout_implemented = simulate_with_drawdown_controls(
        market_returns, breakout_weights, cost_bps=3.5
    )

    mean_reversion_returns = pd.read_parquet(
        portfolio_root / "development_mean_reversion_50_returns.parquet"
    )
    mean_reversion_returns.index = pd.to_datetime(mean_reversion_returns.index)
    mean_reversion_registry = pd.read_csv(portfolio_root / "mean_reversion_50_registry.csv")
    mean_reversion_families = group_mean_reversion_families(
        mean_reversion_returns, mean_reversion_registry
    )

    plain_trend = _read_return(portfolio_root / "development_trend_returns.parquet")
    regime_trend = _read_return(portfolio_root / "development_regime_trend_returns.parquet")
    multi_asset_trend = pd.concat([plain_trend, regime_trend], axis=1).mean(axis=1)
    ml_cl_trend = _read_return(portfolio_root / "ml_trend_screened_portfolio_returns.parquet")
    intraday_reversal = _read_return(
        portfolio_root / "intraday_mean_reversion_portfolio_daily.parquet"
    )

    index = pd.DatetimeIndex(market_returns.index)
    validation_mask = _validation_mask(index, folds)
    ml_cl_trend = ml_cl_trend.reindex(index).where(validation_mask)
    sleeves = pd.DataFrame(
        {
            "ml_cl_trend": ml_cl_trend,
            "multi_asset_trend": multi_asset_trend.reindex(index),
            "intraday_reversal": intraday_reversal.reindex(index),
            "breakout_continuation": breakout_returns.reindex(index),
        },
        index=index,
    )
    sleeves = sleeves.join(mean_reversion_families.reindex(index))
    sleeve_fold, sleeve_aggregate = summarize_returns(sleeves, folds)
    all_sleeve_portfolios = build_candidate_portfolios(sleeves).add_prefix("all_sleeves_")
    selected_sleeves = select_candidate_sleeves(sleeves, sleeve_aggregate)
    selected_portfolios = build_candidate_portfolios(selected_sleeves).add_prefix(
        "candidate_sleeves_"
    )
    portfolios = pd.concat([selected_portfolios, all_sleeve_portfolios], axis=1)
    portfolio_fold, portfolio_aggregate = summarize_returns(portfolios, folds)
    correlation = sleeves.loc[validation_mask].corr(min_periods=20)

    breakout_forecast.to_parquet(output_root / "breakout_forecasts.parquet", index=False)
    breakout_implemented.to_parquet(output_root / "breakout_implemented_weights.parquet")
    pd.DataFrame({"net_return": breakout_returns}).to_parquet(
        output_root / "breakout_returns.parquet"
    )
    sleeves.to_parquet(output_root / "sleeve_returns.parquet")
    portfolios.to_parquet(output_root / "portfolio_returns.parquet")
    sleeve_fold.to_csv(output_root / "sleeve_fold_summary.csv", index=False)
    sleeve_aggregate.to_csv(output_root / "sleeve_aggregate_summary.csv", index=False)
    portfolio_fold.to_csv(output_root / "portfolio_fold_summary.csv", index=False)
    portfolio_aggregate.to_csv(output_root / "portfolio_aggregate_summary.csv", index=False)
    correlation.to_csv(output_root / "validation_correlation.csv")
    registry = {
        "research_stage": "development_candidate",
        "candidate_is_live_ready": False,
        "selection_policy": (
            "positive aggregate return and at least two positive predefined outer folds"
        ),
        "allocation_policy": (
            "equal weight by distinct sleeve or causal trailing-volatility balance; "
            "mean-reversion variants first equal-weighted within family"
        ),
        "funded_candidate_sleeves": list(selected_sleeves.columns),
        "all_sleeves_combination_is_diagnostic_only": True,
        "execution": "all source sleeves net of their stated costs and delayed positions",
        "holdout_accessed": False,
        "excluded_rules": ["exact ZT/ZN displacement reversion trading implementation"],
    }
    (output_root / "research_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )

    print("\nCombined development candidates")
    print(portfolio_aggregate.to_string(index=False))
    print("\nUnderlying sleeves")
    print(sleeve_aggregate.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_candidate_portfolio(args.project_root.resolve())


if __name__ == "__main__":
    main()
