"""Broad development-only strategy search and portfolio backtest."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.candidate_portfolio import build_candidate_portfolios
from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.portfolio import turnover
from systematic_research.trend import construct_target_weights, simulate_with_drawdown_controls


@dataclass(frozen=True)
class StrategySpec:
    """One transparent strategy variant in the broad search."""

    name: str
    theme: str
    feature: str
    direction: float = 1.0
    hold_sessions: int = 1
    cross_sectional: bool = False
    secondary_feature: str | None = None
    secondary_mode: str | None = None


def strategy_specs() -> list[StrategySpec]:
    """Return ten variants for each of six portfolio strategy themes."""
    momentum_features = [
        ("log_return_1s", 1),
        ("momentum_2s", 1),
        ("momentum_5s", 1),
        ("momentum_10s", 2),
        ("momentum_20s", 3),
        ("momentum_60s", 5),
        ("standardized_return_5s", 1),
        ("standardized_return_20s", 2),
        ("standardized_return_60s", 3),
        ("return_zscore_20s", 1),
    ]
    trend_features = [
        ("price_to_sma_10s", 1),
        ("price_to_sma_20s", 2),
        ("price_to_sma_60s", 3),
        ("price_to_sma_120s", 5),
        ("ema_gap_5_20s", 1),
        ("ema_gap_10_60s", 3),
        ("trend_slope_20s", 1),
        ("trend_slope_30s", 2),
        ("trend_slope_60s", 3),
        ("trend_slope_120s", 5),
    ]
    reversion_features = [
        ("log_return_1s", 1),
        ("momentum_2s", 1),
        ("momentum_5s", 2),
        ("momentum_10s", 3),
        ("standardized_return_5s", 1),
        ("standardized_return_20s", 2),
        ("price_to_sma_5s", 1),
        ("price_to_sma_20s", 3),
        ("trend_residual_zscore_20s", 1),
        ("trend_residual_zscore_60s", 3),
    ]
    specs = [
        StrategySpec(f"momentum_{i:02d}", "momentum", feature, hold_sessions=hold)
        for i, (feature, hold) in enumerate(momentum_features, 1)
    ]
    specs += [
        StrategySpec(f"trend_{i:02d}", "trend", feature, hold_sessions=hold)
        for i, (feature, hold) in enumerate(trend_features, 1)
    ]
    specs += [
        StrategySpec(
            f"mean_reversion_{i:02d}",
            "mean_reversion",
            feature,
            direction=-1.0,
            hold_sessions=hold,
        )
        for i, (feature, hold) in enumerate(reversion_features, 1)
    ]

    breakout_inputs = [
        ("close_percentile_20s", "range_percentile_20s", 1),
        ("close_percentile_20s", "volume_percentile_20s", 1),
        ("close_percentile_60s", "range_percentile_60s", 2),
        ("close_percentile_60s", "volume_percentile_60s", 2),
        ("close_percentile_120s", "range_percentile_120s", 3),
        ("close_percentile_120s", "volume_percentile_120s", 3),
        ("close_percentile_20s", "volume_zscore_20s", 1),
        ("close_percentile_60s", "range_zscore_20s", 2),
        ("price_to_sma_20s", "range_percentile_60s", 2),
        ("price_to_sma_60s", "volume_percentile_60s", 3),
    ]
    specs += [
        StrategySpec(
            f"breakout_{i:02d}",
            "breakout",
            feature,
            hold_sessions=hold,
            secondary_feature=secondary,
            secondary_mode="activity",
        )
        for i, (feature, secondary, hold) in enumerate(breakout_inputs, 1)
    ]

    relative_value_inputs = [
        ("log_return_1s", -1.0, 1),
        ("momentum_2s", -1.0, 1),
        ("momentum_5s", -1.0, 2),
        ("momentum_10s", 1.0, 2),
        ("momentum_20s", 1.0, 3),
        ("momentum_60s", 1.0, 5),
        ("price_to_sma_20s", -1.0, 2),
        ("price_to_sma_60s", 1.0, 3),
        ("close_percentile_20s", -1.0, 1),
        ("close_percentile_60s", 1.0, 3),
    ]
    specs += [
        StrategySpec(
            f"relative_value_{i:02d}",
            "relative_value",
            feature,
            direction=direction,
            hold_sessions=hold,
            cross_sectional=True,
        )
        for i, (feature, direction, hold) in enumerate(relative_value_inputs, 1)
    ]

    lead_lag_inputs = [
        ("peer_return_lag1s", None, None, 1.0, 1),
        ("peer_momentum_5s_lag1s", None, None, 1.0, 2),
        ("peer_breadth_lag1s", None, None, 1.0, 1),
        ("peer_return_lag1s", "leadlag_correlation_20s", "signed", 1.0, 1),
        ("peer_return_lag1s", "leadlag_correlation_60s", "signed", 1.0, 2),
        ("peer_return_lag1s", "leadlag_beta_60s", "signed", 1.0, 2),
        ("peer_momentum_5s_lag1s", "leadlag_correlation_20s", "signed", 1.0, 2),
        ("peer_momentum_5s_lag1s", "leadlag_correlation_60s", "signed", 1.0, 3),
        ("peer_return_lag1s", "leadlag_correlation_20s", "signed", -1.0, 1),
        ("peer_momentum_5s_lag1s", "leadlag_beta_60s", "signed", -1.0, 3),
    ]
    specs += [
        StrategySpec(
            f"lead_lag_{i:02d}",
            "lead_lag",
            feature,
            direction=direction,
            hold_sessions=hold,
            secondary_feature=secondary,
            secondary_mode=mode,
        )
        for i, (feature, secondary, mode, direction, hold) in enumerate(lead_lag_inputs, 1)
    ]
    if len(specs) != 60:
        raise RuntimeError("broad registry must contain exactly 60 strategies")
    return specs


def _feature_panel(features: pd.DataFrame, name: str) -> pd.DataFrame:
    if name not in features.columns:
        raise ValueError(f"missing feature: {name}")
    panel = features.pivot(index="trading_date", columns="symbol", values=name).astype(float)
    panel.index = pd.to_datetime(panel.index)
    if "percentile" in name or name == "close_location":
        panel = panel - 0.5
    return panel.sort_index()


def build_signal(features: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Build a bounded causal signal; implementation is delayed in the simulator."""
    raw = _feature_panel(features, spec.feature)
    if spec.cross_sectional:
        raw = raw.rank(axis=1, pct=True) - 0.5
    else:
        scale = raw.abs().expanding(min_periods=60).median().shift(1).replace(0.0, np.nan)
        raw = raw.div(scale)
    if spec.secondary_feature is not None:
        secondary = _feature_panel(features, spec.secondary_feature).reindex_like(raw)
        if spec.secondary_mode == "activity":
            if "percentile" in spec.secondary_feature:
                secondary = (secondary + 0.5).clip(0.0, 1.0)
            else:
                secondary_scale = (
                    secondary.abs().expanding(min_periods=60).median().shift(1).replace(0.0, np.nan)
                )
                secondary = secondary.div(secondary_scale).abs().clip(0.0, 2.0) / 2.0
            raw = raw * (0.5 + 0.5 * secondary)
        elif spec.secondary_mode == "signed":
            raw = raw * secondary.clip(-2.0, 2.0)
    bounded = pd.DataFrame(
        np.tanh(raw.clip(-5.0, 5.0).to_numpy()) * spec.direction,
        index=raw.index,
        columns=raw.columns,
    )
    if spec.hold_sessions > 1:
        bounded = bounded.rolling(spec.hold_sessions, min_periods=1).mean()
    return bounded


def classify_research_tier(sharpe: float, positive_folds: int) -> str:
    """Label every result without using a hard rejection gate."""
    if sharpe >= 1.5 and positive_folds >= 3:
        return "A_promising"
    if sharpe >= 0.75 and positive_folds >= 2:
        return "B_candidate"
    if sharpe > 0.0:
        return "C_exploratory_positive"
    return "D_exploratory_negative"


def _validation_mask(index: pd.DatetimeIndex, folds: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        mask.loc[pd.Timestamp(fold["validation_start"]) : pd.Timestamp(fold["validation_end"])] = (
            True
        )
    return mask


def summarize_results(
    returns: pd.DataFrame,
    weights: dict[str, pd.DataFrame],
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report aggregate and fold evidence for every strategy or portfolio."""
    details: list[dict[str, float | int | str]] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        start, end = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"])
        for name in returns.columns.astype(str):
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
    rows: list[dict[str, float | int | str]] = []
    for name in returns.columns.astype(str):
        sample = returns.loc[mask, name].dropna()
        fold_rows = detail.loc[detail["strategy"].eq(name)]
        positive_folds = int((fold_rows["annualized_return"] > 0.0).sum())
        strategy_weights = weights.get(name)
        if strategy_weights is None:
            average_turnover = np.nan
            active_rebalances_per_week = np.nan
        else:
            validation_weights = strategy_weights.loc[mask]
            daily_turnover = turnover(validation_weights)
            average_turnover = float(daily_turnover.mean())
            weeks = max((sample.index.max() - sample.index.min()).days / 7.0, 1.0)
            active_rebalances_per_week = float(daily_turnover.gt(0.01).sum() / weeks)
        sharpe = sharpe_ratio(sample)
        rows.append(
            {
                "strategy": name,
                "sessions": len(sample),
                "annualized_return": float(sample.mean() * 252),
                "annualized_volatility": annualized_volatility(sample),
                "sharpe": sharpe,
                "maximum_drawdown": maximum_drawdown(sample),
                "positive_folds": positive_folds,
                "worst_fold_sharpe": float(fold_rows["sharpe"].min()),
                "average_daily_turnover": average_turnover,
                "active_rebalances_per_week": active_rebalances_per_week,
                "research_tier": classify_research_tier(sharpe, positive_folds),
            }
        )
    aggregate = pd.DataFrame(rows).sort_values("sharpe", ascending=False, ignore_index=True)
    return detail, aggregate


def _write_quantstats_report(returns: pd.Series, output: Path, title: str) -> str:
    try:
        import quantstats as qs

        qs.reports.html(  # type: ignore[no-untyped-call]
            returns.dropna(), output=str(output), title=title
        )
    except Exception as error:  # pragma: no cover - third-party rendering fallback
        return f"QuantStats report failed: {type(error).__name__}: {error}"
    return "created"


def run_search(
    project_root: Path,
    *,
    cost_bps: float = 3.5,
    top_per_theme: int = 3,
) -> None:
    """Run all strategy variants and build inspectable development portfolios."""
    features = pd.read_parquet(
        project_root / "data/processed/databento_features/development_session_features.parquet"
    )
    panel = pd.read_parquet(
        project_root / "data/processed/databento_research/development_session_panel.parquet"
    )
    market_returns = panel.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    market_returns.index = pd.to_datetime(market_returns.index)
    market_returns = market_returns.sort_index()
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")

    strategy_return_map: dict[str, pd.Series] = {}
    implemented_weights: dict[str, pd.DataFrame] = {}
    registry = strategy_specs()
    for spec in registry:
        signal = build_signal(features, spec)
        stacked = cast(
            pd.Series,
            signal.rename_axis(index="trading_date", columns="symbol").stack(future_stack=True),
        )
        forecast = stacked.rename("forecast").reset_index()
        target = construct_target_weights(forecast, market_returns, minimum_assets=3)
        net, implemented = simulate_with_drawdown_controls(
            market_returns, target, cost_bps=cost_bps
        )
        strategy_return_map[spec.name] = net
        implemented_weights[spec.name] = implemented
    strategy_returns = pd.DataFrame(strategy_return_map, index=market_returns.index)
    strategy_fold, strategy_summary = summarize_results(
        strategy_returns, implemented_weights, folds
    )
    registry_frame = pd.DataFrame([asdict(spec) for spec in registry])
    strategy_summary = strategy_summary.merge(
        registry_frame[["name", "theme", "feature", "hold_sessions"]],
        left_on="strategy",
        right_on="name",
        validate="one_to_one",
    ).drop(columns="name")

    theme_members: dict[str, list[str]] = {}
    for raw_theme, rows in registry_frame.groupby("theme", sort=True):
        theme_members[str(raw_theme)] = rows["name"].astype(str).tolist()
    theme_returns = pd.DataFrame(
        {
            str(theme): strategy_returns[[str(name) for name in names]].mean(axis=1)
            for theme, names in theme_members.items()
        },
        index=strategy_returns.index,
    )
    top_names: list[str] = []
    champion_names: list[str] = []
    for theme in sorted(theme_members):
        ranked = strategy_summary.loc[strategy_summary["theme"].eq(theme)]
        top_names.extend(ranked.head(top_per_theme)["strategy"].astype(str))
        champion_names.append(str(ranked.iloc[0]["strategy"]))
    positive_names = (
        strategy_summary.loc[strategy_summary["annualized_return"].gt(0.0), "strategy"]
        .astype(str)
        .tolist()
    )
    if not positive_names:
        positive_names = champion_names

    theme_allocations = build_candidate_portfolios(theme_returns).add_prefix("theme_")
    portfolios = pd.DataFrame(
        {
            "all_60_equal_weight": strategy_returns.mean(axis=1),
            "six_theme_equal_weight": theme_returns.mean(axis=1),
            "theme_champions_equal_weight": strategy_returns[champion_names].mean(axis=1),
            "top_3_per_theme_equal_weight": strategy_returns[top_names].mean(axis=1),
            "all_positive_equal_weight": strategy_returns[positive_names].mean(axis=1),
        },
        index=strategy_returns.index,
    ).join(theme_allocations)
    portfolio_fold, portfolio_summary = summarize_results(portfolios, {}, folds)

    output_root = project_root / "data/processed/broad_search"
    output_root.mkdir(parents=True, exist_ok=True)
    strategy_returns.to_parquet(output_root / "strategy_returns.parquet")
    theme_returns.to_parquet(output_root / "theme_returns.parquet")
    portfolios.to_parquet(output_root / "portfolio_returns.parquet")
    registry_frame.to_csv(output_root / "strategy_registry.csv", index=False)
    strategy_fold.to_csv(output_root / "strategy_fold_summary.csv", index=False)
    strategy_summary.to_csv(output_root / "strategy_summary.csv", index=False)
    portfolio_fold.to_csv(output_root / "portfolio_fold_summary.csv", index=False)
    portfolio_summary.to_csv(output_root / "portfolio_summary.csv", index=False)
    validation_index = pd.DatetimeIndex(strategy_returns.index)
    strategy_returns.loc[_validation_mask(validation_index, folds)].corr().to_csv(
        output_root / "strategy_correlation.csv"
    )

    best_portfolio = str(portfolio_summary.iloc[0]["strategy"])
    report_path = project_root / "outputs/broad_strategy_search_tear_sheet.html"
    report_status = _write_quantstats_report(
        portfolios[best_portfolio], report_path, f"Broad Strategy Search: {best_portfolio}"
    )
    manifest = {
        "research_stage": "development_only",
        "holdout_accessed": False,
        "strategies_tested": len(registry),
        "themes": sorted(theme_members),
        "cost_bps_one_way": cost_bps,
        "execution_delay_sessions": 1,
        "holding_period_sessions": "1 to 5 overlapping daily cohorts",
        "selection_note": (
            "All results are retained. Tiers are descriptive; top portfolios use development "
            "ranking and therefore require later frozen sequential evaluation."
        ),
        "top_per_theme": top_per_theme,
        "theme_champions": champion_names,
        "best_development_portfolio": best_portfolio,
        "quantstats_report": str(report_path),
        "quantstats_status": report_status,
    }
    (output_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTop strategies")
    print(strategy_summary.head(20).to_string(index=False))
    print("\nDevelopment portfolios")
    print(portfolio_summary.to_string(index=False))
    print(f"\nQuantStats: {report_status}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the broad development strategy search.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--top-per-theme", type=int, default=3)
    arguments = parser.parse_args()
    run_search(
        arguments.project_root.resolve(),
        cost_bps=arguments.cost_bps,
        top_per_theme=arguments.top_per_theme,
    )


if __name__ == "__main__":
    main()
