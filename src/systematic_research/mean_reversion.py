"""Pre-specified development-only mean-reversion strategy tournament."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.trend import _shrunk_covariance, simulate_with_drawdown_controls


@dataclass(frozen=True)
class StrategySpec:
    """A single, pre-declared mean-reversion rule."""

    name: str
    family: str
    feature: str
    threshold: float = 0.0
    cross_sectional_rank: bool = False


def strategy_specs() -> list[StrategySpec]:
    """Return exactly 50 hypotheses grouped into five literature-motivated families."""
    return_reversal = [
        "log_return_1s",
        "momentum_2s",
        "momentum_5s",
        "momentum_10s",
        "momentum_20s",
        "standardized_return_5s",
        "standardized_return_20s",
        "standardized_return_60s",
        "close_return_lag_2s",
        "close_return_lag_5s",
    ]
    moving_average = [
        "price_to_sma_5s",
        "price_to_sma_10s",
        "price_to_sma_20s",
        "price_to_sma_30s",
        "price_to_sma_60s",
        "price_to_sma_120s",
        "sma_gap_5_20s",
        "sma_gap_20_60s",
        "sma_gap_30_120s",
        "price_to_ema_30s",
    ]
    exhaustion = [
        "close_location",
        "overnight_gap_fraction",
        "intraday_tail_imbalance",
        "return_sign_imbalance",
        "range_zscore_20s",
        "range_percentile_20s",
        "range_percentile_60s",
        "volume_zscore_20s",
        "trend_slope_acceleration_20s",
        "upside_downside_volatility_ratio_20s",
    ]
    specs = [
        StrategySpec(f"return_reversal_{i:02d}", "return_reversal", feature)
        for i, feature in enumerate(return_reversal, 1)
    ]
    specs += [
        StrategySpec(f"moving_average_{i:02d}", "moving_average_deviation", feature)
        for i, feature in enumerate(moving_average, 1)
    ]
    for feature in ("trend_residual_zscore_20s", "trend_residual_zscore_60s"):
        for threshold in (0.0, 0.5, 1.0, 1.5, 2.0):
            specs.append(
                StrategySpec(
                    f"residual_{feature[-3:]}_t{threshold:.1f}",
                    "regression_residual",
                    feature,
                    threshold,
                )
            )
    specs += [
        StrategySpec(f"exhaustion_{i:02d}", "range_exhaustion", feature)
        for i, feature in enumerate(exhaustion, 1)
    ]
    for feature in ("log_return_1s", "momentum_2s", "momentum_5s", "momentum_10s", "momentum_20s"):
        for ranked in (False, True):
            suffix = "rank" if ranked else "scaled"
            specs.append(
                StrategySpec(
                    f"cross_sectional_{feature}_{suffix}",
                    "cross_sectional_contrarian",
                    feature,
                    cross_sectional_rank=ranked,
                )
            )
    if len(specs) != 50:
        raise RuntimeError("strategy registry must contain exactly 50 specifications")
    return specs


def build_signal(features: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    """Build a causal contrarian score; positive means long."""
    if spec.feature not in features:
        raise ValueError(f"missing feature: {spec.feature}")
    panel = features.pivot(index="trading_date", columns="symbol", values=spec.feature).astype(
        float
    )
    panel.index = pd.to_datetime(panel.index)
    if spec.feature in {"close_location", "range_percentile_20s", "range_percentile_60s"}:
        panel = panel - 0.5
    elif spec.feature == "upside_downside_volatility_ratio_20s":
        panel = pd.DataFrame(
            np.log(panel.clip(lower=1e-8).to_numpy()), index=panel.index, columns=panel.columns
        )
    if spec.cross_sectional_rank:
        panel = panel.rank(axis=1, pct=True) - 0.5
    else:
        scale = panel.abs().expanding(min_periods=60).median().shift(1).replace(0.0, np.nan)
        panel = panel.div(scale)
    signal = pd.DataFrame(
        -np.tanh(panel.clip(-5.0, 5.0).to_numpy()), index=panel.index, columns=panel.columns
    )
    if spec.threshold > 0.0:
        signal = signal.where(panel.abs() >= spec.threshold, 0.0)
    return pd.DataFrame(signal, index=panel.index, columns=panel.columns)


def construct_weights(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Convert scores to market-neutral, covariance-targeted weights."""
    signal = signal.reindex(index=returns.index, columns=returns.columns)
    output = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for position in range(59, len(returns)):
        date = returns.index[position]
        history = returns.iloc[position - 59 : position + 1].dropna()
        score = signal.loc[date].dropna()
        available = score.index.intersection(history.columns)
        if len(available) < 3:
            continue
        score = score[available] - score[available].median()
        long, short = score.clip(lower=0.0), -score.clip(upper=0.0)
        if long.sum() <= 0.0 or short.sum() <= 0.0:
            continue
        weights = 0.5 * long / long.sum() - 0.5 * short / short.sum()
        covariance = _shrunk_covariance(history[available])
        vector, matrix = weights.to_numpy(float), covariance.to_numpy(float)
        volatility = float(np.sqrt(vector @ matrix @ vector) * np.sqrt(252))
        leverage = min(0.10 / volatility, 2.0) if volatility > 0.0 else 0.0
        output.loc[date, available] = weights * leverage
    return output


def evaluate(
    strategy_returns: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate predefined fold metrics and transparent stability diagnostics."""
    detail: list[dict[str, Any]] = []
    validation_mask = pd.Series(False, index=strategy_returns.index)
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        start, end = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"])
        validation_mask.loc[start:end] = True
        for raw_name in strategy_returns.columns:
            name = str(raw_name)
            sample = pd.Series(strategy_returns[name]).loc[start:end]
            detail.append(
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
    detail_frame = pd.DataFrame(detail)
    rows: list[dict[str, Any]] = []
    for raw_name in strategy_returns.columns:
        name = str(raw_name)
        sample = pd.Series(strategy_returns[name]).loc[validation_mask]
        fold_rows = detail_frame.loc[detail_frame["strategy"] == name]
        rows.append(
            {
                "strategy": name,
                "sessions": len(sample),
                "annualized_return": float(sample.mean() * 252),
                "annualized_volatility": annualized_volatility(sample),
                "sharpe": sharpe_ratio(sample),
                "maximum_drawdown": maximum_drawdown(sample),
                "positive_folds": int((fold_rows["annualized_return"] > 0.0).sum()),
                "worst_fold_sharpe": float(fold_rows["sharpe"].min()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["stable_diagnostic_lead"] = (
        (aggregate["sharpe"] >= 0.75)
        & (aggregate["positive_folds"] >= 3)
        & (aggregate["worst_fold_sharpe"] > -0.50)
    )
    return detail_frame, aggregate.sort_values("sharpe", ascending=False)


def run_tournament(project_root: Path) -> None:
    """Run all 50 specifications without accessing the sealed holdout."""
    features = pd.read_parquet(
        project_root / "data/processed/features/development_session_features.parquet"
    )
    panel = pd.read_parquet(
        project_root / "data/processed/research/development_session_panel.parquet"
    )
    returns = panel.pivot(index="trading_date", columns="symbol", values="close_to_close_return")
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    result: dict[str, pd.Series] = {}
    registry = strategy_specs()
    for spec in registry:
        weights = construct_weights(build_signal(features, spec), returns)
        net, _ = simulate_with_drawdown_controls(returns, weights, cost_bps=3.5)
        result[spec.name] = net
    strategy_returns = pd.DataFrame(result)
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    detail, aggregate = evaluate(strategy_returns, folds)
    output = project_root / "data/processed/portfolios"
    output.mkdir(parents=True, exist_ok=True)
    strategy_returns.to_parquet(output / "development_mean_reversion_50_returns.parquet")
    pd.DataFrame([spec.__dict__ for spec in registry]).to_csv(
        output / "mean_reversion_50_registry.csv", index=False
    )
    detail.to_csv(output / "mean_reversion_50_fold_results.csv", index=False)
    aggregate.to_csv(output / "mean_reversion_50_summary.csv", index=False)
    print(aggregate.head(15).to_string(index=False))
    print(
        "\nStable diagnostic leads: "
        f"{int(aggregate['stable_diagnostic_lead'].sum())}; portfolio testing still required"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run_tournament(args.project_root.resolve())


if __name__ == "__main__":
    main()
