"""Fast development-only intraday search for leading strategy families."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numba import njit  # type: ignore[import-untyped]

from systematic_research.broad_strategy_search import _write_quantstats_report
from systematic_research.candidate_portfolio import summarize_returns
from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio

Theme = Literal["mean_reversion", "relative_value", "lead_lag"]


@dataclass(frozen=True)
class IntradaySpec:
    """A fully declared intraday strategy trial."""

    name: str
    theme: Theme
    timeframe_minutes: int
    lookback_bars: int
    entry_threshold: float
    exit_threshold: float
    maximum_hold_bars: int
    volatility_filter: str
    time_rule: str
    stop_loss: float | None


def strategy_specs() -> list[IntradaySpec]:
    """Create a focused, broad grid without silently rejecting any trial."""
    specifications: list[IntradaySpec] = []
    variants = [
        *[
            (entry, 0.10, hold, "all", "all", None)
            for entry in (0.75, 1.25)
            for hold in (4, 8, 16, 24)
        ],
        (1.00, 0.10, 16, "below_median", "all", None),
        (1.00, 0.10, 16, "above_median", "all", None),
        (1.00, 0.10, 16, "all", "us_core", None),
        (1.00, 0.10, 16, "all", "overnight", None),
        (1.00, 0.10, 16, "all", "all", 0.005),
        (1.00, 0.10, 16, "all", "all", 0.010),
        (1.00, 0.25, 16, "all", "all", None),
        (1.00, 0.40, 16, "all", "all", None),
    ]
    counter = 0
    themes: tuple[Theme, ...] = ("mean_reversion", "relative_value", "lead_lag")
    for timeframe in (15, 30, 60, 240):
        for theme in themes:
            for lookback in (2, 4, 8):
                for entry, exit_, hold, vol_filter, time_rule, stop in variants:
                    counter += 1
                    specifications.append(
                        IntradaySpec(
                            name=f"intraday_{counter:04d}",
                            theme=theme,
                            timeframe_minutes=timeframe,
                            lookback_bars=lookback,
                            entry_threshold=entry,
                            exit_threshold=exit_,
                            maximum_hold_bars=hold,
                            volatility_filter=vol_filter,
                            time_rule=time_rule,
                            stop_loss=stop,
                        )
                    )
    return specifications


def resample_bars(frame: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    """Aggregate minutes on deterministic UTC buckets inside each trading date.

    The bucket label is derived from the clock, never from the last observation a
    product happened to print.  This keeps different products on the same research
    clock without inventing executable bars for a product that did not trade.
    """
    if timeframe_minutes not in {15, 30, 60, 240}:
        raise ValueError("timeframe_minutes must be 15, 30, 60, or 240")
    data = frame.sort_values("timestamp_utc", ignore_index=True).copy()
    timestamp = pd.to_datetime(data["timestamp_utc"], utc=True)
    trading_midnight = pd.to_datetime(data["trading_date"]).dt.tz_localize("America/New_York")
    session_anchor = (trading_midnight - pd.Timedelta(hours=6)).dt.tz_convert("UTC")
    elapsed_minutes = (timestamp - session_anchor).dt.total_seconds().div(60.0)
    data["session_bucket"] = np.floor(elapsed_minutes / timeframe_minutes).astype(int)
    data["bar_start_utc"] = session_anchor + pd.to_timedelta(
        data["session_bucket"] * timeframe_minutes, unit="min"
    )
    data["timestamp_utc"] = timestamp
    grouped = data.groupby(
        ["symbol", "trading_date", "session_bucket", "bar_start_utc"], sort=False
    )
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_count=("close", "size"),
    ).reset_index()
    bars["timestamp_utc"] = bars.pop("bar_start_utc")
    bars["trading_date"] = pd.to_datetime(bars["trading_date"])
    bars["return"] = bars.groupby(["symbol", "trading_date"], sort=False)["close"].pct_change()
    return bars.sort_values(["timestamp_utc", "symbol"], ignore_index=True)


def _panels(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    close = bars.pivot(index="timestamp_utc", columns="symbol", values="close").sort_index()
    returns = bars.pivot(index="timestamp_utc", columns="symbol", values="return").reindex_like(
        close
    )
    dates = (
        bars.groupby("timestamp_utc", sort=True)["trading_date"]
        .max()
        .reindex(close.index)
        .astype("datetime64[ns]")
    )
    return close.astype(float), returns.astype(float), dates


def _rolling_zscore(values: pd.DataFrame, history: int = 160) -> pd.DataFrame:
    prior_mean = values.rolling(history, min_periods=40).mean().shift(1)
    prior_std = values.rolling(history, min_periods=40).std().shift(1).replace(0.0, np.nan)
    return values.sub(prior_mean).div(prior_std).clip(-5.0, 5.0)


def build_score(
    close: pd.DataFrame, returns: pd.DataFrame, theme: Theme, lookback_bars: int
) -> pd.DataFrame:
    """Build causal scores from data available at each completed bar."""
    momentum = close.pct_change(lookback_bars, fill_method=None)
    if theme == "mean_reversion":
        return -_rolling_zscore(momentum)
    if theme == "relative_value":
        return -(momentum.rank(axis=1, pct=True) - 0.5) * 2.0
    peer_move = returns.rsub(returns.mean(axis=1, skipna=True), axis=0)
    lead_signal = peer_move.rolling(lookback_bars, min_periods=1).sum()
    return _rolling_zscore(lead_signal)


def eligibility_mask(returns: pd.DataFrame, volatility_filter: str, time_rule: str) -> pd.DataFrame:
    """Apply causal volatility and clock filters to potential entries."""
    eligible = returns.notna()
    realized = returns.rolling(40, min_periods=20).std().shift(1)
    median = realized.rolling(160, min_periods=40).median().shift(1)
    if volatility_filter == "below_median":
        eligible &= realized.le(median)
    elif volatility_filter == "above_median":
        eligible &= realized.gt(median)
    elif volatility_filter != "all":
        raise ValueError(f"unknown volatility filter: {volatility_filter}")
    timestamps = pd.DatetimeIndex(returns.index)
    local_hour = pd.Series(timestamps.tz_convert("America/New_York").hour, index=returns.index)
    if time_rule == "us_core":
        eligible &= np.broadcast_to(local_hour.between(8, 15).to_numpy()[:, None], eligible.shape)
    elif time_rule == "overnight":
        eligible &= np.broadcast_to(~local_hour.between(8, 15).to_numpy()[:, None], eligible.shape)
    elif time_rule != "all":
        raise ValueError(f"unknown time rule: {time_rule}")
    return eligible


@njit(cache=True)  # type: ignore[untyped-decorator]
def _position_kernel(
    scores: np.ndarray[Any, np.dtype[np.float64]],
    returns: np.ndarray[Any, np.dtype[np.float64]],
    eligible: np.ndarray[Any, np.dtype[np.bool_]],
    session_codes: np.ndarray[Any, np.dtype[np.int64]],
    entry: float,
    exit_: float,
    maximum_hold: int,
    stop_loss: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    rows, columns = scores.shape
    implemented = np.zeros((rows, columns), dtype=np.float64)
    for column in range(columns):
        position = 0.0
        held = 0
        cumulative = 0.0
        for row in range(rows):
            implemented[row, column] = position
            observed_return = returns[row, column]
            if position != 0.0 and np.isfinite(observed_return):
                cumulative += position * observed_return
                held += 1
            score = scores[row, column]
            stopped = stop_loss > 0.0 and cumulative <= -stop_loss
            exits = position != 0.0 and (
                stopped or held >= maximum_hold or not np.isfinite(score) or abs(score) <= exit_
            )
            if exits:
                position = 0.0
                held = 0
                cumulative = 0.0
            if (
                position == 0.0
                and eligible[row, column]
                and np.isfinite(score)
                and abs(score) >= entry
            ):
                position = 1.0 if score > 0.0 else -1.0
    return implemented


def simulate(
    scores: pd.DataFrame,
    returns: pd.DataFrame,
    dates: pd.Series,
    spec: IntradaySpec,
) -> tuple[pd.Series, pd.Series]:
    """Return daily gross returns and conservative sleeve-level turnover."""
    eligible = eligibility_mask(returns, spec.volatility_filter, spec.time_rule)
    session_codes = pd.factorize(dates, sort=True)[0].astype(np.int64)
    positions = _position_kernel(
        scores.to_numpy(dtype=float),
        returns.fillna(0.0).to_numpy(dtype=float),
        eligible.to_numpy(dtype=bool),
        session_codes,
        spec.entry_threshold,
        spec.exit_threshold,
        spec.maximum_hold_bars,
        -1.0 if spec.stop_loss is None else spec.stop_loss,
    )
    gross_exposure = np.abs(positions).sum(axis=1)
    weights = np.divide(
        positions,
        gross_exposure[:, None],
        out=np.zeros_like(positions),
        where=gross_exposure[:, None] > 0.0,
    )
    bar_return = np.nansum(weights * returns.fillna(0.0).to_numpy(dtype=float), axis=1)
    prior = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    bar_turnover = np.abs(weights - prior).sum(axis=1)
    daily_gross = pd.Series(bar_return, index=dates.index).groupby(dates).sum()
    daily_turnover = pd.Series(bar_turnover, index=dates.index).groupby(dates).sum()
    return daily_gross, daily_turnover


def apply_costs(gross: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """Apply a one-way cost assumption without rerunning the strategy."""
    return gross - turnover * cost_bps / 10_000.0


def _validation_mask(index: pd.DatetimeIndex, folds: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=index)
    for raw in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw)
        mask.loc[pd.Timestamp(fold["validation_start"]) : pd.Timestamp(fold["validation_end"])] = (
            True
        )
    return mask


def summarize(
    gross: pd.DataFrame, turnover: pd.DataFrame, registry: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    """Summarize every trial under a declared transaction-cost stress grid."""
    mask = _validation_mask(pd.DatetimeIndex(gross.index), folds)
    rows: list[dict[str, float | int | str]] = []
    for cost_bps in (0.0, 1.75, 3.5, 7.0):
        net = gross - turnover * cost_bps / 10_000.0
        for name in net.columns.astype(str):
            sample = net.loc[mask, name].dropna()
            positive_folds = 0
            worst_fold = np.inf
            for raw in folds.to_dict("records"):
                fold = cast(dict[str, Any], raw)
                fold_sample = net.loc[
                    pd.Timestamp(fold["validation_start"]) : pd.Timestamp(fold["validation_end"]),
                    name,
                ].dropna()
                fold_sharpe = sharpe_ratio(fold_sample)
                worst_fold = min(worst_fold, fold_sharpe)
                positive_folds += int(float(fold_sample.mean()) > 0.0)
            rows.append(
                {
                    "strategy": name,
                    "cost_bps": cost_bps,
                    "sessions": len(sample),
                    "annualized_return": float(sample.mean() * 252.0),
                    "annualized_volatility": annualized_volatility(sample),
                    "sharpe": sharpe_ratio(sample),
                    "maximum_drawdown": maximum_drawdown(sample),
                    "positive_folds": positive_folds,
                    "worst_fold_sharpe": worst_fold,
                    "average_daily_turnover": float(turnover.loc[mask, name].mean()),
                }
            )
    return (
        pd.DataFrame(rows).merge(registry, left_on="strategy", right_on="name").drop(columns="name")
    )


def complementary_portfolios(
    strategy_gross: pd.DataFrame, strategy_turnover: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal, inverse-volatility, and correlation-aware causal portfolios."""
    gross = strategy_gross.astype(float)
    turn = strategy_turnover.reindex_like(gross).astype(float)
    inverse_vol = 1.0 / gross.rolling(60, min_periods=20).std().shift(1).replace(0.0, np.nan)
    inverse_vol = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).fillna(1.0 / gross.shape[1])
    correlation_weights = pd.DataFrame(index=gross.index, columns=gross.columns, dtype=float)
    for row in range(len(gross)):
        history = gross.iloc[max(0, row - 60) : row]
        if len(history) < 20:
            correlation_weights.iloc[row] = 1.0 / gross.shape[1]
            continue
        covariance = history.cov().fillna(0.0)
        volatility = np.sqrt(np.diag(covariance)).clip(1e-8)
        correlation = covariance.to_numpy() / np.outer(volatility, volatility)
        penalty = 1.0 + np.nanmean(np.abs(correlation - np.eye(len(correlation))), axis=1)
        raw = 1.0 / (volatility * penalty)
        raw = np.minimum(raw / raw.sum(), 0.35)
        correlation_weights.iloc[row] = raw / raw.sum()
    equal = pd.DataFrame(1.0 / gross.shape[1], index=gross.index, columns=gross.columns)
    allocations = {
        "equal_weight": equal,
        "causal_inverse_volatility": inverse_vol,
        "causal_correlation_adjusted": correlation_weights,
    }
    portfolio_gross = pd.DataFrame(
        {name: (gross * weight).sum(axis=1) for name, weight in allocations.items()}
    )
    portfolio_turnover = pd.DataFrame(
        {
            name: (turn * weight).sum(axis=1)
            + weight.diff().abs().sum(axis=1).fillna(weight.iloc[0].abs().sum())
            for name, weight in allocations.items()
        }
    )
    return portfolio_gross, portfolio_turnover


def combine_with_saved_baseline(
    baseline: pd.Series,
    intraday: pd.Series,
    *,
    lookbacks: tuple[int, ...] = (20, 40, 60, 90, 120),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine session and intraday sleeves using only lagged volatility estimates."""
    sleeves = pd.concat(
        {"saved_session_baseline": baseline, "intraday_candidate": intraday},
        axis=1,
        sort=True,
    ).dropna()
    combinations: dict[str, pd.Series] = {}
    frozen_weights = pd.DataFrame(index=sleeves.index, columns=sleeves.columns, dtype=float)
    for lookback in lookbacks:
        minimum = max(10, lookback // 3)
        volatility = sleeves.rolling(lookback, min_periods=minimum).std().shift(1)
        inverse = 1.0 / volatility.replace(0.0, np.nan)
        weights = inverse.div(inverse.sum(axis=1), axis=0).fillna(0.5)
        name = f"combined_causal_inverse_vol_{lookback}d"
        combinations[name] = (sleeves * weights).sum(axis=1)
        if lookback == 120 or len(lookbacks) == 1:
            frozen_weights = weights
    return pd.DataFrame(combinations), frozen_weights


def run(project_root: Path) -> Path:
    """Execute the focused search on development data only and persist all results."""
    started = perf_counter()
    minute_root = project_root / "data/processed/research/development_minute_returns"
    source_frames = [
        pd.read_parquet(path) for path in sorted(minute_root.glob("symbol=*/returns.parquet"))
    ]
    source_frames = [frame for frame in source_frames if not frame.empty]
    if not source_frames:
        raise ValueError("no development minute data found")
    minute_data = pd.concat(source_frames, ignore_index=True)
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    specifications = strategy_specs()
    registry = pd.DataFrame([asdict(spec) for spec in specifications])
    gross_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    for timeframe in (30, 60, 240):
        bars = resample_bars(minute_data, timeframe)
        close, returns, dates = _panels(bars)
        score_cache: dict[tuple[Theme, int], pd.DataFrame] = {}
        for spec in [item for item in specifications if item.timeframe_minutes == timeframe]:
            cache_key = (spec.theme, spec.lookback_bars)
            if cache_key not in score_cache:
                score_cache[cache_key] = build_score(close, returns, *cache_key)
            gross_map[spec.name], turnover_map[spec.name] = simulate(
                score_cache[cache_key], returns, dates, spec
            )
    gross = pd.DataFrame(gross_map).sort_index()
    strategy_turnover = pd.DataFrame(turnover_map).reindex(gross.index)
    summary = summarize(gross, strategy_turnover, registry, folds)
    base = summary.loc[summary["cost_bps"].eq(3.5)].sort_values("sharpe", ascending=False)
    qualified = base.loc[base["annualized_return"].gt(0.0) & base["positive_folds"].ge(2)]
    if qualified.empty:
        qualified = base.head(1)
    selected = qualified.head(12)["strategy"].astype(str).tolist()
    portfolio_gross, portfolio_turnover = complementary_portfolios(
        gross[selected], strategy_turnover[selected]
    )
    portfolio_registry = pd.DataFrame(
        {"name": portfolio_gross.columns, "theme": "portfolio", "timeframe_minutes": 0}
    )
    portfolio_summary = summarize(portfolio_gross, portfolio_turnover, portfolio_registry, folds)
    baseline_path = (
        project_root
        / "saved_strategies/theme_champions_baseline_2026-08-23"
        / "runs/20260823T155542Z/portfolio_returns.parquet"
    )
    baseline = pd.read_parquet(baseline_path)["net_return"].astype(float)
    intraday_net = (
        portfolio_gross["causal_correlation_adjusted"]
        - portfolio_turnover["causal_correlation_adjusted"] * 3.5 / 10_000.0
    )
    combined_returns, frozen_weights = combine_with_saved_baseline(baseline, intraday_net)
    combined_fold, combined_summary = summarize_returns(combined_returns, folds)
    output = project_root / "data/processed/intraday_search"
    output.mkdir(parents=True, exist_ok=True)
    gross.to_parquet(output / "strategy_gross_returns.parquet")
    strategy_turnover.to_parquet(output / "strategy_turnover.parquet")
    registry.to_csv(output / "strategy_registry.csv", index=False)
    summary.to_csv(output / "strategy_cost_stress_summary.csv", index=False)
    portfolio_gross.to_parquet(output / "portfolio_gross_returns.parquet")
    portfolio_turnover.to_parquet(output / "portfolio_turnover.parquet")
    portfolio_summary.to_csv(output / "portfolio_cost_stress_summary.csv", index=False)
    combined_returns.to_parquet(output / "combined_portfolio_returns.parquet")
    frozen_weights.to_parquet(output / "frozen_candidate_weights.parquet")
    combined_fold.to_csv(output / "combined_portfolio_fold_summary.csv", index=False)
    combined_summary.to_csv(output / "combined_portfolio_summary.csv", index=False)
    best_portfolio = str(combined_summary.iloc[0]["strategy"])
    report = project_root / "outputs/intraday_strategy_search_tear_sheet.html"
    report_status = _write_quantstats_report(
        combined_returns[best_portfolio], report, f"Development Candidate: {best_portfolio}"
    )
    manifest = {
        "research_stage": "development_only",
        "holdout_accessed": False,
        "source": str(minute_root),
        "nonempty_symbols": sorted(minute_data["symbol"].astype(str).unique()),
        "strategies_tested": len(specifications),
        "timeframes_minutes": [30, 60, 240],
        "themes": ["mean_reversion", "relative_value", "lead_lag"],
        "cost_stress_bps_one_way": [0.0, 1.75, 3.5, 7.0],
        "selected_development_candidates": selected,
        "candidate_policy": (
            "positive net annualized return at 3.5 bps and at least two positive predefined "
            "validation folds; retain up to twelve by development Sharpe"
        ),
        "portfolio_methods": list(portfolio_gross.columns),
        "best_development_portfolio": best_portfolio,
        "saved_session_baseline": str(baseline_path),
        "frozen_candidate_weighting": "120-day lagged inverse volatility; 40-day minimum",
        "elapsed_seconds": perf_counter() - started,
        "quantstats_report": str(report),
        "quantstats_status": report_status,
        "freeze_decision": "frozen_development_candidate_holdout_not_run",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTop strategies at 3.5 bps")
    print(base.head(20).to_string(index=False))
    print("\nPortfolio cost stress")
    print(
        portfolio_summary.sort_values(["cost_bps", "sharpe"], ascending=[True, False]).to_string(
            index=False
        )
    )
    print("\nCombined session and intraday candidates")
    print(combined_summary.to_string(index=False))
    print(f"\nElapsed seconds: {manifest['elapsed_seconds']:.1f}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optimized intraday development search")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run(arguments.project_root.resolve())


if __name__ == "__main__":
    main()
