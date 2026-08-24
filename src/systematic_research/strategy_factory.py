"""StrategyQuant-style product-level strategy factory and portfolio search."""

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
from systematic_research.intraday_strategy_search import _panels, resample_bars
from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.saved_strategy_runner import performance_snapshot

SignalFamily = Literal[
    "momentum",
    "mean_reversion",
    "breakout",
    "trend",
    "lead_lag",
    "volume_confirmation",
    "cross_sectional_momentum",
    "cross_sectional_reversal",
    "residual_reversion",
    "volatility_scaled_momentum",
    "intraday_close_momentum",
    "opening_range_continuation",
    "opening_range_reversal",
    "session_vwap_reversion",
    "range_breakout",
    "range_expansion_reversal",
    "paired_spread_reversion",
    "regime_switch",
]


@dataclass(frozen=True)
class ProductStrategy:
    """One independently tradable strategy for one futures product."""

    name: str
    symbol: str
    timeframe_minutes: int
    signal_family: SignalFamily
    lookback_bars: int
    direction: int
    entry_threshold: float
    exit_threshold: float
    maximum_hold_bars: int
    volatility_filter: str
    time_rule: str
    stop_loss: float
    profit_target: float


@dataclass(frozen=True)
class ResearchWindows:
    """Three non-overlapping development windows."""

    build_end: str
    selection_start: str
    selection_end: str
    portfolio_test_start: str


def make_research_windows(dates: pd.DatetimeIndex) -> ResearchWindows:
    """Reserve the existing outer-validation period for portfolio-level testing."""
    unique = pd.DatetimeIndex(sorted(pd.to_datetime(dates).unique()))
    test_start = pd.Timestamp("2024-10-01")
    pretest = unique[unique < test_start]
    if len(pretest) < 200 or not (unique >= test_start).any():
        raise ValueError("insufficient history for build, selection, and portfolio test windows")
    cut = int(len(pretest) * 0.70)
    return ResearchWindows(
        build_end=pretest[cut - 1].date().isoformat(),
        selection_start=pretest[cut].date().isoformat(),
        selection_end=pretest[-1].date().isoformat(),
        portfolio_test_start=test_start.date().isoformat(),
    )


def generate_population(
    symbols: list[str], *, per_symbol: int = 2000, seed: int = 20260823
) -> list[ProductStrategy]:
    """Generate a deterministic, balanced population from transparent rule blocks."""
    rng = np.random.default_rng(seed)
    families: tuple[SignalFamily, ...] = (
        "momentum",
        "mean_reversion",
        "breakout",
        "trend",
        "lead_lag",
        "volume_confirmation",
    )
    population: list[ProductStrategy] = []
    seen: set[tuple[object, ...]] = set()
    for symbol in sorted(symbols):
        while sum(spec.symbol == symbol for spec in population) < per_symbol:
            values = (
                symbol,
                int(rng.choice([15, 30, 60, 240])),
                str(rng.choice(families)),
                int(rng.choice([2, 4, 8, 16, 24])),
                int(rng.choice([-1, 1])),
                float(rng.choice([0.50, 0.75, 1.00, 1.25, 1.50, 2.00])),
                float(rng.choice([0.00, 0.10, 0.25, 0.50])),
                int(rng.choice([2, 4, 8, 16, 24, 40])),
                str(rng.choice(["all", "below_median", "above_median"])),
                str(rng.choice(["all", "us_core", "overnight"])),
                float(rng.choice([0.005, 0.010, 0.020, 0.040])),
                float(rng.choice([0.005, 0.010, 0.020, 0.040, 0.080])),
            )
            if values in seen:
                continue
            seen.add(values)
            population.append(
                ProductStrategy(
                    name=f"factory_{len(population) + 1:05d}",
                    symbol=values[0],
                    timeframe_minutes=values[1],
                    signal_family=cast(SignalFamily, values[2]),
                    lookback_bars=values[3],
                    direction=values[4],
                    entry_threshold=values[5],
                    exit_threshold=values[6],
                    maximum_hold_bars=values[7],
                    volatility_filter=values[8],
                    time_rule=values[9],
                    stop_loss=values[10],
                    profit_target=values[11],
                )
            )
    return population


def _rolling_zscore(values: pd.DataFrame, history: int = 160) -> pd.DataFrame:
    mean = values.rolling(history, min_periods=40).mean().shift(1)
    std = values.rolling(history, min_periods=40).std().shift(1).replace(0.0, np.nan)
    return values.sub(mean).div(std).clip(-5.0, 5.0)


def build_product_score(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    family: SignalFamily,
    lookback: int,
) -> pd.DataFrame:
    """Build a causal score panel for a reusable rule block."""
    # ``returns`` is reset to NaN at every trading-date boundary by resample_bars.
    # Requiring a complete window therefore prevents signals from spanning an
    # unadjusted futures roll/session gap.
    momentum = returns.rolling(lookback, min_periods=lookback).sum()
    if family == "momentum":
        return _rolling_zscore(momentum)
    if family == "mean_reversion":
        return -_rolling_zscore(momentum)
    if family == "cross_sectional_momentum":
        return (momentum.rank(axis=1, pct=True) - 0.5) * 2.0
    if family == "cross_sectional_reversal":
        return -(momentum.rank(axis=1, pct=True) - 0.5) * 2.0
    if family == "residual_reversion":
        peer = returns.apply(lambda column: returns.drop(columns=column.name).mean(axis=1))
        residual = returns.sub(peer)
        return -_rolling_zscore(residual.rolling(lookback, min_periods=lookback).sum())
    if family == "volatility_scaled_momentum":
        prior_volatility = returns.rolling(lookback * 4, min_periods=lookback).std().shift(1)
        return _rolling_zscore(momentum.div(prior_volatility.replace(0.0, np.nan)))
    if family == "intraday_close_momentum":
        # The factory supplies a session-cumulative implementation because this
        # generic scorer does not receive the per-product trading-date panel.
        return _rolling_zscore(momentum)
    if family in {
        "opening_range_continuation",
        "opening_range_reversal",
        "session_vwap_reversion",
        "range_breakout",
        "range_expansion_reversal",
        "paired_spread_reversion",
        "regime_switch",
    }:
        # Session-aware implementations live in paper_strategy_factory, which
        # receives trading dates and the economic peer map.
        return _rolling_zscore(momentum)
    if family == "breakout":
        rolling_high = high.rolling(lookback, min_periods=lookback).max()
        rolling_low = low.rolling(lookback, min_periods=lookback).min()
        location = close.sub(rolling_low).div(rolling_high.sub(rolling_low).replace(0.0, np.nan))
        return _rolling_zscore(location.sub(0.5))
    if family == "trend":
        fast = close.ewm(span=lookback, adjust=False, min_periods=lookback).mean()
        slow = close.ewm(span=lookback * 4, adjust=False, min_periods=lookback * 4).mean()
        return _rolling_zscore(fast.div(slow).sub(1.0))
    if family == "lead_lag":
        peer = returns.apply(lambda column: returns.drop(columns=column.name).mean(axis=1))
        return _rolling_zscore(peer.rolling(lookback, min_periods=1).sum())
    log_volume = pd.DataFrame(
        np.log1p(volume.clip(lower=0.0).to_numpy(dtype=float)),
        index=volume.index,
        columns=volume.columns,
    )
    volume_impulse = _rolling_zscore(log_volume)
    return _rolling_zscore(momentum) * volume_impulse.clip(-2.0, 2.0)


def entry_eligibility(
    returns: pd.DataFrame, volatility_filter: str, time_rule: str
) -> pd.DataFrame:
    """Build causal entry filters while allowing open positions to exit normally."""
    eligible = returns.notna()
    realized = returns.rolling(40, min_periods=20).std().shift(1)
    median = realized.rolling(160, min_periods=40).median().shift(1)
    if volatility_filter == "below_median":
        eligible &= realized.le(median)
    elif volatility_filter == "above_median":
        eligible &= realized.gt(median)
    timestamps = pd.DatetimeIndex(returns.index)
    local_hour = pd.Series(timestamps.tz_convert("America/New_York").hour, index=returns.index)
    if time_rule == "us_core":
        eligible &= np.broadcast_to(local_hour.between(8, 15).to_numpy()[:, None], eligible.shape)
    elif time_rule == "overnight":
        eligible &= np.broadcast_to(~local_hour.between(8, 15).to_numpy()[:, None], eligible.shape)
    return eligible


@njit(cache=True)  # type: ignore[untyped-decorator]
def _single_product_kernel(
    score: np.ndarray[Any, np.dtype[np.float64]],
    returns: np.ndarray[Any, np.dtype[np.float64]],
    eligible: np.ndarray[Any, np.dtype[np.bool_]],
    entry: float,
    exit_: float,
    maximum_hold: int,
    stop_loss: float,
    profit_target: float,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    position = 0.0
    held = 0
    cumulative = 0.0
    implemented = np.zeros(len(score), dtype=np.float64)
    entries = np.zeros(len(score), dtype=np.float64)
    for row in range(len(score)):
        implemented[row] = position
        observed = returns[row]
        if position != 0.0 and np.isfinite(observed):
            cumulative += position * observed
            held += 1
        current_score = score[row]
        exits = position != 0.0 and (
            cumulative <= -stop_loss
            or cumulative >= profit_target
            or held >= maximum_hold
            or not np.isfinite(current_score)
            or abs(current_score) <= exit_
        )
        if exits:
            position = 0.0
            held = 0
            cumulative = 0.0
        if (
            position == 0.0
            and eligible[row]
            and np.isfinite(current_score)
            and abs(current_score) >= entry
        ):
            position = 1.0 if current_score > 0.0 else -1.0
            entries[row] = 1.0
    return implemented, entries


def simulate_product_strategy(
    score: pd.Series,
    returns: pd.Series,
    eligible: pd.Series,
    dates: pd.Series,
    spec: ProductStrategy,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Simulate one delayed single-product strategy and aggregate it daily."""
    positions, entries = _single_product_kernel(
        score.to_numpy(dtype=float) * spec.direction,
        returns.fillna(0.0).to_numpy(dtype=float),
        eligible.to_numpy(dtype=bool),
        spec.entry_threshold,
        spec.exit_threshold,
        spec.maximum_hold_bars,
        spec.stop_loss,
        spec.profit_target,
    )
    prior = np.concatenate([np.zeros(1), positions[:-1]])
    gross = positions * returns.fillna(0.0).to_numpy(dtype=float)
    turnover = np.abs(positions - prior)
    index = dates.index
    daily_gross = pd.Series(gross, index=index).groupby(dates).sum()
    daily_turnover = pd.Series(turnover, index=index).groupby(dates).sum()
    daily_entries = pd.Series(entries, index=index).groupby(dates).sum()
    return daily_gross, daily_turnover, daily_entries


def _metrics(returns: pd.Series, entries: pd.Series) -> dict[str, float | int]:
    sample = returns.dropna()
    return {
        "sessions": len(sample),
        "annualized_return": float(sample.mean() * 252.0),
        "annualized_volatility": annualized_volatility(sample),
        "sharpe": sharpe_ratio(sample),
        "maximum_drawdown": maximum_drawdown(sample),
        "trades": int(entries.reindex(sample.index).fillna(0.0).sum()),
    }


def select_one_hundred(summary: pd.DataFrame) -> list[str]:
    """Select positive-expectancy strategies without observing portfolio-test results."""
    required = {
        "strategy",
        "symbol",
        "build_annualized_return",
        "selection_annualized_return",
        "build_sharpe",
        "selection_sharpe",
        "build_trades",
        "selection_trades",
    }
    if not required.issubset(summary.columns):
        raise ValueError(f"summary missing columns: {sorted(required - set(summary.columns))}")
    eligible = summary.loc[
        summary["build_annualized_return"].gt(0.0)
        & summary["selection_annualized_return"].gt(0.0)
        & summary["build_trades"].ge(10)
        & summary["selection_trades"].ge(5)
    ].copy()
    eligible["robust_score"] = eligible[["build_sharpe", "selection_sharpe"]].min(
        axis=1
    ) + 0.25 * eligible[["build_sharpe", "selection_sharpe"]].mean(axis=1)
    if len(eligible) < 100:
        counts = eligible.groupby("symbol").size().to_dict()
        raise RuntimeError(f"only {len(eligible)} positive-expectancy candidates: {counts}")
    return eligible.nlargest(100, "robust_score")["strategy"].astype(str).tolist()


def portfolio_candidates(
    selected_returns: pd.DataFrame, selection_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aggressive fixed-capital portfolios from selected strategy returns."""
    names = selected_returns.columns.astype(str).tolist()
    equal = pd.Series(1.0 / len(names), index=names)
    strength = selection_summary.set_index("strategy").loc[names, "selection_annualized_return"]
    strength = strength.clip(lower=0.0)
    expectancy = strength.div(strength.sum())
    ranking = (
        selection_summary.set_index("strategy")
        .loc[names]
        .sort_values("selection_sharpe", ascending=False)
    )
    top_25 = ranking.head(25).index.astype(str)
    top_50 = ranking.head(50).index.astype(str)
    weights = pd.DataFrame(
        {
            "all_100_equal_capital": equal,
            "all_100_expectancy_weighted": expectancy,
            "top_50_equal_capital": pd.Series(1.0 / 50.0, index=top_50),
            "top_25_equal_capital": pd.Series(1.0 / 25.0, index=top_25),
        }
    ).fillna(0.0)
    portfolios = pd.DataFrame(
        {name: selected_returns.mul(weight, axis=1).sum(axis=1) for name, weight in weights.items()}
    )
    return portfolios, weights


def run_factory(project_root: Path, *, per_symbol: int = 2000, cost_bps: float = 3.5) -> Path:
    """Generate, select, and portfolio-test product-level strategies on development only."""
    started = perf_counter()
    minute_root = project_root / "data/processed/research/development_minute_returns"
    pieces = [
        pd.read_parquet(path) for path in sorted(minute_root.glob("symbol=*/returns.parquet"))
    ]
    pieces = [piece for piece in pieces if not piece.empty]
    minute_data = pd.concat(pieces, ignore_index=True)
    symbols = sorted(minute_data["symbol"].astype(str).unique())
    population = generate_population(symbols, per_symbol=per_symbol)
    registry = pd.DataFrame([asdict(spec) for spec in population])

    daily_dates = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    windows = make_research_windows(daily_dates)
    gross_matrix = np.zeros((len(daily_dates), len(population)), dtype=np.float32)
    turnover_matrix = np.zeros_like(gross_matrix)
    entries_matrix = np.zeros_like(gross_matrix)
    date_positions = pd.Series(np.arange(len(daily_dates)), index=daily_dates)

    for timeframe in (30, 60, 240):
        bars = resample_bars(minute_data, timeframe)
        close, returns, dates = _panels(bars)
        high = bars.pivot(index="timestamp_utc", columns="symbol", values="high").reindex_like(
            close
        )
        low = bars.pivot(index="timestamp_utc", columns="symbol", values="low").reindex_like(close)
        volume = bars.pivot(index="timestamp_utc", columns="symbol", values="volume").reindex_like(
            close
        )
        score_cache: dict[tuple[SignalFamily, int], pd.DataFrame] = {}
        eligibility_cache: dict[tuple[str, str], pd.DataFrame] = {}
        indices = [i for i, spec in enumerate(population) if spec.timeframe_minutes == timeframe]
        for index in indices:
            spec = population[index]
            score_key = (spec.signal_family, spec.lookback_bars)
            if score_key not in score_cache:
                score_cache[score_key] = build_product_score(
                    close, high, low, volume, returns, *score_key
                )
            eligibility_key = (spec.volatility_filter, spec.time_rule)
            if eligibility_key not in eligibility_cache:
                eligibility_cache[eligibility_key] = entry_eligibility(returns, *eligibility_key)
            gross, turnover, entries = simulate_product_strategy(
                score_cache[score_key][spec.symbol],
                returns[spec.symbol],
                eligibility_cache[eligibility_key][spec.symbol],
                dates,
                spec,
            )
            positions = date_positions.loc[pd.DatetimeIndex(gross.index)].to_numpy(dtype=int)
            gross_matrix[positions, index] = gross.to_numpy(dtype=np.float32)
            turnover_matrix[positions, index] = turnover.to_numpy(dtype=np.float32)
            entries_matrix[positions, index] = entries.to_numpy(dtype=np.float32)

    gross_frame = pd.DataFrame(gross_matrix, index=daily_dates, columns=registry["name"])
    turnover_frame = pd.DataFrame(turnover_matrix, index=daily_dates, columns=registry["name"])
    entries_frame = pd.DataFrame(entries_matrix, index=daily_dates, columns=registry["name"])
    net = gross_frame - turnover_frame * cost_bps / 10_000.0
    build_mask = daily_dates <= pd.Timestamp(windows.build_end)
    selection_mask = (daily_dates >= pd.Timestamp(windows.selection_start)) & (
        daily_dates <= pd.Timestamp(windows.selection_end)
    )
    summary_rows: list[dict[str, float | int | str]] = []
    for _, raw in registry.iterrows():
        name = str(raw["name"])
        build = _metrics(net.loc[build_mask, name], entries_frame.loc[build_mask, name])
        selection = _metrics(net.loc[selection_mask, name], entries_frame.loc[selection_mask, name])
        row: dict[str, float | int | str] = {"strategy": name, "symbol": str(raw["symbol"])}
        row.update({f"build_{key}": value for key, value in build.items()})
        row.update({f"selection_{key}": value for key, value in selection.items()})
        summary_rows.append(row)
    summary = (
        pd.DataFrame(summary_rows)
        .merge(registry, left_on=["strategy", "symbol"], right_on=["name", "symbol"])
        .drop(columns="name")
    )
    selected = select_one_hundred(summary)
    test_mask = daily_dates >= pd.Timestamp(windows.portfolio_test_start)
    selected_net = net.loc[:, selected]
    portfolios, weights = portfolio_candidates(selected_net, summary)
    portfolio_rows: list[dict[str, float | int | str]] = []
    period_tables: list[pd.DataFrame] = []
    for name in portfolios.columns.astype(str):
        snapshot, periods = performance_snapshot(portfolios.loc[test_mask, name])
        portfolio_rows.append({"portfolio": name, **snapshot})
        period_tables.append(periods.assign(portfolio=name))
    portfolio_summary = pd.DataFrame(portfolio_rows).sort_values("sharpe", ascending=False)
    period_summary = pd.concat(period_tables, ignore_index=True)

    output = project_root / "data/processed/strategy_factory"
    output.mkdir(parents=True, exist_ok=True)
    registry.to_csv(output / "generated_population.csv", index=False)
    summary.to_parquet(output / "candidate_build_selection_summary.parquet", index=False)
    summary.loc[summary["strategy"].isin(selected)].to_csv(
        output / "selected_100_strategies.csv", index=False
    )
    selected_net.to_parquet(output / "selected_100_returns.parquet")
    portfolios.to_parquet(output / "portfolio_returns.parquet")
    weights.to_csv(output / "portfolio_strategy_weights.csv")
    portfolio_summary.to_csv(output / "portfolio_summary.csv", index=False)
    period_summary.to_csv(output / "portfolio_calendar_returns.csv", index=False)
    best = str(portfolio_summary.iloc[0]["portfolio"])
    report = project_root / "outputs/strategy_factory_tear_sheet.html"
    report_status = _write_quantstats_report(
        portfolios.loc[test_mask, best], report, f"100-Strategy Factory: {best}"
    )
    manifest = {
        "research_stage": "new_development_only_after_consumed_holdout",
        "consumed_sealed_year_accessed": False,
        "generated_strategies": len(population),
        "selected_strategies": len(selected),
        "selected_by_product": (
            summary.loc[summary["strategy"].isin(selected)].groupby("symbol").size().to_dict()
        ),
        "products": symbols,
        "windows": asdict(windows),
        "one_way_cost_bps": cost_bps,
        "portfolio_test_was_not_used_for_strategy_selection": True,
        "acceptance_targets": {
            "minimum_sharpe": 2.0,
            "minimum_win_months": 0.60,
            "minimum_win_quarters": 0.50,
            "minimum_total_return": "match_sp500_same_period",
        },
        "candidate_found": False,
        "highest_scoring_diagnostic": best,
        "benchmark": "data/processed/strategy_factory/sp500_benchmark.json",
        "tear_sheet": str(report),
        "tear_sheet_status": report_status,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n100-strategy diagnostic results (not accepted candidates)")
    print(portfolio_summary.to_string(index=False))
    print(f"\nElapsed seconds: {manifest['elapsed_seconds']:.1f}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run product-level strategy factory")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--per-symbol", type=int, default=2000)
    parser.add_argument("--cost-bps", type=float, default=3.5)
    arguments = parser.parse_args()
    run_factory(
        arguments.project_root.resolve(),
        per_symbol=arguments.per_symbol,
        cost_bps=arguments.cost_bps,
    )


if __name__ == "__main__":
    main()
