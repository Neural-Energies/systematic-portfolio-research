"""VectorBT-backed product strategy search on the unsealed Databento development panel."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt  # type: ignore[import-untyped]
from pypfopt import EfficientFrontier, HRPOpt  # type: ignore[import-untyped]
from pypfopt.risk_models import fix_nonpositive_semidefinite  # type: ignore[import-untyped]

from systematic_research.broad_strategy_search import _write_quantstats_report
from systematic_research.intraday_strategy_search import _panels, resample_bars
from systematic_research.saved_strategy_runner import performance_snapshot
from systematic_research.strategy_factory import (
    ProductStrategy,
    ResearchWindows,
    _metrics,
    build_product_score,
    entry_eligibility,
    generate_population,
    make_research_windows,
)

DEVELOPMENT_MINUTE_ROOT = Path("data/processed/databento_research/development_minute_returns")
OUTPUT_ROOT = Path("data/processed/vectorbt_strategy_factory")
RESAMPLE_SCHEMA_VERSION = 4


def load_development_minutes(project_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load only the physically separated Databento development minute partitions."""
    root = project_root / DEVELOPMENT_MINUTE_ROOT
    paths = sorted(root.glob("symbol=*/returns.parquet"))
    if not paths:
        raise FileNotFoundError(f"no Databento development partitions found under {root}")
    pieces = [pd.read_parquet(path) for path in paths]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        raise ValueError(f"all Databento development partitions are empty under {root}")
    minute_data = pd.concat(pieces, ignore_index=True)
    minute_data["timestamp_utc"] = pd.to_datetime(minute_data["timestamp_utc"], utc=True)
    return minute_data, paths


def _source_fingerprint(project_root: Path, paths: Sequence[Path]) -> list[dict[str, int | str]]:
    return [
        {
            "path": path.relative_to(project_root).as_posix(),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    ]


def cached_resampled_bars(
    project_root: Path,
    minute_data: pd.DataFrame,
    source_paths: Sequence[Path],
    timeframe_minutes: int,
) -> tuple[pd.DataFrame, bool]:
    """Reuse resampled development bars only when every input file identity still matches."""
    cache_root = project_root / OUTPUT_ROOT / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    bars_path = cache_root / f"bars_{timeframe_minutes}m.parquet"
    manifest_path = cache_root / f"bars_{timeframe_minutes}m_manifest.json"
    fingerprint = _source_fingerprint(project_root, source_paths)
    if bars_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("source_fingerprint") == fingerprint
            and manifest.get("resample_schema_version") == RESAMPLE_SCHEMA_VERSION
        ):
            return pd.read_parquet(bars_path), True
    bars = resample_bars(minute_data, timeframe_minutes)
    bars.to_parquet(bars_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "timeframe_minutes": timeframe_minutes,
                "source_fingerprint": fingerprint,
                "rows": len(bars),
                "resample_schema_version": RESAMPLE_SCHEMA_VERSION,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return bars, False


def delayed_signals(
    score: pd.Series,
    eligible: pd.Series,
    spec: ProductStrategy,
    trading_dates: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Create next-bar entries and causal score/holding exits for VectorBT execution."""
    directional = score * spec.direction
    allowed = eligible.fillna(False).astype(bool)
    long_active = directional.ge(spec.entry_threshold) & allowed
    short_active = directional.le(-spec.entry_threshold) & allowed
    long_entries = (long_active & ~long_active.shift(1, fill_value=False)).shift(
        1, fill_value=False
    )
    short_entries = (short_active & ~short_active.shift(1, fill_value=False)).shift(
        1, fill_value=False
    )
    if trading_dates is not None:
        same_session = trading_dates.eq(trading_dates.shift(1)).fillna(False)
        long_entries &= same_session
        short_entries &= same_session
    score_exit = directional.abs().le(spec.exit_threshold).shift(1, fill_value=False)
    timed_exit = (long_entries | short_entries).shift(spec.maximum_hold_bars, fill_value=False)
    exits = (score_exit | timed_exit).astype(bool)
    return long_entries.astype(bool), exits, short_entries.astype(bool), exits.copy()


def run_vectorbt_batch(
    bars: pd.DataFrame,
    specs: Sequence[ProductStrategy],
    score_cache: dict[tuple[str, int], pd.DataFrame],
    eligibility_cache: dict[tuple[str, str], pd.DataFrame],
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute one product/timeframe batch and return daily returns, entries, and trade records."""
    if not specs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    symbol = specs[0].symbol
    product_bars = (
        bars.loc[bars["symbol"].eq(symbol)]
        .sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc", keep="last")
        .set_index("timestamp_utc")
    )
    product_index = pd.DatetimeIndex(product_bars.index)
    product_dates = pd.Series(
        pd.to_datetime(product_bars["trading_date"]).to_numpy(), index=product_index
    )
    names = [spec.name for spec in specs]
    long_entries: dict[str, pd.Series] = {}
    long_exits: dict[str, pd.Series] = {}
    short_entries: dict[str, pd.Series] = {}
    short_exits: dict[str, pd.Series] = {}
    for spec in specs:
        score = score_cache[(spec.signal_family, spec.lookback_bars)][symbol].reindex(product_index)
        eligible = eligibility_cache[(spec.volatility_filter, spec.time_rule)][symbol].reindex(
            product_index, fill_value=False
        )
        signals = delayed_signals(score, eligible, spec, product_dates)
        long_entries[spec.name], long_exits[spec.name] = signals[0], signals[1]
        short_entries[spec.name], short_exits[spec.name] = signals[2], signals[3]

    entries_frame = pd.DataFrame(long_entries, index=product_index)
    exits_frame = pd.DataFrame(long_exits, index=product_index)
    short_entries_frame = pd.DataFrame(short_entries, index=product_index)
    short_exits_frame = pd.DataFrame(short_exits, index=product_index)
    session_end = ~product_dates.eq(product_dates.shift(-1)).fillna(False)
    entries_frame.loc[session_end] = False
    short_entries_frame.loc[session_end] = False
    exits_frame.loc[session_end] = True
    short_exits_frame.loc[session_end] = True
    entry_open = product_bars["open"].astype(float)
    valuation_close = product_bars["close"].astype(float)
    portfolio = vbt.Portfolio.from_signals(
        valuation_close,
        entries_frame,
        exits_frame,
        short_entries=short_entries_frame,
        short_exits=short_exits_frame,
        size=np.inf,
        size_type="amount",
        price=entry_open,
        open=entry_open,
        high=product_bars["high"].astype(float),
        low=product_bars["low"].astype(float),
        fees=cost_bps / 10_000.0,
        slippage=0.0,
        sl_stop=np.asarray([spec.stop_loss for spec in specs], dtype=float)[None, :],
        tp_stop=np.asarray([spec.profit_target for spec in specs], dtype=float)[None, :],
        stop_entry_price="fillprice",
        upon_opposite_entry="reverse",
        init_cash=1.0,
        freq=f"{specs[0].timeframe_minutes}min",
    )
    bar_returns = portfolio.returns().reindex(columns=names).fillna(0.0)
    trading_dates = product_dates
    daily_returns = (1.0 + bar_returns).groupby(trading_dates).prod().sub(1.0)

    records = portfolio.trades.records_readable.copy()
    daily_entries = pd.DataFrame(0.0, index=daily_returns.index, columns=names)
    if not records.empty:
        records["Entry Timestamp"] = pd.to_datetime(records["Entry Timestamp"], utc=True)
        timestamp_to_date = trading_dates.to_dict()
        records["trading_date"] = records["Entry Timestamp"].map(timestamp_to_date)
        counts = records.groupby(["trading_date", "Column"]).size().unstack(fill_value=0)
        daily_entries.loc[counts.index, counts.columns] = counts
        records["symbol"] = symbol
        records["timeframe_minutes"] = specs[0].timeframe_minutes
    return daily_returns, daily_entries, records


def select_up_to_one_hundred(summary: pd.DataFrame, target: int = 100) -> list[str]:
    """Rank positive build-and-selection expectancy without failing when fewer than 100 qualify."""
    eligible = summary.loc[
        summary["build_annualized_return"].gt(0.0)
        & summary["selection_annualized_return"].gt(0.0)
        & summary["build_trades"].ge(3)
        & summary["selection_trades"].ge(3)
    ].copy()
    eligible["robust_score"] = eligible[["build_sharpe", "selection_sharpe"]].min(
        axis=1
    ) + 0.25 * eligible[["build_sharpe", "selection_sharpe"]].mean(axis=1)
    return eligible.nlargest(target, "robust_score")["strategy"].astype(str).tolist()


def optimized_portfolios(
    selected_returns: pd.DataFrame,
    summary: pd.DataFrame,
    allocation_mask: pd.Series | np.ndarray[Any, np.dtype[np.bool_]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Create fixed and PyPortfolioOpt allocations using only the selection window."""
    names = selected_returns.columns.astype(str).tolist()
    if not names:
        raise ValueError("no positive build-and-selection strategies were available")
    equal = pd.Series(1.0 / len(names), index=names)
    strength = (
        summary.set_index("strategy").loc[names, "selection_annualized_return"].clip(lower=0.0)
    )
    expectancy = strength.div(strength.sum()) if strength.sum() > 0 else equal
    weight_columns: dict[str, pd.Series] = {
        "equal_weight": equal,
        "positive_expectancy_weighted": expectancy,
    }
    statuses: dict[str, str] = {}
    allocation = selected_returns.loc[allocation_mask, names].replace([np.inf, -np.inf], np.nan)
    allocation = allocation.dropna(how="all").fillna(0.0)
    if len(names) >= 2 and len(allocation) >= 20:
        try:
            hrp = HRPOpt(allocation).optimize()
            weight_columns["pypfopt_hrp"] = pd.Series(hrp, dtype=float).reindex(names).fillna(0.0)
            statuses["pypfopt_hrp"] = "ok"
        except (ArithmeticError, ValueError) as error:
            statuses["pypfopt_hrp"] = f"failed: {error}"
        try:
            mean = allocation.mean() * 252.0
            covariance = fix_nonpositive_semidefinite(allocation.cov() * 252.0)
            maximum_weight = max(0.10, min(1.0, 2.0 / len(names)))
            frontier = EfficientFrontier(
                mean,
                covariance,
                weight_bounds=(0.0, maximum_weight),
            )
            frontier.max_sharpe(risk_free_rate=0.0)
            cleaned = frontier.clean_weights(cutoff=0.0)
            weight_columns["pypfopt_max_sharpe"] = (
                pd.Series(cleaned, dtype=float).reindex(names).fillna(0.0)
            )
            statuses["pypfopt_max_sharpe"] = "ok"
        except (ArithmeticError, ValueError) as error:
            statuses["pypfopt_max_sharpe"] = f"failed: {error}"
    weights = pd.DataFrame(weight_columns).fillna(0.0)
    weights = weights.div(weights.sum(axis=0), axis=1)
    portfolios = pd.DataFrame(
        {name: selected_returns.mul(weight, axis=1).sum(axis=1) for name, weight in weights.items()}
    )
    return portfolios, weights, statuses


def _summarize_strategies(
    registry: pd.DataFrame,
    returns: pd.DataFrame,
    entries: pd.DataFrame,
    windows: ResearchWindows,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(returns.index)
    build_mask = dates <= pd.Timestamp(windows.build_end)
    selection_mask = (dates >= pd.Timestamp(windows.selection_start)) & (
        dates <= pd.Timestamp(windows.selection_end)
    )
    rows: list[dict[str, float | int | str]] = []
    for raw in registry.to_dict(orient="records"):
        name = str(raw["name"])
        build = _metrics(returns.loc[build_mask, name], entries.loc[build_mask, name])
        selection = _metrics(returns.loc[selection_mask, name], entries.loc[selection_mask, name])
        row: dict[str, float | int | str] = {"strategy": name, "symbol": str(raw["symbol"])}
        row.update({f"build_{key}": value for key, value in build.items()})
        row.update({f"selection_{key}": value for key, value in selection.items()})
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .merge(registry, left_on=["strategy", "symbol"], right_on=["name", "symbol"])
        .drop(columns="name")
    )


def run_factory(
    project_root: Path,
    *,
    per_symbol: int = 100,
    cost_bps: float = 3.5,
    seed: int = 20260823,
) -> Path:
    """Run a reproducible VectorBT search without loading the sealed year."""
    started = perf_counter()
    minute_data, source_paths = load_development_minutes(project_root)
    symbols = sorted(minute_data["symbol"].astype(str).unique())
    population = generate_population(symbols, per_symbol=per_symbol, seed=seed)
    registry = pd.DataFrame([asdict(spec) for spec in population])
    daily_dates = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    windows = make_research_windows(daily_dates)
    all_returns: list[pd.DataFrame] = []
    all_entries: list[pd.DataFrame] = []
    all_records: list[pd.DataFrame] = []
    cache_hits: dict[str, bool] = {}

    for timeframe in (15, 30, 60, 240):
        bars, cache_hit = cached_resampled_bars(project_root, minute_data, source_paths, timeframe)
        cache_hits[f"{timeframe}m"] = cache_hit
        close, returns, _ = _panels(bars)
        high = bars.pivot(index="timestamp_utc", columns="symbol", values="high").reindex_like(
            close
        )
        low = bars.pivot(index="timestamp_utc", columns="symbol", values="low").reindex_like(close)
        volume = bars.pivot(index="timestamp_utc", columns="symbol", values="volume").reindex_like(
            close
        )
        timeframe_specs = [spec for spec in population if spec.timeframe_minutes == timeframe]
        score_cache: dict[tuple[str, int], pd.DataFrame] = {}
        eligibility_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for spec in timeframe_specs:
            score_key = (spec.signal_family, spec.lookback_bars)
            if score_key not in score_cache:
                score_cache[score_key] = build_product_score(
                    close, high, low, volume, returns, spec.signal_family, spec.lookback_bars
                )
            eligibility_key = (spec.volatility_filter, spec.time_rule)
            if eligibility_key not in eligibility_cache:
                eligibility_cache[eligibility_key] = entry_eligibility(
                    returns, spec.volatility_filter, spec.time_rule
                )
        for symbol in symbols:
            batch_specs = [spec for spec in timeframe_specs if spec.symbol == symbol]
            batch_returns, batch_entries, records = run_vectorbt_batch(
                bars,
                batch_specs,
                score_cache,
                eligibility_cache,
                cost_bps=cost_bps,
            )
            all_returns.append(batch_returns)
            all_entries.append(batch_entries)
            if not records.empty:
                all_records.append(records)

    strategy_returns = pd.concat(all_returns, axis=1).reindex(daily_dates).fillna(0.0)
    strategy_entries = pd.concat(all_entries, axis=1).reindex(daily_dates).fillna(0.0)
    summary = _summarize_strategies(registry, strategy_returns, strategy_entries, windows)
    selected = select_up_to_one_hundred(summary)
    selected_returns = strategy_returns.loc[:, selected]
    selection_mask = (daily_dates >= pd.Timestamp(windows.selection_start)) & (
        daily_dates <= pd.Timestamp(windows.selection_end)
    )
    portfolios, weights, optimizer_status = optimized_portfolios(
        selected_returns, summary, selection_mask
    )
    test_mask = daily_dates >= pd.Timestamp(windows.portfolio_test_start)
    portfolio_rows: list[dict[str, float | int | str]] = []
    period_tables: list[pd.DataFrame] = []
    for name in portfolios.columns.astype(str):
        snapshot, periods = performance_snapshot(portfolios.loc[test_mask, name])
        portfolio_rows.append({"portfolio": name, **snapshot})
        period_tables.append(periods.assign(portfolio=name))
    portfolio_summary = pd.DataFrame(portfolio_rows).sort_values("sharpe", ascending=False)
    period_summary = pd.concat(period_tables, ignore_index=True)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    registry.to_csv(output / "generated_population.csv", index=False)
    summary.to_parquet(output / "candidate_build_selection_summary.parquet", index=False)
    strategy_returns.to_parquet(output / "all_strategy_returns.parquet")
    strategy_entries.to_parquet(output / "all_strategy_entries.parquet")
    summary.loc[summary["strategy"].isin(selected)].to_csv(
        output / "selected_strategies.csv", index=False
    )
    selected_returns.to_parquet(output / "selected_strategy_returns.parquet")
    portfolios.to_parquet(output / "portfolio_returns.parquet")
    weights.to_csv(output / "portfolio_strategy_weights.csv")
    portfolio_summary.to_csv(output / "portfolio_summary.csv", index=False)
    period_summary.to_csv(output / "portfolio_calendar_returns.csv", index=False)
    if all_records:
        pd.concat(all_records, ignore_index=True).to_parquet(
            output / "vectorbt_trade_ledger.parquet"
        )
    best = str(portfolio_summary.iloc[0]["portfolio"])
    report = output / "tear_sheet.html"
    report_status = _write_quantstats_report(
        portfolios.loc[test_mask, best], report, f"VectorBT Strategy Factory: {best}"
    )
    manifest = {
        "engine": "vectorbt.Portfolio.from_signals",
        "vectorbt_version": vbt.__version__,
        "portfolio_optimizer": "PyPortfolioOpt",
        "research_stage": "development_only",
        "sealed_year_accessed": False,
        "development_source_root": DEVELOPMENT_MINUTE_ROOT.as_posix(),
        "source_fingerprint": _source_fingerprint(project_root, source_paths),
        "products": symbols,
        "generated_strategies": len(population),
        "selected_positive_expectancy_strategies": len(selected),
        "seed": seed,
        "per_symbol": per_symbol,
        "windows": asdict(windows),
        "one_way_all_in_cost_bps": cost_bps,
        "return_unit": "return_on_one_unit_of_notional",
        "execution_price": "next_resampled_bar_open",
        "cache_hits": cache_hits,
        "optimizer_status": optimizer_status,
        "highest_scoring_diagnostic": best,
        "tear_sheet_status": report_status,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest = project_root / OUTPUT_ROOT / "latest_run.json"
    latest.write_text(
        json.dumps({"run_id": run_id, "path": str(output)}, indent=2), encoding="utf-8"
    )
    print(portfolio_summary.to_string(index=False))
    print(f"\nSelected strategies: {len(selected)}")
    print(f"Elapsed seconds: {manifest['elapsed_seconds']:.1f}")
    print(f"Saved run: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VectorBT product strategy factory")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--per-symbol", type=int, default=100)
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20260823)
    arguments = parser.parse_args()
    run_factory(
        arguments.project_root.resolve(),
        per_symbol=arguments.per_symbol,
        cost_bps=arguments.cost_bps,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
