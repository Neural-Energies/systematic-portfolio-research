"""Low-turnover, roll-safe daily-session strategy search and independent replay."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt  # type: ignore[import-untyped]
from pypfopt import HRPOpt  # type: ignore[import-untyped]

from systematic_research.broad_strategy_search import _write_quantstats_report
from systematic_research.metrics import sharpe_ratio
from systematic_research.saved_strategy_runner import performance_snapshot
from systematic_research.strategy_factory import ResearchWindows, _metrics, make_research_windows

DEVELOPMENT_PANEL = Path("data/processed/databento_research/development_session_panel.parquet")
OUTPUT_ROOT = Path("data/processed/daily_session_factory")


@dataclass(frozen=True)
class DailySessionSpec:
    """A signal observed at one session close and traded next session open-to-close."""

    name: str
    symbol: str
    family: str
    lookback_sessions: int
    entry_threshold: float


def load_development_panel(project_root: Path) -> pd.DataFrame:
    """Load only the physically separated development session panel."""
    path = project_root / DEVELOPMENT_PANEL
    frame = pd.read_parquet(path)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame["session_open_utc"] = pd.to_datetime(frame["session_open_utc"], utc=True)
    frame["session_close_utc"] = pd.to_datetime(frame["session_close_utc"], utc=True)
    return frame.sort_values(["symbol", "trading_date"], ignore_index=True)


def session_panels(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create session-only panels; returns exclude every overnight and roll gap."""
    indexed = frame.copy()
    indexed["session_return"] = indexed["close"].div(indexed["open"]).sub(1.0)
    indexed["range_fraction"] = indexed["high"].sub(indexed["low"]).div(indexed["open"])
    values: dict[str, pd.DataFrame] = {}
    for column in ("open", "close", "volume", "session_return", "range_fraction"):
        values[column] = indexed.pivot(
            index="trading_date", columns="symbol", values=column
        ).sort_index()
    return values


def generate_population(symbols: list[str]) -> list[DailySessionSpec]:
    """Generate a complete deterministic grid with economically fixed directions."""
    population: list[DailySessionSpec] = []
    families = ("session_momentum", "session_reversal")
    for symbol in sorted(symbols):
        for family in families:
            for lookback in (1, 2, 3, 5, 10, 20, 40, 60):
                for threshold in (0.0, 0.5, 1.0, 1.5):
                    population.append(
                        DailySessionSpec(
                            name=f"daily_{len(population) + 1:04d}",
                            symbol=symbol,
                            family=family,
                            lookback_sessions=lookback,
                            entry_threshold=threshold,
                        )
                    )
    return population


def score_panels(session_returns: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    """Build causal standardized trailing-session scores."""
    scores: dict[tuple[str, int], pd.DataFrame] = {}
    for lookback in (1, 2, 3, 5, 10, 20, 40, 60):
        trailing = session_returns.rolling(lookback, min_periods=lookback).sum()
        scale = session_returns.rolling(max(20, lookback * 2), min_periods=20).std().shift(1)
        standardized = trailing.div(scale.mul(np.sqrt(lookback)).replace(0.0, np.nan))
        scores[("session_momentum", lookback)] = standardized.clip(-8.0, 8.0)
        scores[("session_reversal", lookback)] = -standardized.clip(-8.0, 8.0)
    return scores


def simulate_population(
    specs: list[DailySessionSpec],
    session_returns: pd.DataFrame,
    scores: dict[tuple[str, int], pd.DataFrame],
    *,
    one_way_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Search with prior-close signals, next-open entries, and same-session exits."""
    net: dict[str, pd.Series] = {}
    gross: dict[str, pd.Series] = {}
    entries: dict[str, pd.Series] = {}
    for spec in specs:
        observed = scores[(spec.family, spec.lookback_sessions)][spec.symbol]
        position = (
            np.sign(observed).where(observed.abs().ge(spec.entry_threshold), 0.0).shift(1)
        ).fillna(0.0)
        strategy_gross = position * session_returns[spec.symbol].fillna(0.0)
        sides = position.abs() * 2.0
        gross[spec.name] = strategy_gross
        entries[spec.name] = position.ne(0.0).astype(float)
        net[spec.name] = strategy_gross - sides * one_way_cost_bps / 10_000.0
    index = session_returns.index
    return (
        pd.DataFrame(net, index=index),
        pd.DataFrame(gross, index=index),
        pd.DataFrame(entries, index=index),
    )


def summarize_population(
    specs: list[DailySessionSpec],
    returns: pd.DataFrame,
    entries: pd.DataFrame,
    windows: ResearchWindows,
) -> pd.DataFrame:
    """Summarize only build and selection for candidate ranking."""
    index = pd.DatetimeIndex(returns.index)
    build = index <= pd.Timestamp(windows.build_end)
    selection = (index >= pd.Timestamp(windows.selection_start)) & (
        index <= pd.Timestamp(windows.selection_end)
    )
    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = asdict(spec)
        row.update(
            {
                f"build_{key}": value
                for key, value in _metrics(
                    returns.loc[build, spec.name], entries.loc[build, spec.name]
                ).items()
            }
        )
        row.update(
            {
                f"selection_{key}": value
                for key, value in _metrics(
                    returns.loc[selection, spec.name], entries.loc[selection, spec.name]
                ).items()
            }
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["robust_score"] = summary[["build_sharpe", "selection_sharpe"]].min(axis=1)
    summary["robust_score"] += 0.25 * summary[["build_sharpe", "selection_sharpe"]].mean(axis=1)
    return summary


def select_complementary(
    summary: pd.DataFrame,
    returns: pd.DataFrame,
    selection_mask: np.ndarray[Any, np.dtype[np.bool_]],
    *,
    target: int = 12,
    maximum_per_symbol: int = 2,
    maximum_correlation: float = 0.65,
) -> list[str]:
    """Select positive early-window candidates with explicit concentration limits."""
    eligible = summary.loc[
        summary["build_annualized_return"].gt(0.0)
        & summary["selection_annualized_return"].gt(0.0)
        & summary["build_trades"].ge(20)
        & summary["selection_trades"].ge(8)
    ].sort_values("robust_score", ascending=False)
    selected: list[str] = []
    symbol_counts: dict[str, int] = {}
    history = returns.loc[selection_mask]
    for raw in eligible.to_dict("records"):
        name = str(raw["name"])
        symbol = str(raw["symbol"])
        if symbol_counts.get(symbol, 0) >= maximum_per_symbol:
            continue
        if selected:
            correlation = history[selected].corrwith(history[name]).abs().max()
            if pd.notna(correlation) and float(correlation) > maximum_correlation:
                continue
        selected.append(name)
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        if len(selected) >= target:
            break
    if len(selected) < 3:
        raise ValueError(f"only {len(selected)} complementary early-window candidates")
    return selected


def build_parameter_ensembles(
    summary: pd.DataFrame, returns: pd.DataFrame, *, minimum_sleeves: int = 3
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Average eligible parameter variants before allocating across sleeves."""
    eligible = summary.loc[
        summary["build_annualized_return"].gt(0.0)
        & summary["selection_annualized_return"].gt(0.0)
        & summary["build_trades"].ge(20)
        & summary["selection_trades"].ge(8)
    ]
    output: dict[str, pd.Series] = {}
    members: dict[str, list[str]] = {}
    for (raw_symbol, raw_family), group in eligible.groupby(["symbol", "family"], sort=True):
        symbol, family = str(raw_symbol), str(raw_family)
        names = group["name"].astype(str).tolist()
        if len(names) < 2:
            continue
        sleeve = f"{symbol}_{family}"
        members[sleeve] = names
        output[sleeve] = returns[names].mean(axis=1)
    if len(output) < minimum_sleeves:
        raise ValueError(f"only {len(output)} parameter-ensemble sleeves")
    return pd.DataFrame(output, index=returns.index), members


def summarize_and_select_ensembles(
    returns: pd.DataFrame,
    windows: ResearchWindows,
    *,
    target: int = 8,
    minimum_robust_sharpe: float = 0.25,
    maximum_correlation: float = 0.70,
) -> tuple[pd.DataFrame, list[str]]:
    """Rank parameter ensembles on early windows and retain distinct sleeves."""
    index = pd.DatetimeIndex(returns.index)
    build = index <= pd.Timestamp(windows.build_end)
    selection = (index >= pd.Timestamp(windows.selection_start)) & (
        index <= pd.Timestamp(windows.selection_end)
    )
    rows: list[dict[str, Any]] = []
    for name in returns.columns.astype(str):
        build_metrics = _metrics(returns.loc[build, name], returns.loc[build, name].ne(0.0))
        selection_metrics = _metrics(
            returns.loc[selection, name], returns.loc[selection, name].ne(0.0)
        )
        rows.append(
            {
                "sleeve": name,
                **{f"build_{key}": value for key, value in build_metrics.items()},
                **{f"selection_{key}": value for key, value in selection_metrics.items()},
            }
        )
    summary = pd.DataFrame(rows)
    summary["robust_sharpe"] = summary[["build_sharpe", "selection_sharpe"]].min(axis=1)
    ranked = summary.loc[
        summary["build_annualized_return"].gt(0.0)
        & summary["selection_annualized_return"].gt(0.0)
        & summary["robust_sharpe"].ge(minimum_robust_sharpe)
    ].sort_values("robust_sharpe", ascending=False)
    selected: list[str] = []
    early = returns.loc[build | selection]
    for name in ranked["sleeve"].astype(str):
        if selected:
            correlation = early[selected].corrwith(early[name]).abs().max()
            if pd.notna(correlation) and float(correlation) > maximum_correlation:
                continue
        selected.append(name)
        if len(selected) >= target:
            break
    if len(selected) < 3:
        raise ValueError(f"only {len(selected)} robust parameter ensembles")
    return summary, selected


def fixed_allocations(
    returns: pd.DataFrame, selection_mask: np.ndarray[Any, np.dtype[np.bool_]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit equal, inverse-volatility, and HRP weights on selection only."""
    names = returns.columns.astype(str).tolist()
    allocation = returns.loc[selection_mask, names]
    equal = pd.Series(1.0 / len(names), index=names)
    inverse = 1.0 / allocation.std().replace(0.0, np.nan)
    inverse = inverse.fillna(0.0).div(inverse.fillna(0.0).sum())
    weights: dict[str, pd.Series] = {"equal_weight": equal, "inverse_volatility": inverse}
    if len(names) >= 2:
        hrp = pd.Series(HRPOpt(allocation).optimize(), dtype=float).reindex(names).fillna(0.0)
        weights["hrp"] = hrp.div(hrp.sum())
    weight_frame = pd.DataFrame(weights)
    portfolios = pd.DataFrame(
        {name: returns.mul(weight, axis=1).sum(axis=1) for name, weight in weights.items()}
    )
    return portfolios, weight_frame


def vectorbt_replay(
    frame: pd.DataFrame,
    specs: list[DailySessionSpec],
    scores: dict[tuple[str, int], pd.DataFrame],
    *,
    one_way_cost_bps: float,
) -> pd.DataFrame:
    """Independently replay selected strategies on explicit session open/close events."""
    output: dict[str, pd.Series] = {}
    for spec in specs:
        product = frame.loc[frame["symbol"].eq(spec.symbol)].sort_values("trading_date")
        observed = scores[(spec.family, spec.lookback_sessions)][spec.symbol]
        direction = np.sign(observed).where(observed.abs().ge(spec.entry_threshold), 0.0).shift(1)
        direction = direction.reindex(pd.DatetimeIndex(product["trading_date"])).fillna(0.0)
        event_index = pd.DatetimeIndex(
            np.ravel(
                np.column_stack(
                    [
                        product["session_open_utc"].to_numpy(),
                        product["session_close_utc"].to_numpy(),
                    ]
                )
            )
        )
        prices = pd.Series(
            np.ravel(np.column_stack([product["open"].to_numpy(), product["close"].to_numpy()])),
            index=event_index,
            dtype=float,
        )
        long_entries = pd.Series(False, index=event_index)
        short_entries = pd.Series(False, index=event_index)
        exits = pd.Series(False, index=event_index)
        long_entries.iloc[0::2] = direction.gt(0.0).to_numpy()
        short_entries.iloc[0::2] = direction.lt(0.0).to_numpy()
        exits.iloc[1::2] = True
        portfolio = vbt.Portfolio.from_signals(
            prices,
            long_entries,
            exits,
            short_entries=short_entries,
            short_exits=exits,
            size=np.inf,
            fees=one_way_cost_bps / 10_000.0,
            slippage=0.0,
            init_cash=1.0,
            freq="1min",
        )
        event_returns = portfolio.returns()
        dates = pd.Series(np.repeat(product["trading_date"].to_numpy(), 2), index=event_index)
        output[spec.name] = (1.0 + event_returns).groupby(dates).prod().sub(1.0)
    return pd.DataFrame(output).sort_index()


def stable_candidate_selection(
    specs: list[DailySessionSpec], returns: pd.DataFrame, *, folds: int = 4
) -> tuple[list[str], pd.DataFrame]:
    """Choose one rule per product only when every chronological fold is positive."""
    blocks = np.array_split(np.arange(len(returns)), folds)
    spec_by_name = {spec.name: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    for name in returns.columns.astype(str):
        fold_sharpes = [sharpe_ratio(returns[name].iloc[block]) for block in blocks]
        spec = spec_by_name[name]
        rows.append(
            {
                **asdict(spec),
                "full_sharpe": sharpe_ratio(returns[name]),
                "positive_folds": sum(value > 0.0 for value in fold_sharpes),
                "worst_fold_sharpe": min(fold_sharpes),
                **{f"fold_{number + 1}_sharpe": value for number, value in enumerate(fold_sharpes)},
            }
        )
    audit = pd.DataFrame(rows)
    stable = audit.loc[
        audit["positive_folds"].eq(folds)
        & audit["full_sharpe"].ge(1.0)
        & audit["worst_fold_sharpe"].gt(0.0)
    ]
    selected = (
        stable.sort_values("full_sharpe", ascending=False)
        .drop_duplicates("symbol")
        .head(6)["name"]
        .astype(str)
        .tolist()
    )
    if len(selected) < 3:
        raise ValueError(f"only {len(selected)} products have all-fold-stable strategies")
    return selected, audit


def capped_inverse_volatility_weights(
    returns: pd.DataFrame, *, maximum_weight: float = 0.50
) -> pd.Series:
    """Fit inverse-volatility weights and redistribute any concentration excess."""
    raw = 1.0 / returns.std().replace(0.0, np.nan)
    weights = raw.fillna(0.0).div(raw.fillna(0.0).sum())
    for _ in range(len(weights)):
        over = weights.gt(maximum_weight)
        if not over.any():
            break
        excess = float((weights[over] - maximum_weight).sum())
        weights.loc[over] = maximum_weight
        under = ~over
        if weights.loc[under].sum() <= 0.0:
            break
        weights.loc[under] += excess * weights.loc[under] / weights.loc[under].sum()
    return pd.Series(weights.div(weights.sum()), dtype=float)


def block_bootstrap_summary(
    returns: pd.Series, *, block_length: int = 20, samples: int = 2000, seed: int = 20260823
) -> dict[str, float]:
    """Estimate uncertainty with deterministic moving-block resampling."""
    values = returns.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sharpes = np.empty(samples)
    annual_returns = np.empty(samples)
    starts = np.arange(max(1, len(values) - block_length + 1))
    blocks_needed = int(np.ceil(len(values) / block_length))
    for sample in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        boot = np.concatenate([values[start : start + block_length] for start in chosen])[
            : len(values)
        ]
        annual_returns[sample] = boot.mean() * 252.0
        standard_deviation = boot.std(ddof=1)
        sharpes[sample] = (
            boot.mean() / standard_deviation * np.sqrt(252.0)
            if standard_deviation > 0.0
            else np.nan
        )
    return {
        "annualized_return_p05": float(np.nanquantile(annual_returns, 0.05)),
        "annualized_return_p50": float(np.nanquantile(annual_returns, 0.50)),
        "annualized_return_p95": float(np.nanquantile(annual_returns, 0.95)),
        "sharpe_p05": float(np.nanquantile(sharpes, 0.05)),
        "sharpe_p50": float(np.nanquantile(sharpes, 0.50)),
        "sharpe_p95": float(np.nanquantile(sharpes, 0.95)),
    }


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    """Run the roll-safe daily-session discovery and audit workflow."""
    started = perf_counter()
    frame = load_development_panel(project_root)
    panels = session_panels(frame)
    session_returns = panels["session_return"]
    specs = generate_population(session_returns.columns.astype(str).tolist())
    scores = score_panels(session_returns)
    net, gross, entries = simulate_population(
        specs, session_returns, scores, one_way_cost_bps=one_way_cost_bps
    )
    windows = make_research_windows(pd.DatetimeIndex(session_returns.index))
    summary = summarize_population(specs, net, entries, windows)
    index = pd.DatetimeIndex(net.index)
    selection_mask = (index >= pd.Timestamp(windows.selection_start)) & (
        index <= pd.Timestamp(windows.selection_end)
    )
    selected = select_complementary(summary, net, selection_mask)
    selected_specs = [spec for spec in specs if spec.name in selected]
    individual_portfolios, individual_weights = fixed_allocations(net[selected], selection_mask)
    ensemble_fast, ensemble_members = build_parameter_ensembles(summary, net)
    ensemble_summary, selected_sleeves = summarize_and_select_ensembles(ensemble_fast, windows)
    selected_ensemble_returns = ensemble_fast[selected_sleeves]
    ensemble_portfolios, ensemble_weights = fixed_allocations(
        selected_ensemble_returns, selection_mask
    )
    portfolios = pd.concat(
        [
            individual_portfolios.add_prefix("individual_"),
            ensemble_portfolios.add_prefix("ensemble_"),
        ],
        axis=1,
    )
    primary = "ensemble_hrp" if "ensemble_hrp" in portfolios else "ensemble_inverse_volatility"
    confirmation_mask = index >= pd.Timestamp(windows.portfolio_test_start)

    vectorbt_returns = vectorbt_replay(
        frame, selected_specs, scores, one_way_cost_bps=one_way_cost_bps
    ).reindex(index)
    replay_difference = net[selected].sub(vectorbt_returns[selected]).abs().max().max()

    portfolio_rows: list[dict[str, Any]] = []
    for raw_name in portfolios:
        name = str(raw_name)
        snapshot, _ = performance_snapshot(portfolios.loc[confirmation_mask, name])
        portfolio_rows.append({"portfolio": name, **snapshot})
    portfolio_summary = pd.DataFrame(portfolio_rows)

    primary_weight_column = primary.removeprefix("ensemble_")
    primary_weights = ensemble_weights[primary_weight_column]
    stress_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5):
        eligible_names = sorted(
            {name for sleeve in selected_sleeves for name in ensemble_members[sleeve]}
        )
        stressed = gross[eligible_names] - entries[eligible_names] * 2.0 * cost / 10_000.0
        stressed_sleeves = pd.DataFrame(
            {sleeve: stressed[ensemble_members[sleeve]].mean(axis=1) for sleeve in selected_sleeves}
        )
        portfolio_return = stressed_sleeves.mul(primary_weights, axis=1).sum(axis=1)
        snapshot, _ = performance_snapshot(portfolio_return.loc[confirmation_mask])
        stress_rows.append({"one_way_cost_bps": cost, **snapshot})

    stable_names, temporal_audit = stable_candidate_selection(specs, net)
    stable_specs = [spec for spec in specs if spec.name in stable_names]
    stable_weights = capped_inverse_volatility_weights(net[stable_names])
    stable_vectorbt = vectorbt_replay(
        frame, stable_specs, scores, one_way_cost_bps=one_way_cost_bps
    ).reindex(index)
    stable_engine_difference = (
        net[stable_names].sub(stable_vectorbt[stable_names]).abs().max().max()
    )
    stable_unlevered = pd.Series(
        stable_vectorbt.mul(stable_weights, axis=1).sum(axis=1), dtype=float
    )
    stable_volatility = float(stable_unlevered.std(ddof=1) * np.sqrt(252.0))
    stable_leverage = min(2.0, 0.10 / stable_volatility) if stable_volatility > 0.0 else 1.0
    stable_portfolio = stable_unlevered * stable_leverage
    stable_snapshot, _ = performance_snapshot(stable_portfolio)
    stable_confirmation, _ = performance_snapshot(stable_portfolio.loc[confirmation_mask])
    stable_fold_rows: list[dict[str, Any]] = []
    for fold_number, block in enumerate(np.array_split(np.arange(len(index)), 4), start=1):
        snapshot, _ = performance_snapshot(stable_portfolio.iloc[block])
        stable_fold_rows.append({"fold": fold_number, **snapshot})
    stable_cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5):
        replay = vectorbt_replay(frame, stable_specs, scores, one_way_cost_bps=cost).reindex(index)
        stable_stressed = pd.Series(
            replay.mul(stable_weights, axis=1).sum(axis=1) * stable_leverage,
            dtype=float,
        )
        snapshot, _ = performance_snapshot(stable_stressed)
        stable_cost_rows.append({"one_way_cost_bps": cost, **snapshot})

    lookbacks = [1, 2, 3, 5, 10, 20, 40, 60]
    thresholds = [0.0, 0.5, 1.0, 1.5]
    neighbor_rows: list[pd.DataFrame] = []
    for spec in stable_specs:
        lookback_position = lookbacks.index(spec.lookback_sessions)
        threshold_position = thresholds.index(spec.entry_threshold)
        neighbors = temporal_audit.loc[
            temporal_audit["symbol"].eq(spec.symbol)
            & temporal_audit["family"].eq(spec.family)
            & temporal_audit["lookback_sessions"]
            .map(lookbacks.index)
            .sub(lookback_position)
            .abs()
            .le(1)
            & temporal_audit["entry_threshold"]
            .map(thresholds.index)
            .sub(threshold_position)
            .abs()
            .le(1)
            & ~temporal_audit["name"].eq(spec.name)
        ].copy()
        neighbor_rows.append(neighbors.assign(selected_strategy=spec.name))
    neighbor_audit = pd.concat(neighbor_rows, ignore_index=True)
    neighbor_support = neighbor_audit.groupby("selected_strategy")["full_sharpe"].apply(
        lambda values: int((values > 0.0).sum())
    )
    bootstrap = block_bootstrap_summary(stable_portfolio)
    benchmark_path = project_root / "data/processed/strategy_factory/sp500_benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark_total = float(benchmark["total_return"])
    stable_cost_frame = pd.DataFrame(stable_cost_rows)
    cost_35_sharpe = float(
        stable_cost_frame.loc[stable_cost_frame["one_way_cost_bps"].eq(3.5), "sharpe"].iloc[0]
    )
    audit_gates = {
        "full_development_sharpe_at_least_2": float(stable_snapshot["sharpe"]) >= 2.0,
        "every_chronological_fold_positive": all(
            float(row["sharpe"]) > 0.0 for row in stable_fold_rows
        ),
        "winning_months_at_least_60_percent": float(stable_snapshot["win_months"]) >= 0.60,
        "winning_quarters_at_least_50_percent": (float(stable_snapshot["win_quarters"]) >= 0.50),
        "sharpe_above_1_5_at_3_5_bps": cost_35_sharpe >= 1.5,
        "beats_sp500_in_confirmation": (
            float(stable_confirmation["total_return"]) >= benchmark_total
        ),
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
        "vectorbt_daily_return_difference_below_5e_5": float(stable_engine_difference) < 5e-5,
        "at_least_two_positive_parameter_neighbors_each": bool((neighbor_support >= 2).all()),
        "maximum_strategy_weight_at_most_50_percent": float(stable_weights.max()) <= 0.50 + 1e-12,
    }
    audit_passed = all(audit_gates.values())

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([asdict(spec) for spec in specs]).to_csv(
        output / "strategy_registry.csv", index=False
    )
    summary.to_parquet(output / "build_selection_summary.parquet", index=False)
    summary.loc[summary["name"].isin(selected)].to_csv(
        output / "selected_strategies.csv", index=False
    )
    net[selected].to_parquet(output / "selected_returns_fast_engine.parquet")
    vectorbt_returns[selected].to_parquet(output / "selected_returns_vectorbt.parquet")
    portfolios.to_parquet(output / "portfolio_returns.parquet")
    individual_weights.to_csv(output / "individual_portfolio_weights.csv")
    ensemble_weights.to_csv(output / "ensemble_portfolio_weights.csv")
    (output / "ensemble_members.json").write_text(
        json.dumps(ensemble_members, indent=2), encoding="utf-8"
    )
    ensemble_summary.to_csv(output / "ensemble_build_selection_summary.csv", index=False)
    (output / "selected_ensemble_sleeves.json").write_text(
        json.dumps(selected_sleeves, indent=2), encoding="utf-8"
    )
    portfolio_summary.to_csv(output / "portfolio_confirmation_summary.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "fixed_weight_cost_stress.csv", index=False)
    temporal_audit.to_parquet(output / "all_strategy_temporal_stability.parquet", index=False)
    stable_vectorbt.to_parquet(output / "stable_candidate_vectorbt_returns.parquet")
    pd.DataFrame({"candidate_return": stable_portfolio}).to_parquet(
        output / "stable_candidate_portfolio_returns.parquet"
    )
    stable_weights.to_csv(output / "stable_candidate_weights.csv", header=["weight"])
    pd.DataFrame(stable_fold_rows).to_csv(output / "stable_candidate_fold_audit.csv", index=False)
    stable_cost_frame.to_csv(output / "stable_candidate_cost_stress.csv", index=False)
    neighbor_audit.to_csv(output / "stable_candidate_neighbor_audit.csv", index=False)
    (output / "stable_candidate_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )
    (output / "stable_candidate_audit_gates.json").write_text(
        json.dumps(audit_gates, indent=2), encoding="utf-8"
    )
    report_status = _write_quantstats_report(
        portfolios.loc[confirmation_mask, primary],
        output / "tear_sheet.html",
        f"Daily Session Factory: {primary}",
    )
    stable_report_status = _write_quantstats_report(
        stable_portfolio,
        output / "stable_candidate_tear_sheet.html",
        "Stable Daily Session Candidate",
    )
    manifest = {
        "research_stage": "development_only",
        "sealed_year_accessed": False,
        "generated_strategies": len(specs),
        "selected_strategies": len(selected),
        "selected_parameter_ensembles": selected_sleeves,
        "windows": asdict(windows),
        "signal_timing": "session d close",
        "execution": "session d+1 open to session d+1 close",
        "overnight_exposure": False,
        "roll_gap_exposure": False,
        "one_way_cost_bps": one_way_cost_bps,
        "primary_portfolio_predeclared": primary,
        "maximum_per_symbol": 2,
        "maximum_selection_correlation": 0.65,
        "fast_vs_vectorbt_max_daily_return_difference": float(replay_difference),
        "stable_candidate": {
            "strategies": stable_names,
            "weights": stable_weights.to_dict(),
            "fixed_leverage": stable_leverage,
            "target_annualized_volatility": 0.10,
            "full_development": stable_snapshot,
            "confirmation": stable_confirmation,
            "benchmark_confirmation_total_return": benchmark_total,
            "fast_vs_vectorbt_max_daily_return_difference": float(stable_engine_difference),
            "audit_gates": audit_gates,
            "audit_passed": audit_passed,
            "tear_sheet_status": stable_report_status,
        },
        "vectorbt_version": vbt.__version__,
        "tear_sheet_status": report_status,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (project_root / OUTPUT_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "path": str(output)}, indent=2), encoding="utf-8"
    )
    print(portfolio_summary.to_string(index=False))
    print(pd.DataFrame(stress_rows).to_string(index=False))
    print(f"Fast/VectorBT max difference: {replay_difference:.12g}")
    print("\nStable candidate")
    print(pd.DataFrame([stable_snapshot]).to_string(index=False))
    print(pd.DataFrame(stable_fold_rows).to_string(index=False))
    print(stable_cost_frame.to_string(index=False))
    print(json.dumps(audit_gates, indent=2))
    print(f"Saved run: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run roll-safe daily session strategy search")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    args = parser.parse_args()
    run_factory(args.project_root.resolve(), one_way_cost_bps=args.one_way_cost_bps)


if __name__ == "__main__":
    main()
