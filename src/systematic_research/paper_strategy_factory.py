"""Development-only search over economically motivated, paper-backed strategy blocks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt  # type: ignore[import-untyped]

from systematic_research.broad_strategy_search import _write_quantstats_report
from systematic_research.intraday_strategy_search import _panels
from systematic_research.saved_strategy_runner import performance_snapshot
from systematic_research.strategy_factory import (
    ProductStrategy,
    ResearchWindows,
    build_product_score,
    entry_eligibility,
    make_research_windows,
)
from systematic_research.vectorbt_strategy_factory import (
    OUTPUT_ROOT,
    _source_fingerprint,
    _summarize_strategies,
    cached_resampled_bars,
    load_development_minutes,
    optimized_portfolios,
    run_vectorbt_batch,
)

PAPER_OUTPUT_ROOT = OUTPUT_ROOT / "paper_runs"

ECONOMIC_PEERS = {
    "6E": "6J",
    "6J": "6E",
    "CL": "NG",
    "NG": "CL",
    "ES": "NKD",
    "NKD": "ES",
    "GC": "HG",
    "HG": "GC",
    "ZC": "CL",
    "ZN": "ES",
}

PAPER_HYPOTHESES: dict[str, dict[str, str]] = {
    "momentum": {
        "hypothesis": "intraday_time_series_momentum",
        "source": "Gao, Han, Li and Zhou (2018), Market Intraday Momentum",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866",
        "adaptation": "standardized trailing within-session return predicts the next bar",
    },
    "volatility_scaled_momentum": {
        "hypothesis": "volatility_managed_momentum",
        "source": "Moreira and Muir (2017), Volatility-Managed Portfolios",
        "url": "https://www.nber.org/papers/w22208",
        "adaptation": "within-session momentum divided by lagged realized volatility",
    },
    "mean_reversion": {
        "hypothesis": "short_horizon_reversal",
        "source": "Dai, Medhat, Novy-Marx and Rizova (2024), Reversals and Liquidity Provision",
        "url": "https://www.nber.org/papers/w30917",
        "adaptation": "contrarian response to standardized short-horizon product returns",
    },
    "cross_sectional_momentum": {
        "hypothesis": "cross_asset_momentum",
        "source": "Asness, Moskowitz and Pedersen (2013), Value and Momentum Everywhere",
        "url": "https://papers.ssrn.com/abstract=2174501",
        "adaptation": "rank products by contemporaneous trailing within-session return",
    },
    "cross_sectional_reversal": {
        "hypothesis": "cross_asset_short_reversal",
        "source": "Dai, Medhat, Novy-Marx and Rizova (2024), Reversals and Liquidity Provision",
        "url": "https://www.nber.org/papers/w30917",
        "adaptation": "buy relative laggards and sell leaders over short intraday windows",
    },
    "residual_reversion": {
        "hypothesis": "relative_value_convergence",
        "source": "Gatev, Goetzmann and Rouwenhorst (2006), Pairs Trading",
        "url": "https://papers.ssrn.com/abstract=141615",
        "adaptation": "fade own return residual versus an equal-weight peer basket",
    },
    "lead_lag": {
        "hypothesis": "cross_market_lead_lag",
        "source": "Asness, Moskowitz and Pedersen (2013), Value and Momentum Everywhere",
        "url": "https://papers.ssrn.com/abstract=2174501",
        "adaptation": "lagged peer-basket momentum predicts the product's next bar",
    },
    "volume_confirmation": {
        "hypothesis": "high_attention_intraday_momentum",
        "source": "Gao, Han, Li and Zhou (2018), Market Intraday Momentum",
        "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866",
        "adaptation": "interact intraday momentum with a causal volume surprise",
    },
    "intraday_close_momentum": {
        "hypothesis": "rest_of_day_predicts_final_interval",
        "source": "Baltussen et al. (2021), Hedging Demand and Market Intraday Momentum",
        "url": "https://doi.org/10.1016/j.jfineco.2021.05.045",
        "adaptation": (
            "late-session continuation: signal three bars from session end "
            "and hold the next full bar"
        ),
    },
    "opening_range_continuation": {
        "hypothesis": "opening_range_breakout",
        "source": "Holmberg, Lonnback and Lundstrom (2012), Opening Range Breakouts",
        "url": "https://doi.org/10.1016/j.frl.2012.03.005",
        "adaptation": "trade in the direction of the standardized opening-session return",
    },
    "opening_range_reversal": {
        "hypothesis": "opening_price_pressure_reversal",
        "source": "Dai, Medhat, Novy-Marx and Rizova (2024), Reversals and Liquidity Provision",
        "url": "https://www.nber.org/papers/w30917",
        "adaptation": "fade unusually large opening-session returns",
    },
    "session_vwap_reversion": {
        "hypothesis": "price_volume_liquidity_reversal",
        "source": "Campbell, Grossman and Wang (1993), Trading Volume and Serial Correlation",
        "url": "https://doi.org/10.2307/2118462",
        "adaptation": "fade standardized price deviation from causal session VWAP",
    },
    "range_breakout": {
        "hypothesis": "intraday_range_breakout",
        "source": "Holmberg, Lonnback and Lundstrom (2012), Opening Range Breakouts",
        "url": "https://doi.org/10.1016/j.frl.2012.03.005",
        "adaptation": "trade closes beyond the prior within-session high-low range",
    },
    "range_expansion_reversal": {
        "hypothesis": "high_volatility_liquidity_reversal",
        "source": "Dai, Medhat, Novy-Marx and Rizova (2024), Reversals and Liquidity Provision",
        "url": "https://www.nber.org/papers/w30917",
        "adaptation": "fade signed returns when true range expands relative to lagged ATR",
    },
    "paired_spread_reversion": {
        "hypothesis": "economic_pair_convergence",
        "source": "Gatev, Goetzmann and Rouwenhorst (2006), Pairs Trading",
        "url": "https://papers.ssrn.com/abstract=141615",
        "adaptation": "fade within-session cumulative-return spreads for economic peers",
    },
    "regime_switch": {
        "hypothesis": "volatility_conditioned_momentum_reversal",
        "source": "Moreira and Muir (2017); Dai et al. (2024)",
        "url": "https://www.nber.org/papers/w22208",
        "adaptation": "follow trends below lagged median volatility and fade above it",
    },
}

SECOND_WAVE_FAMILIES = (
    "opening_range_continuation",
    "opening_range_reversal",
    "session_vwap_reversion",
    "range_breakout",
    "range_expansion_reversal",
    "paired_spread_reversion",
    "regime_switch",
)


def generate_paper_population(
    symbols: list[str],
    *,
    per_symbol: int = 80,
    seed: int = 20260823,
    families_override: tuple[str, ...] | None = None,
) -> list[ProductStrategy]:
    """Create deterministic variants while preserving each hypothesis' economic direction."""
    rng = np.random.default_rng(seed)
    families = families_override or tuple(PAPER_HYPOTHESES)
    population: list[ProductStrategy] = []
    for symbol in sorted(symbols):
        candidates: list[tuple[Any, ...]] = []
        for family in families:
            for timeframe in (15, 30, 60, 240):
                for lookback in (2, 4, 8, 16):
                    for threshold in (0.5, 0.75, 1.0, 1.25):
                        candidates.append(
                            (
                                family,
                                timeframe,
                                lookback,
                                threshold,
                                float(rng.choice([0.0, 0.10, 0.25])),
                                int(rng.choice([2, 4, 8, 16])),
                                str(rng.choice(["all", "below_median", "above_median"])),
                                str(rng.choice(["all", "us_core", "overnight"])),
                                float(rng.choice([0.005, 0.010, 0.020])),
                                float(rng.choice([0.005, 0.010, 0.020, 0.040])),
                            )
                        )
        chosen: list[tuple[Any, ...]] = []
        quota, remainder = divmod(per_symbol, len(families))
        for family_number, family in enumerate(families):
            family_candidates = [row for row in candidates if row[0] == family]
            take = quota + (1 if family_number < remainder else 0)
            order = rng.permutation(len(family_candidates))[:take]
            chosen.extend(family_candidates[int(index)] for index in order)
        for values in chosen:
            time_rule = "preclose" if values[0] == "intraday_close_momentum" else values[7]
            population.append(
                ProductStrategy(
                    name=f"paper_{len(population) + 1:05d}",
                    symbol=symbol,
                    timeframe_minutes=values[1],
                    signal_family=values[0],
                    lookback_bars=values[2],
                    direction=1,
                    entry_threshold=values[3],
                    exit_threshold=values[4],
                    maximum_hold_bars=values[5],
                    volatility_filter=values[6],
                    time_rule=time_rule,
                    stop_loss=values[8],
                    profit_target=values[9],
                )
            )
    return population


def select_development_candidates(summary: pd.DataFrame, target: int = 100) -> list[str]:
    """Rank only on build/selection; retain diagnostics when strict qualifiers are scarce."""
    ranked = summary.copy()
    ranked["robust_score"] = ranked[["build_sharpe", "selection_sharpe"]].min(axis=1)
    ranked["robust_score"] += 0.25 * ranked[["build_sharpe", "selection_sharpe"]].mean(axis=1)
    qualified = ranked.loc[
        ranked["build_annualized_return"].gt(0)
        & ranked["selection_annualized_return"].gt(0)
        & ranked["build_trades"].ge(3)
        & ranked["selection_trades"].ge(3)
    ].nlargest(target, "robust_score")
    minimum_diagnostic_set = min(20, target, len(ranked))
    if len(qualified) < minimum_diagnostic_set:
        supplements = ranked.loc[~ranked["strategy"].isin(qualified["strategy"])].nlargest(
            minimum_diagnostic_set - len(qualified), "robust_score"
        )
        qualified = pd.concat([qualified, supplements], ignore_index=True)
    return qualified["strategy"].astype(str).tolist()


def _prepare_caches(
    bars: pd.DataFrame, specs: list[ProductStrategy]
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]]:
    close, returns, _ = _panels(bars)
    high = bars.pivot(index="timestamp_utc", columns="symbol", values="high").reindex_like(close)
    low = bars.pivot(index="timestamp_utc", columns="symbol", values="low").reindex_like(close)
    open_ = bars.pivot(index="timestamp_utc", columns="symbol", values="open").reindex_like(close)
    volume = bars.pivot(index="timestamp_utc", columns="symbol", values="volume").reindex_like(
        close
    )
    date_panel = bars.pivot(
        index="timestamp_utc", columns="symbol", values="trading_date"
    ).reindex_like(close)
    scores: dict[tuple[str, int], pd.DataFrame] = {}
    eligibility: dict[tuple[str, str], pd.DataFrame] = {}
    for spec in specs:
        score_key = (spec.signal_family, spec.lookback_bars)
        if score_key not in scores:
            if spec.signal_family == "intraday_close_momentum":
                cumulative = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
                for symbol in returns.columns:
                    cumulative[symbol] = returns[symbol].groupby(date_panel[symbol]).cumsum()
                prior_mean = cumulative.rolling(160, min_periods=40).mean().shift(1)
                prior_std = cumulative.rolling(160, min_periods=40).std().shift(1)
                scores[score_key] = cumulative.sub(prior_mean).div(prior_std.replace(0.0, np.nan))
            elif spec.signal_family in SECOND_WAVE_FAMILIES:
                raw = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
                for symbol in returns.columns:
                    product_dates = date_panel[symbol]
                    product_returns = returns[symbol]
                    if spec.signal_family.startswith("opening_range_"):
                        bar_opening_return = close[symbol].div(open_[symbol]).sub(1.0)
                        daily_opening = bar_opening_return.groupby(product_dates).first()
                        prior_scale = daily_opening.rolling(40, min_periods=20).std().shift(1)
                        daily_score = daily_opening.div(prior_scale.replace(0.0, np.nan))
                        raw[symbol] = product_dates.map(daily_score)
                        if spec.signal_family == "opening_range_reversal":
                            raw[symbol] *= -1.0
                    elif spec.signal_family == "session_vwap_reversion":
                        typical = (high[symbol] + low[symbol] + close[symbol]) / 3.0
                        cumulative_dollars = (
                            (typical * volume[symbol]).groupby(product_dates).cumsum()
                        )
                        cumulative_volume = volume[symbol].groupby(product_dates).cumsum()
                        vwap = cumulative_dollars.div(cumulative_volume.replace(0.0, np.nan))
                        raw[symbol] = -close[symbol].div(vwap).sub(1.0)
                    elif spec.signal_family == "range_breakout":
                        lookback = spec.lookback_bars
                        prior_high = (
                            high[symbol]
                            .groupby(product_dates)
                            .transform(
                                lambda values, window=lookback: (
                                    values.rolling(window).max().shift(1)
                                )
                            )
                        )
                        prior_low = (
                            low[symbol]
                            .groupby(product_dates)
                            .transform(
                                lambda values, window=lookback: (
                                    values.rolling(window).min().shift(1)
                                )
                            )
                        )
                        midpoint = (prior_high + prior_low) / 2.0
                        half_range = (prior_high - prior_low) / 2.0
                        raw[symbol] = (
                            close[symbol].sub(midpoint).div(half_range.replace(0.0, np.nan))
                        )
                    elif spec.signal_family == "range_expansion_reversal":
                        true_range = high[symbol].sub(low[symbol]).div(close[symbol])
                        prior_atr = true_range.rolling(spec.lookback_bars * 4).mean().shift(1)
                        raw[symbol] = -product_returns * true_range.div(
                            prior_atr.replace(0.0, np.nan)
                        )
                    elif spec.signal_family == "paired_spread_reversion":
                        peer = ECONOMIC_PEERS.get(str(symbol))
                        if peer not in returns:
                            continue
                        own_cumulative = product_returns.groupby(product_dates).cumsum()
                        peer_cumulative = returns[peer].groupby(date_panel[peer]).cumsum()
                        raw[symbol] = -(own_cumulative - peer_cumulative)
                    else:
                        momentum = product_returns.rolling(
                            spec.lookback_bars, min_periods=spec.lookback_bars
                        ).sum()
                        volatility = product_returns.rolling(40, min_periods=20).std().shift(1)
                        median = volatility.rolling(160, min_periods=40).median().shift(1)
                        direction = pd.Series(
                            np.where(volatility.le(median), 1.0, -1.0), index=returns.index
                        )
                        raw[symbol] = momentum * direction
                prior_mean = raw.rolling(160, min_periods=40).mean().shift(1)
                prior_std = raw.rolling(160, min_periods=40).std().shift(1)
                scores[score_key] = raw.sub(prior_mean).div(prior_std.replace(0.0, np.nan))
            else:
                scores[score_key] = build_product_score(
                    close, high, low, volume, returns, spec.signal_family, spec.lookback_bars
                )
        eligibility_key = (spec.volatility_filter, spec.time_rule)
        if eligibility_key not in eligibility:
            base_time_rule = "all" if spec.time_rule == "preclose" else spec.time_rule
            allowed = entry_eligibility(returns, spec.volatility_filter, base_time_rule)
            if spec.time_rule == "preclose":
                preclose = pd.DataFrame(False, index=returns.index, columns=returns.columns)
                for symbol in returns.columns:
                    valid_dates = date_panel[symbol].dropna()
                    is_preclose_signal = (
                        valid_dates.eq(valid_dates.shift(-1))
                        & valid_dates.eq(valid_dates.shift(-2))
                        & ~valid_dates.eq(valid_dates.shift(-3))
                    )
                    preclose.loc[is_preclose_signal.index, symbol] = is_preclose_signal
                allowed &= preclose
            eligibility[eligibility_key] = allowed
    return scores, eligibility


def _execute(
    population: list[ProductStrategy],
    bars_by_timeframe: dict[int, pd.DataFrame],
    caches: dict[
        int,
        tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]],
    ],
    dates: pd.DatetimeIndex,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns_parts: list[pd.DataFrame] = []
    entries_parts: list[pd.DataFrame] = []
    records_parts: list[pd.DataFrame] = []
    for timeframe, bars in bars_by_timeframe.items():
        timeframe_specs = [spec for spec in population if spec.timeframe_minutes == timeframe]
        scores, eligibility = caches[timeframe]
        for symbol in sorted({spec.symbol for spec in timeframe_specs}):
            specs = [spec for spec in timeframe_specs if spec.symbol == symbol]
            daily_returns, daily_entries, records = run_vectorbt_batch(
                bars, specs, scores, eligibility, cost_bps=cost_bps
            )
            returns_parts.append(daily_returns)
            entries_parts.append(daily_entries)
            if not records.empty:
                records_parts.append(records)
    returns = pd.concat(returns_parts, axis=1).reindex(dates).fillna(0.0)
    entries = pd.concat(entries_parts, axis=1).reindex(dates).fillna(0.0)
    records = pd.concat(records_parts, ignore_index=True) if records_parts else pd.DataFrame()
    return returns, entries, records


def run_paper_factory(
    project_root: Path,
    *,
    per_symbol: int = 80,
    cost_bps: float = 3.5,
    seed: int = 20260823,
    second_wave_only: bool = False,
) -> Path:
    """Run the paper-backed search strictly on the development partition."""
    started = perf_counter()
    minute_data, source_paths = load_development_minutes(project_root)
    symbols = sorted(minute_data["symbol"].astype(str).unique())
    family_set = SECOND_WAVE_FAMILIES if second_wave_only else None
    population = generate_paper_population(
        symbols, per_symbol=per_symbol, seed=seed, families_override=family_set
    )
    registry = pd.DataFrame([asdict(spec) for spec in population])
    references = pd.DataFrame.from_dict(PAPER_HYPOTHESES, orient="index").rename_axis(
        "signal_family"
    )
    registry = registry.merge(references.reset_index(), on="signal_family", how="left")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    windows: ResearchWindows = make_research_windows(dates)
    bars_by_timeframe: dict[int, pd.DataFrame] = {}
    caches: dict[
        int,
        tuple[dict[tuple[str, int], pd.DataFrame], dict[tuple[str, str], pd.DataFrame]],
    ] = {}
    cache_hits: dict[str, bool] = {}
    for timeframe in (15, 30, 60, 240):
        bars, hit = cached_resampled_bars(project_root, minute_data, source_paths, timeframe)
        bars_by_timeframe[timeframe] = bars
        cache_hits[f"{timeframe}m"] = hit
        specs = [spec for spec in population if spec.timeframe_minutes == timeframe]
        caches[timeframe] = _prepare_caches(bars, specs)

    strategy_returns, strategy_entries, records = _execute(
        population, bars_by_timeframe, caches, dates, cost_bps
    )
    summary = _summarize_strategies(registry, strategy_returns, strategy_entries, windows)
    selected = select_development_candidates(summary)
    selected_returns = strategy_returns[selected]
    selection_mask = (dates >= pd.Timestamp(windows.selection_start)) & (
        dates <= pd.Timestamp(windows.selection_end)
    )
    portfolios, weights, optimizer_status = optimized_portfolios(
        selected_returns, summary, selection_mask
    )
    primary = "pypfopt_hrp" if "pypfopt_hrp" in portfolios else "equal_weight"
    test_mask = dates >= pd.Timestamp(windows.portfolio_test_start)
    portfolio_rows: list[dict[str, float | int | str]] = []
    for name in portfolios.columns.astype(str):
        snapshot, _ = performance_snapshot(portfolios.loc[test_mask, name])
        portfolio_rows.append({"portfolio": name, **snapshot})
    portfolio_summary = pd.DataFrame(portfolio_rows)

    primary_weights = weights[primary]
    stress_rows: list[dict[str, float | int | str]] = []
    selected_specs = [spec for spec in population if spec.name in selected]
    for stressed_cost in (0.0, 3.5, 7.0, 10.0):
        stressed_returns, _, _ = _execute(
            selected_specs, bars_by_timeframe, caches, dates, stressed_cost
        )
        portfolio_return = stressed_returns[selected].mul(primary_weights, axis=1).sum(axis=1)
        snapshot, _ = performance_snapshot(portfolio_return.loc[test_mask])
        stress_rows.append({"one_way_cost_bps": stressed_cost, **snapshot})

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / PAPER_OUTPUT_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    registry.to_csv(output / "paper_strategy_registry.csv", index=False)
    references.to_csv(output / "research_sources.csv")
    summary.to_parquet(output / "candidate_build_selection_summary.parquet", index=False)
    summary.loc[summary["strategy"].isin(selected)].to_csv(
        output / "selected_strategies.csv", index=False
    )
    selected_returns.to_parquet(output / "selected_strategy_returns.parquet")
    weights.to_csv(output / "portfolio_strategy_weights.csv")
    portfolios.to_parquet(output / "portfolio_returns.parquet")
    portfolio_summary.to_csv(output / "portfolio_summary.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(output / "fixed_portfolio_cost_stress.csv", index=False)
    if not records.empty:
        records.loc[records["Column"].isin(selected)].to_parquet(output / "trade_ledger.parquet")
    report_status = _write_quantstats_report(
        portfolios.loc[test_mask, primary],
        output / "tear_sheet.html",
        f"Paper-backed development portfolio: {primary}",
    )
    manifest = {
        "engine": "vectorbt.Portfolio.from_signals",
        "vectorbt_version": vbt.__version__,
        "research_stage": "development_only",
        "sealed_year_accessed": False,
        "paper_replication": False,
        "paper_hypotheses_are_adaptations": True,
        "products": symbols,
        "generated_strategies": len(population),
        "selected_strategies": len(selected),
        "primary_portfolio_predeclared": primary,
        "seed": seed,
        "per_symbol": per_symbol,
        "strategy_wave": "second_wave_only" if second_wave_only else "all_paper_families",
        "windows": asdict(windows),
        "one_way_all_in_cost_bps": cost_bps,
        "execution": "signal close; next actual product bar open; flat by trading date",
        "stop_reference": "fill_price",
        "unadjusted_roll_gap_policy": "no cross-trading-date positions or return signals",
        "source_fingerprint": _source_fingerprint(project_root, source_paths),
        "cache_hits": cache_hits,
        "optimizer_status": optimizer_status,
        "tear_sheet_status": report_status,
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (project_root / PAPER_OUTPUT_ROOT / "latest_run.json").write_text(
        json.dumps({"run_id": run_id, "path": str(output)}, indent=2), encoding="utf-8"
    )
    print(portfolio_summary.to_string(index=False))
    print(pd.DataFrame(stress_rows).to_string(index=False))
    print(f"Saved run: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-backed strategy factory")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--per-symbol", type=int, default=80)
    parser.add_argument("--cost-bps", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--second-wave-only", action="store_true")
    args = parser.parse_args()
    run_paper_factory(
        args.project_root.resolve(),
        per_symbol=args.per_symbol,
        cost_bps=args.cost_bps,
        seed=args.seed,
        second_wave_only=args.second_wave_only,
    )


if __name__ == "__main__":
    main()
