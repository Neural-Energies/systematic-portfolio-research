"""Session-scale dependency-network relative value for preregistered hypothesis H024."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from systematic_research.cot_hedging_pressure import sha256
from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.pca_residual_reversal import chained_selection, replay_selection
from systematic_research.unique_hypothesis_factory import extended_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/graph_laplacian_rv")


@dataclass(frozen=True)
class NetworkVariant:
    name: str
    correlation_window_sessions: int
    momentum_sessions: int
    entry_threshold: float
    holding_sessions: int


def session_panel(minute_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session open-to-close returns and per-product close pivots, development only."""
    sessions = (
        minute_data.sort_values(["symbol", "trading_date", "timestamp_utc"])
        .groupby(["symbol", "trading_date"], sort=True)
        .agg(session_open=("open", "first"), session_close=("close", "last"))
        .reset_index()
    )
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(sessions["trading_date"]).unique()))
    opens = (
        sessions.pivot(index="trading_date", columns="symbol", values="session_open")
        .reindex(daily_index)
        .sort_index(axis=1)
    )
    closes = (
        sessions.pivot(index="trading_date", columns="symbol", values="session_close")
        .reindex(daily_index)
        .sort_index(axis=1)
    )
    returns = closes.div(opens).sub(1.0)
    return returns, closes


def network_weights(correlations: pd.DataFrame, max_neighbors: int) -> pd.DataFrame:
    """Row-stochastic positive-part dependency weights; isolated nodes fall back to uniform."""
    values = correlations.to_numpy(dtype=float).copy()
    values[values < 0.0] = 0.0
    np.fill_diagonal(values, 0.0)
    if max_neighbors < values.shape[1]:
        for row_index in range(values.shape[0]):
            row = values[row_index]
            if np.count_nonzero(row) > max_neighbors:
                cutoff = np.partition(row, -max_neighbors)[-max_neighbors]
                row[row < cutoff] = 0.0
    totals = values.sum(axis=1)
    weights = np.empty_like(values)
    uniform_value = 1.0 / values.shape[1]
    for row_index, total in enumerate(totals):
        weights[row_index] = values[row_index] / total if total > 0.0 else uniform_value
    return pd.DataFrame(weights, index=correlations.index, columns=correlations.columns)


def residual_scores(
    returns: pd.DataFrame,
    correlation_window_sessions: int,
    momentum_sessions: int,
    max_neighbors: int,
) -> pd.DataFrame:
    """Causal dislocation scores: own standardized momentum minus network-implied momentum."""
    output = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)
    values = returns.to_numpy(dtype=float)
    for location in range(momentum_sessions, len(returns)):
        history_start = max(0, location - correlation_window_sessions)
        history = values[history_start:location]
        frame = pd.DataFrame(history, columns=returns.columns).dropna(axis=1, how="any")
        if len(frame) < max(30, correlation_window_sessions // 2):
            continue
        scale = frame.std(ddof=1).replace(0.0, np.nan)
        usable = list(scale.dropna().index)
        if len(usable) < 6:
            continue
        correlations = frame[usable].corr()
        weights = network_weights(correlations, max_neighbors).reindex(index=usable, columns=usable)
        window = values[max(0, location - momentum_sessions) : location]
        window_frame = pd.DataFrame(window, columns=returns.columns)[usable].dropna(how="any")
        if window_frame.empty:
            continue
        volatility = returns[usable].iloc[:location].tail(correlation_window_sessions).std(ddof=1)
        standardized = window_frame.div(volatility.replace(0.0, np.nan), axis=1)
        momentum = standardized.sum(axis=0)
        aligned_momentum = momentum.reindex(usable).to_numpy(dtype=float)
        implied = np.asarray(weights) @ aligned_momentum
        residual = aligned_momentum - implied
        dispersion = float(np.nanstd(residual, ddof=1))
        if not np.isfinite(dispersion) or dispersion <= 0.0:
            continue
        scored = pd.Series(residual / dispersion, index=usable)
        output.loc[returns.index[location], scored.index] = scored.to_numpy(dtype=float)
    return output


def balanced_positions(
    scores: pd.DataFrame, threshold: float, holding_sessions: int
) -> pd.DataFrame:
    """Dollar-neutral fade of extreme dislocations held as overlapping K-session cohorts."""
    entered = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for location, row in scores.iterrows():
        valid = row.dropna()
        if len(valid) < 6:
            continue
        longs = valid[valid <= -threshold]
        shorts = valid[valid >= threshold]
        if longs.empty or shorts.empty:
            continue
        entered.loc[location, longs.index] = -0.5 / len(longs)
        entered.loc[location, shorts.index] = 0.5 / len(shorts)
    cohorts = [entered.shift(steps) for steps in range(1, holding_sessions + 1)]
    stacked = pd.concat(cohorts).groupby(level=0).mean()
    return stacked.reindex(scores.index).fillna(0.0)


def build_population(
    returns: pd.DataFrame,
) -> tuple[list[NetworkVariant], pd.DataFrame, pd.DataFrame]:
    variants: list[NetworkVariant] = []
    gross_returns: dict[str, pd.Series] = {}
    trade_sides: dict[str, pd.Series] = {}
    for correlation_window in (40, 60):
        for momentum_sessions in (5, 10):
            scores = residual_scores(returns, correlation_window, momentum_sessions, 4)
            for threshold in (1.0, 1.5):
                for holding_sessions in (3, 5):
                    name = (
                        f"h024_w{correlation_window:02d}_m{momentum_sessions:02d}_"
                        f"t{str(threshold).replace('.', 'p')}_k{holding_sessions}"
                    )
                    variant = NetworkVariant(
                        name,
                        correlation_window,
                        momentum_sessions,
                        threshold,
                        holding_sessions,
                    )
                    variants.append(variant)
                    positions = balanced_positions(scores, threshold, holding_sessions)
                    gross = positions.mul(returns.fillna(0.0)).sum(axis=1)
                    previous = positions.shift()
                    previous.iloc[0] = 0.0
                    sides = positions.sub(previous).abs().sum(axis=1)
                    gross_returns[name] = gross
                    trade_sides[name] = sides
    return (
        variants,
        pd.DataFrame(gross_returns, index=returns.index),
        pd.DataFrame(trade_sides, index=returns.index),
    )


def outlier_removal_table(portfolio: pd.Series) -> pd.DataFrame:
    clean = portfolio.dropna()
    ordered = clean.sort_values(ascending=False)
    rows: list[dict[str, Any]] = []
    total_sum = float(clean.sum())
    for remove in (0, 1, 3, 5, 10):
        trimmed = clean.drop(ordered.index[:remove])
        sharpe = (
            float(trimmed.mean() / trimmed.std(ddof=1) * np.sqrt(252.0))
            if trimmed.std(ddof=1)
            else 0.0
        )
        rows.append(
            {
                "best_days_removed": remove,
                "total_return": float(
                    np.prod(1.0 + trimmed.to_numpy(dtype=float)) - 1.0
                ),
                "sharpe": sharpe,
                "share_of_net_sum_removed": float(ordered.iloc[:remove].sum() / total_sum)
                if total_sum > 0.0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def subperiod_table(portfolio: pd.Series, parts: int = 4) -> pd.DataFrame:
    clean = portfolio.dropna()
    blocks = np.array_split(np.arange(len(clean)), parts)
    rows = []
    for number, block in enumerate(blocks, start=1):
        segment = clean.iloc[block]
        sharpe = (
            float(segment.mean() / segment.std(ddof=1) * np.sqrt(252.0))
            if segment.std(ddof=1)
            else 0.0
        )
        rows.append(
            {
                "subperiod": number,
                "start": str(segment.index.min().date()),
                "end": str(segment.index.max().date()),
                "total_return": float((1.0 + segment).prod() - 1.0),
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    returns, _ = session_panel(minute_data)
    variants, gross, sides = build_population(returns)
    net = gross - sides * one_way_cost_bps / 10_000.0
    portfolio, selections = chained_selection(net, sides)
    snapshot = extended_snapshot(portfolio) if not portfolio.empty else {}
    blocks = np.array_split(np.arange(len(net)), 4)
    fold_rows = []
    for fold in range(2, 5):
        fold_return = portfolio.reindex(net.index[blocks[fold - 1]]).dropna()
        if not fold_return.empty:
            fold_rows.append({"fold": fold, **extended_snapshot(fold_return)})
    cost_rows = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed = replay_selection(gross - sides * cost / 10_000.0, selections)
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed)})
    bootstrap = (
        block_bootstrap_summary(portfolio, block_length=20, samples=5000)
        if not portfolio.empty
        else {}
    )
    removal = outlier_removal_table(portfolio) if not portfolio.empty else pd.DataFrame()
    periods = subperiod_table(portfolio) if not portfolio.empty else pd.DataFrame()
    audit_gates = {
        "selection_available_all_evaluation_folds": len(fold_rows) == 3,
        "gross_expectancy_positive": bool(len(cost_rows))
        and float(cost_rows[0]["annualized_return"]) > 0.0,
        "one_bp_expectancy_positive": bool(len(cost_rows))
        and float(cost_rows[2]["annualized_return"]) > 0.0,
        "walk_forward_sharpe_at_least_2": bool(snapshot) and float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": bool(snapshot)
        and max(float(snapshot["win_weeks"]), float(snapshot["win_months"])) >= 0.70,
        "all_evaluation_folds_positive": len(fold_rows) == 3
        and all(float(row["annualized_return"]) > 0.0 for row in fold_rows),
        "bootstrap_fifth_percentile_sharpe_positive": bool(bootstrap)
        and float(bootstrap["sharpe_p05"]) > 0.0,
        "positive_after_removing_best_ten_days": bool(not removal.empty)
        and float(removal["sharpe"].iloc[-1]) > 0.0,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True)
    pd.DataFrame([asdict(variant) for variant in variants]).to_csv(
        output / "strategy_registry.csv", index=False
    )
    gross.to_parquet(output / "all_strategy_gross_returns.parquet")
    sides.to_parquet(output / "all_strategy_trade_sides.parquet")
    net.to_parquet(output / "all_strategy_net_returns.parquet")
    portfolio.to_frame().to_parquet(output / "walk_forward_portfolio_returns.parquet")
    selections.to_csv(output / "walk_forward_selections.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    removal.to_csv(output / "outlier_removal.csv", index=False)
    periods.to_csv(output / "subperiod_audit.csv", index=False)
    manifest = {
        "hypothesis_id": "H024",
        "research_stage": "development_only_chained_walk_forward",
        "sealed_year_accessed": False,
        "unique_hypothesis_count": 1,
        "parameter_sensitivity_variants": len(variants),
        "base_one_way_cost_bps": one_way_cost_bps,
        "execution_model": (
            "signal from sessions <= t; overlapping 1..K session cohorts; "
            "costs charged on position transitions"
        ),
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "market_sources": [path.relative_to(project_root).as_posix() for path in market_sources],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    master = project_root / "saved_strategies" / f"hypothesis_H024_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/graph_laplacian_rv.py",
        "research_program/hypothesis_registry.csv",
        "research_program/RESEARCH_LOOP_PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
    ):
        source = project_root / relative
        shutil.copy2(source, source_snapshot / relative.replace("/", "__"))
    (master / "MASTER_INDEX.md").write_text(
        "\n".join(
            [
                "# H024 Dependency-Network Relative Value — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Parameter sensitivity variants: {len(variants)}",
                "- Sealed year accessed: False",
            ]
        ),
        encoding="utf-8",
    )
    checksum_rows = [
        {"path": path.relative_to(master).as_posix(), "sha256": sha256(path)}
        for path in sorted(item for item in master.rglob("*") if item.is_file())
    ]
    pd.DataFrame(checksum_rows).to_csv(master / "SHA256SUMS.csv", index=False)
    print(json.dumps(manifest, indent=2))
    print(f"Saved run: {output}")
    print(f"Saved master archive: {master}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H024 graph-Laplacian relative value")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
