"""Causal cross-futures PCA residual-reversal research for hypothesis H019."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.intraday_strategy_search import resample_bars
from systematic_research.metrics import sharpe_ratio
from systematic_research.unique_hypothesis_factory import (
    _intraday_trade_sides,
    extended_snapshot,
)
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/pca_residual_reversal")
TIMEFRAMES = (15, 30, 60, 240)


@dataclass(frozen=True)
class PcaVariant:
    name: str
    timeframe_minutes: int
    training_sessions: int
    residual_threshold: float
    confirmation_bars: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _balanced_tail_weights(residual_scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    signed = pd.DataFrame(
        np.sign(residual_scores.to_numpy(dtype=float)),
        index=residual_scores.index,
        columns=residual_scores.columns,
    )
    raw = -signed.where(residual_scores.abs().ge(threshold), 0.0)
    long_leg = raw.clip(lower=0.0)
    short_leg = -raw.clip(upper=0.0)
    long_count = long_leg.sum(axis=1)
    short_count = short_leg.sum(axis=1)
    valid = long_count.gt(0.0) & short_count.gt(0.0)
    weights = long_leg.div(long_count.replace(0.0, np.nan), axis=0) * 0.5
    weights -= short_leg.div(short_count.replace(0.0, np.nan), axis=0) * 0.5
    output: pd.DataFrame = weights.where(valid, 0.0).fillna(0.0)
    return output


def _causal_residual_scores(
    close_returns: pd.DataFrame,
    trading_dates: pd.Series,
    training_sessions: int,
) -> pd.DataFrame:
    """Estimate the first PC only from sessions strictly before each signal day."""
    output = pd.DataFrame(np.nan, index=close_returns.index, columns=close_returns.columns)
    unique_dates = pd.Index(pd.to_datetime(trading_dates.dropna().unique())).sort_values()
    normalized_dates = pd.to_datetime(trading_dates).to_numpy()
    for location, current_date in enumerate(unique_dates):
        if location < training_sessions:
            continue
        history_dates = unique_dates[location - training_sessions : location]
        training_mask = np.isin(normalized_dates, history_dates.to_numpy())
        current_mask = normalized_dates == current_date.to_datetime64()
        training = close_returns.loc[training_mask].dropna(how="any")
        if len(training) < max(80, training_sessions * 3):
            continue
        scale = training.std(ddof=1).replace(0.0, np.nan)
        standardized_training = training.div(scale, axis=1).dropna(how="any")
        if len(standardized_training) < 80:
            continue
        covariance = standardized_training.cov().to_numpy(dtype=float)
        _, eigenvectors = np.linalg.eigh(covariance)
        loading = eigenvectors[:, -1]
        training_values = standardized_training.to_numpy(dtype=float)
        training_residual = training_values - np.outer(training_values @ loading, loading)
        residual_scale = pd.Series(
            training_residual.std(axis=0, ddof=1), index=close_returns.columns
        ).replace(0.0, np.nan)
        current = close_returns.loc[current_mask].div(scale, axis=1)
        current_values = current.to_numpy(dtype=float)
        residual = current_values - np.outer(current_values @ loading, loading)
        output.loc[current_mask] = residual / residual_scale.to_numpy(dtype=float)
    return output


def build_population(
    minute_data: pd.DataFrame,
) -> tuple[
    list[PcaVariant],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    variants: list[PcaVariant] = []
    gross_returns: dict[str, pd.Series] = {}
    trade_sides: dict[str, pd.Series] = {}
    product_gross: dict[str, pd.DataFrame] = {}
    product_sides: dict[str, pd.DataFrame] = {}

    for timeframe in TIMEFRAMES:
        bars = resample_bars(minute_data, timeframe).sort_values(
            ["timestamp_utc", "symbol"], ignore_index=True
        )
        opens = bars.pivot(index="timestamp_utc", columns="symbol", values="open").sort_index()
        closes = bars.pivot(index="timestamp_utc", columns="symbol", values="close").reindex_like(
            opens
        )
        dates = bars.pivot(
            index="timestamp_utc", columns="symbol", values="trading_date"
        ).reindex_like(opens)
        date_by_bar = pd.to_datetime(dates.bfill(axis=1).iloc[:, 0])
        execution_returns = closes.div(opens).sub(1.0)
        close_returns = closes.pct_change(fill_method=None)
        same_session = dates.eq(dates.shift(1))
        close_returns = close_returns.where(same_session)

        for training_sessions in (20, 40):
            scores = _causal_residual_scores(close_returns, date_by_bar, training_sessions)
            for threshold in (1.25, 1.75, 2.25):
                for confirmation_bars in (1, 2):
                    candidate_scores = scores
                    if confirmation_bars == 2:
                        agrees = scores.mul(scores.shift(1)).gt(0.0)
                        prior_extreme = scores.shift(1).abs().ge(threshold)
                        candidate_scores = scores.where(agrees & prior_extreme & same_session)
                    name = (
                        f"h019r1_{timeframe:03d}m_train{training_sessions:02d}_"
                        f"threshold{str(threshold).replace('.', 'p')}_confirm{confirmation_bars}"
                    )
                    variant = PcaVariant(
                        name,
                        timeframe,
                        training_sessions,
                        threshold,
                        confirmation_bars,
                    )
                    variants.append(variant)
                    signal_weights = _balanced_tail_weights(candidate_scores, threshold)
                    execution_weights = signal_weights.shift(1).where(same_session, 0.0).fillna(0.0)
                    gross_by_product = execution_weights.mul(execution_returns.fillna(0.0))
                    sides_by_product = pd.DataFrame(
                        {
                            symbol: _intraday_trade_sides(
                                execution_weights[symbol], pd.to_datetime(dates[symbol])
                            )
                            for symbol in execution_weights.columns
                        },
                        index=execution_weights.index,
                    )
                    daily_gross = (
                        gross_by_product.groupby(date_by_bar)
                        .sum()
                        .reindex(daily_index, fill_value=0.0)
                    )
                    daily_sides = (
                        sides_by_product.groupby(date_by_bar)
                        .sum()
                        .reindex(daily_index, fill_value=0.0)
                    )
                    product_gross[name] = daily_gross
                    product_sides[name] = daily_sides
                    gross_returns[name] = daily_gross.sum(axis=1)
                    trade_sides[name] = daily_sides.sum(axis=1)

    return (
        variants,
        pd.DataFrame(gross_returns, index=daily_index),
        pd.DataFrame(trade_sides, index=daily_index),
        product_gross,
        product_sides,
    )


def chained_selection(
    net_returns: pd.DataFrame,
    trade_sides: pd.DataFrame,
    *,
    folds: int = 4,
) -> tuple[pd.Series, pd.DataFrame]:
    blocks = np.array_split(np.arange(len(net_returns)), folds)
    portfolio = pd.Series(np.nan, index=net_returns.index, name="portfolio_return")
    records: list[dict[str, Any]] = []
    for evaluation_number in range(1, folds):
        training_rows = np.concatenate(blocks[:evaluation_number])
        evaluation_rows = blocks[evaluation_number]
        ranked: list[tuple[float, str, list[float]]] = []
        for name in net_returns.columns:
            prior_sharpes = [
                sharpe_ratio(net_returns[name].iloc[block]) for block in blocks[:evaluation_number]
            ]
            trading_days = int((trade_sides[name].iloc[training_rows] > 0.0).sum())
            if trading_days < max(12, evaluation_number * 6):
                continue
            if not np.isfinite(prior_sharpes).all():
                continue
            score = float(np.mean(prior_sharpes) - 0.25 * np.std(prior_sharpes))
            ranked.append((score, str(name), prior_sharpes))
        ranked.sort(reverse=True)
        if not ranked:
            records.append(
                {"evaluation_fold": evaluation_number + 1, "strategy": "__NO_SELECTION__"}
            )
            continue
        score, selected, prior_sharpes = ranked[0]
        training = net_returns[selected].iloc[training_rows]
        annualized_volatility = float(training.std(ddof=1) * np.sqrt(252.0))
        leverage = min(2.0, 0.10 / annualized_volatility) if annualized_volatility else 1.0
        portfolio.iloc[evaluation_rows] = net_returns[selected].iloc[evaluation_rows] * leverage
        records.append(
            {
                "evaluation_fold": evaluation_number + 1,
                "strategy": selected,
                "training_score": score,
                "prior_fold_sharpes": json.dumps(prior_sharpes),
                "leverage": leverage,
            }
        )
    return portfolio.dropna(), pd.DataFrame(records)


def replay_selection(net_returns: pd.DataFrame, selections: pd.DataFrame) -> pd.Series:
    blocks = np.array_split(np.arange(len(net_returns)), 4)
    output = pd.Series(np.nan, index=net_returns.index, name="portfolio_return")
    for _, record in selections.iterrows():
        strategy = str(record["strategy"])
        if strategy == "__NO_SELECTION__":
            continue
        rows = blocks[int(str(record["evaluation_fold"])) - 1]
        leverage = float(str(record["leverage"]))
        output.iloc[rows] = net_returns[strategy].iloc[rows] * leverage
    return output.dropna()


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    variants, gross, sides, product_gross, product_sides = build_population(minute_data)
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
        block_bootstrap_summary(portfolio, block_length=20, samples=5000, seed=20260823)
        if not portfolio.empty
        else {}
    )
    audit_gates = {
        "selection_available_all_evaluation_folds": len(fold_rows) == 3,
        "gross_expectancy_positive": float(cost_rows[0]["annualized_return"]) > 0.0,
        "one_bp_expectancy_positive": float(cost_rows[2]["annualized_return"]) > 0.0,
        "walk_forward_sharpe_at_least_2": bool(snapshot) and float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": bool(snapshot)
        and max(float(snapshot["win_weeks"]), float(snapshot["win_months"])) >= 0.70,
        "all_evaluation_folds_positive": len(fold_rows) == 3
        and all(float(row["annualized_return"]) > 0.0 for row in fold_rows),
        "bootstrap_fifth_percentile_sharpe_positive": bool(bootstrap)
        and float(bootstrap["sharpe_p05"]) > 0.0,
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
    manifest = {
        "hypothesis_id": "H019-R1",
        "research_stage": "development_only_chained_walk_forward",
        "sealed_year_accessed": False,
        "unique_hypothesis_count": 1,
        "parameter_sensitivity_variants": len(variants),
        "timeframes_minutes": list(TIMEFRAMES),
        "base_one_way_cost_bps": one_way_cost_bps,
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "market_sources": [path.relative_to(project_root).as_posix() for path in market_sources],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    master = project_root / "saved_strategies" / f"hypothesis_H019_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    registry = pd.DataFrame([asdict(variant) for variant in variants]).set_index("name")
    for variant_name in gross.columns:
        variant = registry.loc[variant_name].to_dict()
        for symbol in product_gross[variant_name].columns:
            folder = master / "strategies" / variant_name / symbol
            folder.mkdir(parents=True)
            definition = {"hypothesis_id": "H019-R1", "symbol": symbol, **variant}
            (folder / "strategy_definition.json").write_text(
                json.dumps(definition, indent=2), encoding="utf-8"
            )
            pd.DataFrame(
                {
                    "gross_contribution": product_gross[variant_name][symbol],
                    "trade_sides": product_sides[variant_name][symbol],
                }
            ).to_parquet(folder / "daily_evidence.parquet")
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/pca_residual_reversal.py",
        "research_program/hypothesis_registry.csv",
        "research_program/RESEARCH_LOOP_PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
    ):
        source = project_root / relative
        shutil.copy2(source, source_snapshot / relative.replace("/", "__"))
    product_folder_count = len(variants) * len(product_gross[variants[0].name].columns)
    (master / "MASTER_INDEX.md").write_text(
        "\n".join(
            [
                "# H019-R1 PCA Residual Reversal — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Parameter sensitivity variants: {len(variants)}",
                f"- Product strategy folders: {product_folder_count}",
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
    parser = argparse.ArgumentParser(description="Run H019 PCA residual reversal")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
