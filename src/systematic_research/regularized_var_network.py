"""Causal regularized cross-futures VAR research for hypothesis H021."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.intraday_strategy_search import resample_bars
from systematic_research.pca_residual_reversal import (
    _balanced_tail_weights,
    chained_selection,
    replay_selection,
    sha256,
)
from systematic_research.unique_hypothesis_factory import (
    _intraday_trade_sides,
    extended_snapshot,
)
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/regularized_var_network")
TIMEFRAMES = (15, 30, 60, 240)


@dataclass(frozen=True)
class VarVariant:
    name: str
    timeframe_minutes: int
    training_sessions: int
    ridge_penalty: float
    forecast_threshold: float


def _causal_var_forecasts(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    trading_dates: pd.Series,
    *,
    training_sessions: int,
    ridge_penalty: float,
) -> pd.DataFrame:
    """Fit a multi-output ridge VAR using sessions strictly before the forecast day."""
    output = pd.DataFrame(np.nan, index=features.index, columns=targets.columns)
    unique_dates = pd.Index(pd.to_datetime(trading_dates.dropna().unique())).sort_values()
    normalized_dates = pd.to_datetime(trading_dates).to_numpy()
    for location, current_date in enumerate(unique_dates):
        if location < training_sessions:
            continue
        history_dates = unique_dates[location - training_sessions : location]
        training_mask = np.isin(normalized_dates, history_dates.to_numpy())
        current_mask = normalized_dates == current_date.to_datetime64()
        training = pd.concat(
            {"x": features.loc[training_mask], "y": targets.loc[training_mask]}, axis=1
        ).dropna(how="any")
        if len(training) < max(80, training_sessions * 3):
            continue
        x = cast(pd.DataFrame, training.xs("x", axis=1, level=0))
        y = cast(pd.DataFrame, training.xs("y", axis=1, level=0))
        x_mean = x.mean()
        x_scale = x.std(ddof=1).replace(0.0, np.nan)
        y_mean = y.mean()
        y_scale = y.std(ddof=1).replace(0.0, np.nan)
        standardized_x = x.sub(x_mean).div(x_scale)
        standardized_y = y.sub(y_mean).div(y_scale)
        valid_columns = standardized_x.columns[standardized_x.notna().all()]
        valid_targets = standardized_y.columns[standardized_y.notna().all()]
        if len(valid_columns) != features.shape[1] or len(valid_targets) != targets.shape[1]:
            continue
        x_values = standardized_x.to_numpy(dtype=float)
        y_values = standardized_y.to_numpy(dtype=float)
        penalty = np.eye(x_values.shape[1]) * ridge_penalty
        coefficients = np.linalg.solve(x_values.T @ x_values + penalty, x_values.T @ y_values)
        current_x = features.loc[current_mask].sub(x_mean).div(x_scale)
        prediction = current_x.to_numpy(dtype=float) @ coefficients
        output.loc[current_mask] = prediction
    return output


def build_population(
    minute_data: pd.DataFrame,
) -> tuple[
    list[VarVariant],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    variants = []
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
        same_session = dates.eq(dates.shift(1))
        next_same_session = dates.eq(dates.shift(-1))
        execution_returns = closes.div(opens).sub(1.0)
        features = closes.pct_change(fill_method=None).where(same_session)
        next_targets = execution_returns.shift(-1).where(next_same_session)
        for training_sessions in (20, 40):
            for ridge_penalty in (10.0, 100.0):
                forecasts = _causal_var_forecasts(
                    features,
                    next_targets,
                    date_by_bar,
                    training_sessions=training_sessions,
                    ridge_penalty=ridge_penalty,
                )
                for threshold in (0.10, 0.25):
                    name = (
                        f"h021_{timeframe:03d}m_train{training_sessions:02d}_"
                        f"ridge{int(ridge_penalty):03d}_"
                        f"threshold{str(threshold).replace('.', 'p')}"
                    )
                    variant = VarVariant(
                        name, timeframe, training_sessions, ridge_penalty, threshold
                    )
                    variants.append(variant)
                    signal_weights = _balanced_tail_weights(-forecasts, threshold)
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


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    variants, gross, sides, product_gross, product_sides = build_population(minute_data)
    net = gross - sides * one_way_cost_bps / 10_000.0
    portfolio, selections = chained_selection(net, sides)
    snapshot = extended_snapshot(portfolio)
    blocks = np.array_split(np.arange(len(net)), 4)
    fold_rows = []
    for fold in range(2, 5):
        fold_return = portfolio.reindex(net.index[blocks[fold - 1]]).dropna()
        fold_rows.append({"fold": fold, **extended_snapshot(fold_return)})
    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed = replay_selection(gross - sides * cost / 10_000.0, selections)
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed)})
    bootstrap = block_bootstrap_summary(portfolio, block_length=20, samples=5000, seed=20260823)
    audit_gates = {
        "gross_expectancy_positive": float(cost_rows[0]["annualized_return"]) > 0.0,
        "one_bp_expectancy_positive": float(cost_rows[2]["annualized_return"]) > 0.0,
        "walk_forward_sharpe_at_least_2": float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": max(
            float(snapshot["win_weeks"]), float(snapshot["win_months"])
        )
        >= 0.70,
        "all_evaluation_folds_positive": all(
            float(row["annualized_return"]) > 0.0 for row in fold_rows
        ),
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
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
        "hypothesis_id": "H021",
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

    master = project_root / "saved_strategies" / f"hypothesis_H021_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    registry = pd.DataFrame([asdict(variant) for variant in variants]).set_index("name")
    for variant_name in gross.columns:
        definition = registry.loc[variant_name].to_dict()
        for symbol in product_gross[variant_name].columns:
            folder = master / "strategies" / variant_name / symbol
            folder.mkdir(parents=True)
            (folder / "strategy_definition.json").write_text(
                json.dumps({"hypothesis_id": "H021", "symbol": symbol, **definition}, indent=2),
                encoding="utf-8",
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
        "src/systematic_research/regularized_var_network.py",
        "research_program/hypothesis_registry.csv",
        "research_program/RESEARCH_LOOP_PROTOCOL.md",
        "pyproject.toml",
        "uv.lock",
    ):
        source = project_root / relative
        shutil.copy2(source, source_snapshot / relative.replace("/", "__"))
    folder_count = len(variants) * len(product_gross[variants[0].name].columns)
    (master / "MASTER_INDEX.md").write_text(
        "\n".join(
            [
                "# H021 Regularized VAR Network — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Parameter sensitivity variants: {len(variants)}",
                f"- Product strategy folders: {folder_count}",
                "- Sealed year accessed: False",
            ]
        ),
        encoding="utf-8",
    )
    checksums = [
        {"path": path.relative_to(master).as_posix(), "sha256": sha256(path)}
        for path in sorted(item for item in master.rglob("*") if item.is_file())
    ]
    pd.DataFrame(checksums).to_csv(master / "SHA256SUMS.csv", index=False)
    print(json.dumps(manifest, indent=2))
    print(f"Saved run: {output}")
    print(f"Saved master archive: {master}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run H021 regularized VAR network")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
