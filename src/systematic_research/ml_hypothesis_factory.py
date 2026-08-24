"""Pooled walk-forward ML test for preregistered intraday hypothesis H017."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.intraday_strategy_search import resample_bars
from systematic_research.unique_hypothesis_factory import extended_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/ml_hypothesis_factory")
FEATURE_COLUMNS = (
    "overnight_return",
    "eu_open_return",
    "pre_open_return",
    "open_30_return",
    "overnight_volatility",
    "observed_volume_surprise",
    "prior_target_return",
    "prior_session_momentum_5",
    "prior_target_volatility_20",
    "overnight_cross_rank",
    "open_30_cross_rank",
    "cross_market_overnight",
    "weekday_sin",
    "weekday_cos",
)


def _window_summary(
    bars: pd.DataFrame, mask: pd.Series, prefix: str, minimum_bars: int
) -> pd.DataFrame:
    subset = bars.loc[mask, ["symbol", "trading_date", "bar_return", "volume"]]
    grouped = subset.groupby(["symbol", "trading_date"], sort=True)
    summary = grouped.agg(
        **{
            f"{prefix}_gross": ("bar_return", lambda values: values.add(1.0).prod() - 1.0),
            f"{prefix}_volatility": ("bar_return", "std"),
            f"{prefix}_volume": ("volume", "sum"),
            f"{prefix}_bars": ("bar_return", "size"),
        }
    )
    valid = summary[f"{prefix}_bars"].ge(minimum_bars)
    return summary.where(valid, axis="index")


def build_daily_learning_frame(minute_data: pd.DataFrame) -> pd.DataFrame:
    """Create features known by 10:00 ET and a strictly later return target."""
    bars = resample_bars(minute_data, 15).sort_values(
        ["symbol", "trading_date", "timestamp_utc"], ignore_index=True
    )
    bars["bar_return"] = bars["close"].div(bars["open"]).sub(1.0)
    local = pd.to_datetime(bars["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    minutes = local.dt.hour.mul(60).add(local.dt.minute)
    masks = {
        "overnight": minutes.ge(18 * 60) | minutes.lt(9 * 60 + 30),
        "eu_open": minutes.ge(23 * 60 + 30) | minutes.lt(3 * 60 + 30),
        "pre_open": minutes.ge(3 * 60 + 30) & minutes.lt(9 * 60 + 30),
        "open_30": minutes.ge(9 * 60 + 30) & minutes.lt(10 * 60),
        "target": minutes.ge(10 * 60) & minutes.lt(16 * 60),
    }
    minimums = {"overnight": 55, "eu_open": 14, "pre_open": 21, "open_30": 2, "target": 22}
    pieces = [_window_summary(bars, mask, name, minimums[name]) for name, mask in masks.items()]
    frame = pd.concat(pieces, axis=1).reset_index()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame = frame.sort_values(["symbol", "trading_date"], ignore_index=True)
    frame = frame.rename(
        columns={
            "overnight_gross": "overnight_return",
            "eu_open_gross": "eu_open_return",
            "pre_open_gross": "pre_open_return",
            "open_30_gross": "open_30_return",
            "target_gross": "target_return",
            "overnight_volatility": "overnight_volatility",
        }
    )
    frame["observed_volume"] = (
        frame["overnight_volume"].fillna(0.0)
        + frame["pre_open_volume"].fillna(0.0)
        + frame["open_30_volume"].fillna(0.0)
    )
    grouped = frame.groupby("symbol", sort=False)
    prior_volume = grouped["observed_volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean().shift(1)
    )
    frame["observed_volume_surprise"] = frame["observed_volume"].div(
        prior_volume.replace(0.0, np.nan)
    )
    frame["prior_target_return"] = grouped["target_return"].shift(1)
    frame["prior_session_momentum_5"] = grouped["target_return"].transform(
        lambda values: values.rolling(5, min_periods=5).sum().shift(1)
    )
    frame["prior_target_volatility_20"] = grouped["target_return"].transform(
        lambda values: values.rolling(20, min_periods=20).std().shift(1)
    )
    frame["overnight_cross_rank"] = frame.groupby("trading_date")["overnight_return"].rank(pct=True)
    frame["open_30_cross_rank"] = frame.groupby("trading_date")["open_30_return"].rank(pct=True)
    frame["cross_market_overnight"] = frame.groupby("trading_date")["overnight_return"].transform(
        "mean"
    )
    weekday = frame["trading_date"].dt.dayofweek.astype(float)
    frame["weekday_sin"] = np.sin(2.0 * np.pi * weekday / 5.0)
    frame["weekday_cos"] = np.cos(2.0 * np.pi * weekday / 5.0)
    return frame.dropna(subset=["target_return"]).reset_index(drop=True)


def _design_matrix(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    numeric = frame.loc[:, FEATURE_COLUMNS].astype(float)
    dummies = pd.get_dummies(frame["symbol"], prefix="symbol", dtype=float)
    expected = [f"symbol_{symbol}" for symbol in symbols]
    return numeric.join(dummies).reindex(columns=[*FEATURE_COLUMNS, *expected], fill_value=0.0)


def walk_forward_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Fit prespecified pooled ridge and nonlinear models on preceding date blocks."""
    dates = pd.DatetimeIndex(sorted(frame["trading_date"].unique()))
    blocks = np.array_split(dates, 4)
    symbols = sorted(frame["symbol"].astype(str).unique())
    output: list[pd.DataFrame] = []
    for evaluation_index in range(1, 4):
        training_dates = pd.DatetimeIndex(np.concatenate(blocks[:evaluation_index]))
        evaluation_dates = blocks[evaluation_index]
        training = frame.loc[frame["trading_date"].isin(training_dates)].copy()
        evaluation = frame.loc[frame["trading_date"].isin(evaluation_dates)].copy()
        x_train = _design_matrix(training, symbols)
        x_evaluation = _design_matrix(evaluation, symbols)
        medians = x_train.median()
        x_train = x_train.fillna(medians).fillna(0.0)
        x_evaluation = x_evaluation.fillna(medians).fillna(0.0)
        target = training["target_return"].astype(float)

        scaler = StandardScaler()
        ridge = Ridge(alpha=10.0)
        ridge.fit(scaler.fit_transform(x_train), target)
        ridge_prediction = ridge.predict(scaler.transform(x_evaluation))
        nonlinear = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=100,
            l2_regularization=10.0,
            random_state=20260823,
        )
        nonlinear.fit(x_train, target)
        nonlinear_prediction = nonlinear.predict(x_evaluation)
        predicted = evaluation[["trading_date", "symbol", "target_return"]].copy()
        predicted["evaluation_fold"] = evaluation_index + 1
        predicted["ridge_prediction"] = ridge_prediction
        predicted["nonlinear_prediction"] = nonlinear_prediction
        predicted["ensemble_prediction"] = (ridge_prediction + nonlinear_prediction) / 2.0
        output.append(predicted)
    return pd.concat(output, ignore_index=True)


def strategy_returns(
    predictions: pd.DataFrame,
    *,
    prediction_column: str,
    entry_hurdle_one_way_bps: float,
    one_way_cost_bps: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """Trade the four strongest cost-covering product forecasts once per day."""
    records = predictions.copy()
    hurdle = 2.0 * entry_hurdle_one_way_bps / 10_000.0
    roundtrip_cost = 2.0 * one_way_cost_bps / 10_000.0
    records["absolute_prediction"] = records[prediction_column].abs()
    records["conviction_rank"] = records.groupby("trading_date")["absolute_prediction"].rank(
        method="first", ascending=False
    )
    records["position"] = (
        records[prediction_column]
        .where(records["absolute_prediction"].ge(hurdle), 0.0)
        .apply(np.sign)
        .where(records["conviction_rank"].le(4.0), 0.0)
    )
    records["gross_leg_return"] = records["position"] * records["target_return"]
    records["net_leg_return"] = records["gross_leg_return"] - (
        records["position"].abs() * roundtrip_cost
    )
    active = records["position"].abs().groupby(records["trading_date"]).sum()
    portfolio = (
        records["net_leg_return"]
        .groupby(records["trading_date"])
        .sum()
        .div(active.replace(0.0, np.nan))
    )
    return portfolio.reindex(sorted(records["trading_date"].unique())).fillna(0.0), records


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    """Run, audit, and archive H017 without accessing the consumed sealed year."""
    started = perf_counter()
    minute_data, source_paths = load_development_minutes(project_root)
    frame = build_daily_learning_frame(minute_data)
    predictions = walk_forward_predictions(frame)
    models = {
        "ridge": "ridge_prediction",
        "nonlinear": "nonlinear_prediction",
        "ensemble": "ensemble_prediction",
    }
    portfolio_returns: dict[str, pd.Series] = {}
    trade_records: dict[str, pd.DataFrame] = {}
    for model, column in models.items():
        portfolio_returns[model], trade_records[model] = strategy_returns(
            predictions,
            prediction_column=column,
            entry_hurdle_one_way_bps=one_way_cost_bps,
            one_way_cost_bps=one_way_cost_bps,
        )

    primary = portfolio_returns["ensemble"]
    fold_rows: list[dict[str, Any]] = []
    evaluation_folds = predictions["evaluation_fold"].dropna().astype(int).unique()
    for fold in evaluation_folds:
        rows = predictions.loc[predictions["evaluation_fold"].eq(fold)]
        dates = pd.DatetimeIndex(sorted(rows["trading_date"].unique()))
        fold_rows.append(
            {"fold": int(fold), **extended_snapshot(primary.reindex(dates).fillna(0.0))}
        )
    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed, _ = strategy_returns(
            predictions,
            prediction_column="ensemble_prediction",
            entry_hurdle_one_way_bps=one_way_cost_bps,
            one_way_cost_bps=cost,
        )
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed)})
    snapshot = extended_snapshot(primary)
    bootstrap = block_bootstrap_summary(primary, samples=5000)
    audit_gates = {
        "walk_forward_sharpe_at_least_2": float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": max(
            float(snapshot["win_weeks"]), float(snapshot["win_months"])
        )
        >= 0.70,
        "every_evaluation_fold_positive": all(
            float(row["annualized_return"]) > 0.0 for row in fold_rows
        ),
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True)
    frame.to_parquet(output / "daily_learning_frame.parquet", index=False)
    predictions.to_parquet(output / "walk_forward_predictions.parquet", index=False)
    pd.DataFrame(portfolio_returns).to_parquet(output / "model_portfolio_returns.parquet")
    pd.DataFrame(fold_rows).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    for model, records in trade_records.items():
        records.to_parquet(output / f"{model}_trade_records.parquet", index=False)

    manifest = {
        "hypothesis_id": "H017",
        "research_stage": "development_only_expanding_walk_forward",
        "sealed_year_accessed": False,
        "raw_model_variants": list(models),
        "parameter_variants_are_not_unique_ideas": True,
        "base_one_way_cost_bps": one_way_cost_bps,
        "primary_model": "ensemble",
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "development_inputs": [path.relative_to(project_root).as_posix() for path in source_paths],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    master = project_root / "saved_strategies" / f"hypothesis_H017_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        if source.is_file():
            shutil.copy2(source, evidence / source.name)
    for model, records in trade_records.items():
        for symbol, rows in records.groupby("symbol"):
            folder = master / "strategies" / model / str(symbol)
            folder.mkdir(parents=True)
            definition = {
                "hypothesis_id": "H017",
                "model": model,
                "symbol": str(symbol),
                "entry_time": "10:00 America/New_York",
                "exit_time": "16:00 America/New_York",
                "prediction_hurdle_roundtrip_bps": 2.0 * one_way_cost_bps,
                "portfolio_rule": "top four absolute forecasts per day",
            }
            (folder / "strategy_definition.json").write_text(
                json.dumps(definition, indent=2), encoding="utf-8"
            )
            rows.to_parquet(folder / "walk_forward_evidence.parquet", index=False)
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/ml_hypothesis_factory.py",
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
                "# H017 Pooled Intraday ML — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Primary net Sharpe: {snapshot['sharpe']:.4f}",
                "- Sealed year accessed: False",
                "- Strategy folders: 30 (three model variants across ten products)",
            ]
        ),
        encoding="utf-8",
    )
    checksum_rows = []
    for path in sorted(item for item in master.rglob("*") if item.is_file()):
        checksum_rows.append(
            {
                "path": path.relative_to(master).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    pd.DataFrame(checksum_rows).to_csv(master / "SHA256SUMS.csv", index=False)
    print(json.dumps(manifest, indent=2))
    print(f"Saved run: {output}")
    print(f"Saved master archive: {master}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered pooled ML hypothesis H017")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
