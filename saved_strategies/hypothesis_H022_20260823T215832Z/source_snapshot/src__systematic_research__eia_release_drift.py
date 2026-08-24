"""Post-EIA-release drift research for hypothesis H022."""

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
from pandas.tseries.holiday import USFederalHolidayCalendar

from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.metrics import sharpe_ratio
from systematic_research.pca_residual_reversal import sha256
from systematic_research.unique_hypothesis_factory import extended_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/eia_release_drift")
EIA_SOURCE = "https://www.eia.gov/reports/upcoming.php"
RELEASE_RULES = {"CL": 2, "NG": 3}  # Wednesday and Thursday, respectively.


@dataclass(frozen=True)
class EiaVariant:
    name: str
    symbol: str
    response_minutes: int
    hold_minutes: int
    response_threshold: float


def eligible_standard_release_dates(daily_dates: pd.DatetimeIndex, symbol: str) -> pd.DatetimeIndex:
    """Use standard release weekdays and exclude weeks disrupted by federal holidays."""
    weekday = RELEASE_RULES[symbol]
    candidates = daily_dates[daily_dates.weekday == weekday]
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(
        start=daily_dates.min() - pd.Timedelta(days=7), end=daily_dates.max()
    )
    eligible = []
    for date in candidates:
        monday = date - pd.Timedelta(days=date.weekday())
        disrupted = ((holidays >= monday) & (holidays <= date)).any()
        if not disrupted:
            eligible.append(date)
    # The 2025 National Day of Mourning shifted the gas report to Wednesday noon.
    exceptional_nonstandard = {pd.Timestamp("2025-01-09")}
    return pd.DatetimeIndex(date for date in eligible if date not in exceptional_nonstandard)


def event_returns(
    minute_data: pd.DataFrame,
    *,
    symbol: str,
    response_minutes: int,
    hold_minutes: int,
    threshold: float,
    daily_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    product = minute_data.loc[minute_data["symbol"].eq(symbol)].copy()
    local = pd.to_datetime(product["timestamp_utc"], utc=True).dt.tz_convert("America/New_York")
    product["local_minutes"] = local.dt.hour * 60 + local.dt.minute
    product["trading_date"] = pd.to_datetime(product["trading_date"])
    records = []
    release_dates = eligible_standard_release_dates(daily_index, symbol)
    for date in release_dates:
        day = product.loc[product["trading_date"].eq(date)]
        response_start = 10 * 60 + 30
        response_end = response_start + response_minutes
        target_end = response_end + hold_minutes
        response = day.loc[
            day["local_minutes"].ge(response_start) & day["local_minutes"].lt(response_end)
        ]
        target = day.loc[
            day["local_minutes"].ge(response_end) & day["local_minutes"].lt(target_end)
        ]
        if len(response) < response_minutes or len(target) < hold_minutes:
            continue
        records.append(
            {
                "trading_date": date,
                "response_return": float(
                    response["close"].iloc[-1] / response["open"].iloc[0] - 1.0
                ),
                "target_return": float(target["close"].iloc[-1] / target["open"].iloc[0] - 1.0),
            }
        )
    events = pd.DataFrame(records).set_index("trading_date")
    causal_scale = events["response_return"].rolling(20, min_periods=12).std().shift(1)
    events["response_score"] = events["response_return"].div(causal_scale.replace(0.0, np.nan))
    events["position"] = np.sign(events["response_return"]).where(
        events["response_score"].abs().ge(threshold), 0.0
    )
    gross = events["position"].mul(events["target_return"]).reindex(daily_index).fillna(0.0)
    sides = events["position"].abs().mul(2.0).reindex(daily_index).fillna(0.0)
    return gross, sides, events


def build_population(
    minute_data: pd.DataFrame,
) -> tuple[list[EiaVariant], pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    variants = []
    gross = {}
    sides = {}
    evidence = {}
    for symbol in sorted(RELEASE_RULES):
        for response_minutes in (15, 30):
            for hold_minutes in (30, 60, 240):
                for threshold in (0.5, 1.0):
                    name = (
                        f"h022_{symbol}_{response_minutes:02d}m_response_"
                        f"{hold_minutes:03d}m_hold_threshold{str(threshold).replace('.', 'p')}"
                    )
                    variant = EiaVariant(name, symbol, response_minutes, hold_minutes, threshold)
                    variants.append(variant)
                    gross[name], sides[name], evidence[name] = event_returns(
                        minute_data,
                        symbol=symbol,
                        response_minutes=response_minutes,
                        hold_minutes=hold_minutes,
                        threshold=threshold,
                        daily_index=daily_index,
                    )
    return variants, pd.DataFrame(gross), pd.DataFrame(sides), evidence


def chained_product_selection(
    variants: list[EiaVariant],
    net: pd.DataFrame,
    sides: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    registry = pd.DataFrame([asdict(variant) for variant in variants]).set_index("name")
    blocks = np.array_split(np.arange(len(net)), 4)
    portfolio = pd.Series(np.nan, index=net.index, name="portfolio_return")
    records = []
    for evaluation_number in range(1, 4):
        training_rows = np.concatenate(blocks[:evaluation_number])
        evaluation_rows = blocks[evaluation_number]
        selected = []
        for symbol in sorted(RELEASE_RULES):
            names = registry.index[registry["symbol"].eq(symbol)]
            ranked = []
            for name in names:
                if int((sides[name].iloc[training_rows] > 0.0).sum()) < 3:
                    continue
                prior = [
                    sharpe_ratio(net[name].iloc[block]) for block in blocks[:evaluation_number]
                ]
                if not np.isfinite(prior).all():
                    continue
                score = float(np.mean(prior) - 0.25 * np.std(prior))
                ranked.append((score, str(name), prior))
            ranked.sort(reverse=True)
            if not ranked:
                continue
            score, name, prior = ranked[0]
            selected.append(name)
            records.append(
                {
                    "evaluation_fold": evaluation_number + 1,
                    "symbol": symbol,
                    "strategy": name,
                    "training_score": score,
                    "prior_fold_sharpes": json.dumps(prior),
                }
            )
        if selected:
            portfolio.iloc[evaluation_rows] = net[selected].iloc[evaluation_rows].sum(axis=1)
    return portfolio.dropna(), pd.DataFrame(records)


def replay(net: pd.DataFrame, selections: pd.DataFrame) -> pd.Series:
    blocks = np.array_split(np.arange(len(net)), 4)
    output = pd.Series(np.nan, index=net.index, name="portfolio_return")
    for fold in range(2, 5):
        names = selections.loc[selections["evaluation_fold"].eq(fold), "strategy"].tolist()
        if names:
            rows = blocks[fold - 1]
            output.iloc[rows] = net[names].iloc[rows].sum(axis=1)
    return output.dropna()


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    variants, gross, sides, event_evidence = build_population(minute_data)
    net = gross - sides * one_way_cost_bps / 10_000.0
    portfolio, selections = chained_product_selection(variants, net, sides)
    snapshot = extended_snapshot(portfolio)
    blocks = np.array_split(np.arange(len(net)), 4)
    fold_rows = [
        {
            "fold": fold,
            **extended_snapshot(portfolio.reindex(net.index[blocks[fold - 1]]).dropna()),
        }
        for fold in range(2, 5)
    ]
    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed = replay(gross - sides * cost / 10_000.0, selections)
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
        "hypothesis_id": "H022",
        "research_stage": "development_only_chained_event_study",
        "sealed_year_accessed": False,
        "unique_hypothesis_count": 1,
        "parameter_sensitivity_variants": len(variants),
        "base_one_way_cost_bps": one_way_cost_bps,
        "event_source": EIA_SOURCE,
        "release_rules": {"CL": "Wednesday 10:30 ET", "NG": "Thursday 10:30 ET"},
        "holiday_handling": "exclude federal-holiday-disrupted weeks and known 2025 exception",
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "market_sources": [path.relative_to(project_root).as_posix() for path in market_sources],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    master = project_root / "saved_strategies" / f"hypothesis_H022_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    registry = pd.DataFrame([asdict(variant) for variant in variants]).set_index("name")
    for name, frame in event_evidence.items():
        folder = master / "strategies" / name
        folder.mkdir(parents=True)
        definition = {"hypothesis_id": "H022", **registry.loc[name].to_dict()}
        (folder / "strategy_definition.json").write_text(
            json.dumps(definition, indent=2), encoding="utf-8"
        )
        frame.to_parquet(folder / "event_evidence.parquet")
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/eia_release_drift.py",
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
                "# H022 EIA Release Drift — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Strategy folders: {len(variants)}",
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
    parser = argparse.ArgumentParser(description="Run H022 EIA release drift")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
