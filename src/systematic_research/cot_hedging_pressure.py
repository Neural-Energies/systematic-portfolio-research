"""CFTC hedging-pressure research for preregistered hypothesis H018."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.unique_hypothesis_factory import extended_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/cot_hedging_pressure")
CFTC_CODES = {
    "CL": "067651",
    "NG": "023651",
    "GC": "088691",
    "HG": "085692",
    "ZC": "002602",
}
CFTC_URLS = {
    year: f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
    for year in (2023, 2024, 2025)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cftc_reports(project_root: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Load official annual futures-only CFTC archives with source fingerprints."""
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    root = project_root / "data/external/cftc"
    for year, url in CFTC_URLS.items():
        path = root / f"fut_disagg_txt_{year}.zip"
        sources.append(
            {
                "year": str(year),
                "url": url,
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256(path),
            }
        )
        with zipfile.ZipFile(path) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as handle:
                frame = pd.read_csv(
                    handle,
                    low_memory=False,
                    dtype={"CFTC_Contract_Market_Code": str},
                )
        frames.append(frame)
    reports = pd.concat(frames, ignore_index=True)
    reports["CFTC_Contract_Market_Code"] = reports["CFTC_Contract_Market_Code"].str.strip()
    inverse = {code: symbol for symbol, code in CFTC_CODES.items()}
    reports = reports.loc[reports["CFTC_Contract_Market_Code"].isin(inverse)].copy()
    reports["symbol"] = reports["CFTC_Contract_Market_Code"].map(inverse)
    reports["report_date"] = pd.to_datetime(reports["Report_Date_as_YYYY-MM-DD"])
    reports["release_date"] = reports["report_date"] + pd.Timedelta(days=3)
    reports["hedging_pressure"] = reports["Prod_Merc_Positions_Short_All"].sub(
        reports["Prod_Merc_Positions_Long_All"]
    ) / reports["Open_Interest_All"].replace(0.0, np.nan)
    columns = [
        "symbol",
        "report_date",
        "release_date",
        "hedging_pressure",
        "Open_Interest_All",
        "Prod_Merc_Positions_Long_All",
        "Prod_Merc_Positions_Short_All",
        "Market_and_Exchange_Names",
        "CFTC_Contract_Market_Code",
    ]
    return reports.loc[:, columns].sort_values(["report_date", "symbol"]), sources


def build_signal_panels(
    reports: pd.DataFrame, daily_index: pd.DatetimeIndex
) -> dict[float, pd.DataFrame]:
    """Map Friday-available cross-sectional reports to the next trading session."""
    pivot = reports.pivot(index="report_date", columns="symbol", values="hedging_pressure")
    pivot = pivot.dropna(subset=list(CFTC_CODES))
    ranks = pivot.rank(axis=1, method="average")
    center = (len(CFTC_CODES) + 1.0) / 2.0
    scale = (len(CFTC_CODES) - 1.0) / 2.0
    scores = ranks.sub(center).div(scale)
    release_dates = reports.groupby("report_date")["release_date"].max().reindex(scores.index)
    release_values = pd.to_datetime(release_dates).to_numpy()
    outputs: dict[float, pd.DataFrame] = {}
    for threshold in (0.5, 1.0):
        panel = pd.DataFrame(0.0, index=daily_index, columns=sorted(CFTC_CODES))
        for row_number, (_, score) in enumerate(scores.iterrows()):
            release_date = pd.Timestamp(release_values[row_number])
            target_location = int(daily_index.searchsorted(release_date, side="right"))
            if target_location >= len(daily_index):
                continue
            target_date = daily_index[target_location]
            position = score.where(score.abs().ge(threshold), 0.0).apply(np.sign)
            panel.loc[target_date, position.index] = position.astype(float)
        outputs[threshold] = panel
    return outputs


def portfolio_returns(
    positions: pd.DataFrame, session_returns: pd.DataFrame, one_way_cost_bps: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Equal-weight active long and short legs on each weekly signal date."""
    aligned = positions.reindex_like(session_returns).fillna(0.0)
    gross_legs = aligned * session_returns.fillna(0.0)
    net_legs = gross_legs - aligned.abs() * 2.0 * one_way_cost_bps / 10_000.0
    active = aligned.abs().sum(axis=1)
    portfolio = net_legs.sum(axis=1).div(active.replace(0.0, np.nan)).fillna(0.0)
    return portfolio.rename("portfolio_return"), net_legs


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    """Run the fixed H018 strategy, audits, and complete per-strategy archive."""
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    sessions = (
        minute_data.sort_values(["symbol", "trading_date", "timestamp_utc"])
        .groupby(["symbol", "trading_date"], sort=True)
        .agg(session_open=("open", "first"), session_close=("close", "last"))
        .reset_index()
    )
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(sessions["trading_date"]).unique()))
    session_open = sessions.pivot(
        index="trading_date", columns="symbol", values="session_open"
    ).reindex(daily_index)
    session_close = sessions.pivot(
        index="trading_date", columns="symbol", values="session_close"
    ).reindex(daily_index)
    session_returns = session_close.div(session_open).sub(1.0).reindex(columns=sorted(CFTC_CODES))
    reports, cftc_sources = load_cftc_reports(project_root)
    panels = build_signal_panels(reports, daily_index)
    variants: dict[float, pd.Series] = {}
    net_legs: dict[float, pd.DataFrame] = {}
    for threshold, positions in panels.items():
        variants[threshold], net_legs[threshold] = portfolio_returns(
            positions, session_returns, one_way_cost_bps
        )
    primary_threshold = 0.5
    primary = variants[primary_threshold]

    blocks = np.array_split(np.arange(len(primary)), 4)
    fold_rows = [
        {"fold": number, **extended_snapshot(primary.iloc[rows])}
        for number, rows in enumerate(blocks, start=1)
    ]
    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed, _ = portfolio_returns(panels[primary_threshold], session_returns, cost)
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed)})
    snapshot = extended_snapshot(primary)
    bootstrap = block_bootstrap_summary(primary, block_length=20, samples=5000)
    audit_gates = {
        "sharpe_at_least_2": float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": max(
            float(snapshot["win_weeks"]), float(snapshot["win_months"])
        )
        >= 0.70,
        "at_least_two_positive_outer_folds": sum(
            float(row["annualized_return"]) > 0.0 for row in fold_rows[1:]
        )
        >= 2,
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True)
    reports.to_parquet(output / "cftc_reports_filtered.parquet", index=False)
    session_returns.to_parquet(output / "development_session_returns.parquet")
    for threshold, positions in panels.items():
        label = str(threshold).replace(".", "p")
        positions.to_parquet(output / f"positions_threshold_{label}.parquet")
        variants[threshold].to_frame().to_parquet(
            output / f"portfolio_returns_threshold_{label}.parquet"
        )
    pd.DataFrame(fold_rows).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    manifest = {
        "hypothesis_id": "H018",
        "research_stage": "development_only_preregistered_external_signal",
        "sealed_year_accessed": False,
        "primary_threshold": primary_threshold,
        "base_one_way_cost_bps": one_way_cost_bps,
        "position_definition": "(producer_short-producer_long)/open_interest",
        "information_lag": (
            "Tuesday report assumed available Friday; trade first session after Friday"
        ),
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "cftc_sources": cftc_sources,
        "market_sources": [path.relative_to(project_root).as_posix() for path in market_sources],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    master = project_root / "saved_strategies" / f"hypothesis_H018_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    for threshold, positions in panels.items():
        label = str(threshold).replace(".", "p")
        for symbol in positions.columns:
            folder = master / "strategies" / f"threshold_{label}" / symbol
            folder.mkdir(parents=True)
            (folder / "strategy_definition.json").write_text(
                json.dumps(
                    {
                        "hypothesis_id": "H018",
                        "symbol": symbol,
                        "cross_sectional_threshold": threshold,
                        "direction": "long high producer hedging pressure; short low",
                        "hold": "next available session open to close",
                        "one_way_cost_bps": one_way_cost_bps,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "position": positions[symbol],
                    "session_return": session_returns[symbol],
                    "net_strategy_return": net_legs[threshold][symbol],
                }
            ).to_parquet(folder / "daily_evidence.parquet")
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/cot_hedging_pressure.py",
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
                "# H018 CFTC Hedging Pressure — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Primary net Sharpe: {snapshot['sharpe']:.4f}",
                "- Unique hypothesis count: 1",
                "- Strategy folders: 10 (two sensitivity variants across five products)",
                "- Sealed year accessed: False",
            ]
        ),
        encoding="utf-8",
    )
    checksum_rows = []
    for path in sorted(item for item in master.rglob("*") if item.is_file()):
        checksum_rows.append(
            {
                "path": path.relative_to(master).as_posix(),
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(checksum_rows).to_csv(master / "SHA256SUMS.csv", index=False)
    print(json.dumps(manifest, indent=2))
    print(f"Saved run: {output}")
    print(f"Saved master archive: {master}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CFTC hedging-pressure hypothesis H018")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
