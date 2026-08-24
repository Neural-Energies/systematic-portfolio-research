"""CFTC managed-money information-diffusion research for hypothesis H020."""

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

from systematic_research.cot_hedging_pressure import CFTC_CODES, CFTC_URLS, portfolio_returns
from systematic_research.daily_session_factory import block_bootstrap_summary
from systematic_research.unique_hypothesis_factory import extended_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/managed_money_diffusion")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_managed_money(
    project_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    frames = []
    sources = []
    for year, url in CFTC_URLS.items():
        path = project_root / "data/external/cftc" / f"fut_disagg_txt_{year}.zip"
        sources.append(
            {
                "year": str(year),
                "url": url,
                "path": path.relative_to(project_root).as_posix(),
                "sha256": sha256(path),
            }
        )
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as handle:
                frames.append(
                    pd.read_csv(
                        handle,
                        low_memory=False,
                        dtype={"CFTC_Contract_Market_Code": str},
                    )
                )
    reports = pd.concat(frames, ignore_index=True)
    reports["CFTC_Contract_Market_Code"] = reports["CFTC_Contract_Market_Code"].str.strip()
    inverse = {code: symbol for symbol, code in CFTC_CODES.items()}
    reports = reports.loc[reports["CFTC_Contract_Market_Code"].isin(inverse)].copy()
    reports["symbol"] = reports["CFTC_Contract_Market_Code"].map(inverse)
    reports["report_date"] = pd.to_datetime(reports["Report_Date_as_YYYY-MM-DD"])
    reports["release_date"] = reports["report_date"] + pd.Timedelta(days=3)
    reports["managed_money_net_share"] = reports["M_Money_Positions_Long_All"].sub(
        reports["M_Money_Positions_Short_All"]
    ) / reports["Open_Interest_All"].replace(0.0, np.nan)
    columns = [
        "symbol",
        "report_date",
        "release_date",
        "managed_money_net_share",
        "M_Money_Positions_Long_All",
        "M_Money_Positions_Short_All",
        "Open_Interest_All",
        "CFTC_Contract_Market_Code",
        "Market_and_Exchange_Names",
    ]
    return reports.loc[:, columns].sort_values(["report_date", "symbol"]), sources


def build_position_variants(
    reports: pd.DataFrame, daily_index: pd.DatetimeIndex
) -> dict[tuple[int, float], pd.DataFrame]:
    pivot = reports.pivot(
        index="report_date", columns="symbol", values="managed_money_net_share"
    ).dropna(subset=list(CFTC_CODES))
    ranks = pivot.rank(axis=1, method="average")
    center = (len(CFTC_CODES) + 1.0) / 2.0
    scale = (len(CFTC_CODES) - 1.0) / 2.0
    scores = ranks.sub(center).div(scale)
    release_dates = reports.groupby("report_date")["release_date"].max().reindex(scores.index)
    release_values = pd.to_datetime(release_dates).to_numpy()
    outputs = {}
    for hold_sessions in (1, 3, 5):
        for threshold in (0.5, 1.0):
            panel = pd.DataFrame(0.0, index=daily_index, columns=sorted(CFTC_CODES))
            for row_number, (_, score) in enumerate(scores.iterrows()):
                release_date = pd.Timestamp(release_values[row_number])
                start = int(daily_index.searchsorted(release_date, side="right"))
                if start >= len(daily_index):
                    continue
                position = score.where(score.abs().ge(threshold), 0.0).apply(np.sign)
                dates = daily_index[start : start + hold_sessions]
                panel.loc[dates, position.index] = position.astype(float).to_numpy()
            outputs[(hold_sessions, threshold)] = panel
    return outputs


def run_factory(project_root: Path, *, one_way_cost_bps: float = 1.0) -> Path:
    started = perf_counter()
    minute_data, market_sources = load_development_minutes(project_root)
    sessions = (
        minute_data.sort_values(["symbol", "trading_date", "timestamp_utc"])
        .groupby(["symbol", "trading_date"], sort=True)
        .agg(session_open=("open", "first"), session_close=("close", "last"))
        .reset_index()
    )
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(sessions["trading_date"]).unique()))
    opens = sessions.pivot(index="trading_date", columns="symbol", values="session_open")
    closes = sessions.pivot(index="trading_date", columns="symbol", values="session_close")
    session_returns = (
        closes.div(opens).sub(1.0).reindex(index=daily_index, columns=sorted(CFTC_CODES))
    )
    reports, cftc_sources = load_managed_money(project_root)
    panels = build_position_variants(reports, daily_index)
    returns = {
        key: portfolio_returns(panel, session_returns, one_way_cost_bps)[0]
        for key, panel in panels.items()
    }
    primary_key = (5, 0.5)
    primary = returns[primary_key]
    blocks = np.array_split(np.arange(len(primary)), 4)
    fold_rows = [
        {"fold": number, **extended_snapshot(primary.iloc[rows])}
        for number, rows in enumerate(blocks, start=1)
    ]
    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed, _ = portfolio_returns(panels[primary_key], session_returns, cost)
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed)})
    snapshot = extended_snapshot(primary)
    bootstrap = block_bootstrap_summary(primary, block_length=20, samples=5000, seed=20260823)
    audit_gates = {
        "sharpe_at_least_2": float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": max(
            float(snapshot["win_weeks"]), float(snapshot["win_months"])
        )
        >= 0.70,
        "at_least_two_positive_chronological_folds": sum(
            float(row["annualized_return"]) > 0.0 for row in fold_rows
        )
        >= 2,
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
    }
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True)
    reports.to_parquet(output / "managed_money_reports.parquet", index=False)
    session_returns.to_parquet(output / "development_session_returns.parquet")
    registry_rows = []
    for (hold, threshold), panel in panels.items():
        label = f"hold{hold}_threshold{str(threshold).replace('.', 'p')}"
        panel.to_parquet(output / f"positions_{label}.parquet")
        returns[(hold, threshold)].to_frame().to_parquet(output / f"returns_{label}.parquet")
        registry_rows.append({"variant": label, "hold_sessions": hold, "threshold": threshold})
    pd.DataFrame(registry_rows).to_csv(output / "strategy_registry.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    manifest = {
        "hypothesis_id": "H020",
        "research_stage": "development_only_preregistered_external_signal",
        "sealed_year_accessed": False,
        "primary_variant": "hold5_threshold0p5",
        "base_one_way_cost_bps": one_way_cost_bps,
        "signal": "managed-money net-long share; follow high and short low",
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

    master = project_root / "saved_strategies" / f"hypothesis_H020_{run_id}"
    evidence = master / "evidence"
    evidence.mkdir(parents=True)
    for source in output.iterdir():
        shutil.copy2(source, evidence / source.name)
    for (hold, threshold), panel in panels.items():
        label = f"hold{hold}_threshold{str(threshold).replace('.', 'p')}"
        for symbol in panel.columns:
            folder = master / "strategies" / label / symbol
            folder.mkdir(parents=True)
            (folder / "strategy_definition.json").write_text(
                json.dumps(
                    {
                        "hypothesis_id": "H020",
                        "symbol": symbol,
                        "hold_sessions": hold,
                        "threshold": threshold,
                        "direction": "follow managed-money cross-sectional rank",
                        "session_rule": "trade open to close; flat overnight",
                        "one_way_cost_bps": one_way_cost_bps,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {"position": panel[symbol], "session_return": session_returns[symbol]}
            ).to_parquet(folder / "daily_evidence.parquet")
    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/managed_money_diffusion.py",
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
                "# H020 Managed-Money Diffusion — Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Audit passed: {manifest['audit_passed']}",
                "- Unique hypothesis count: 1",
                "- Product strategy folders: 30",
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
    parser = argparse.ArgumentParser(description="Run H020 managed-money diffusion")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=1.0)
    arguments = parser.parse_args()
    run_factory(arguments.project_root.resolve(), one_way_cost_bps=arguments.one_way_cost_bps)


if __name__ == "__main__":
    main()
