"""Package every frozen daily-session strategy inside its portfolio master folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd

from systematic_research.saved_strategy_runner import performance_snapshot

MASTER_ROOT = Path("saved_strategies/stable_daily_session_2026-08-23")
RUN_ROOT = Path("data/processed/daily_session_factory/runs")


def sha256_file(path: Path) -> str:
    """Return a stable content checksum for one archive file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strategy_folder_name(position: int, spec: dict[str, Any]) -> str:
    """Build a readable, deterministic strategy folder name."""
    return (
        f"{position:02d}_{spec['symbol']}_{spec['name']}_{spec['family']}_"
        f"lb{int(spec['lookback_sessions'])}_threshold{float(spec['entry_threshold']):g}"
    )


def _copy_if_present(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _write_strategy_readme(
    path: Path,
    *,
    spec: dict[str, Any],
    weight: float,
    leverage: float,
    development: dict[str, Any],
    sealed: dict[str, Any],
) -> None:
    content = f"""# {spec["symbol"]} — {spec["name"]}

This folder is the self-contained record for one component of the frozen
`stable_daily_session_2026-08-23` portfolio.

## Rule

- Product: {spec["symbol"]}
- Family: {spec["family"]}
- Lookback: {int(spec["lookback_sessions"])} sessions
- Entry threshold: {float(spec["entry_threshold"]):g}
- Signal observed: session d close
- Execution: session d+1 open to session d+1 close
- Overnight exposure: none
- Frozen portfolio weight: {weight:.12f}
- Frozen portfolio leverage: {leverage:.12f}

## Saved performance

- Development raw Sharpe: {float(development["raw"]["sharpe"]):.4f}
- Development weighted-contribution return: {float(development["weighted"]["total_return"]):.2%}
- Sealed raw Sharpe: {float(sealed["raw"]["sharpe"]):.4f}
- Sealed weighted-contribution return: {float(sealed["weighted"]["total_return"]):.2%}
- Sealed weighted-contribution Sharpe: {float(sealed["weighted"]["sharpe"]):.4f}

`development_returns.parquet` and `sealed_returns.parquet` contain both the
raw strategy return and its contribution after applying the frozen portfolio
weight and leverage. No strategy was refit or rerun while creating this folder.
"""
    path.write_text(content, encoding="utf-8")


def archive_candidate(project_root: Path) -> Path:
    """Create an indexed, checksummed archive without rerunning the sealed strategy."""
    master = project_root / MASTER_ROOT
    definition = cast(
        dict[str, Any], json.loads((master / "frozen_candidate.json").read_text(encoding="utf-8"))
    )
    run_id = str(definition["source_run_id"])
    run_root = project_root / RUN_ROOT / run_id
    development_returns = pd.read_parquet(run_root / "stable_candidate_vectorbt_returns.parquet")
    sealed_returns = pd.read_parquet(master / "sealed_strategy_returns.parquet")
    temporal = pd.read_parquet(run_root / "all_strategy_temporal_stability.parquet")
    neighbors = pd.read_csv(run_root / "stable_candidate_neighbor_audit.csv")
    weights = pd.Series(definition["weights"], dtype=float)
    leverage = float(definition["fixed_leverage"])

    strategies_root = master / "strategies"
    portfolio_root = master / "portfolio"
    evidence_root = master / "evidence"
    source_root = master / "source_snapshot"
    for folder in (strategies_root, portfolio_root, evidence_root, source_root):
        folder.mkdir(parents=True, exist_ok=True)

    strategy_index: list[dict[str, Any]] = []
    for position, raw_spec in enumerate(definition["strategy_specs"], start=1):
        spec = cast(dict[str, Any], raw_spec)
        name = str(spec["name"])
        folder_name = strategy_folder_name(position, spec)
        folder = strategies_root / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        weight = float(weights[name])

        development_frame = pd.DataFrame(
            {
                "raw_strategy_return": development_returns[name],
                "weighted_portfolio_contribution": development_returns[name] * weight * leverage,
            }
        )
        sealed_frame = pd.DataFrame(
            {
                "raw_strategy_return": sealed_returns[name],
                "weighted_portfolio_contribution": sealed_returns[name] * weight * leverage,
            }
        )
        development_frame.to_parquet(folder / "development_returns.parquet")
        sealed_frame.to_parquet(folder / "sealed_returns.parquet")
        development_raw, _ = performance_snapshot(development_frame["raw_strategy_return"])
        development_weighted, _ = performance_snapshot(
            development_frame["weighted_portfolio_contribution"]
        )
        sealed_raw, _ = performance_snapshot(sealed_frame["raw_strategy_return"])
        sealed_weighted, _ = performance_snapshot(sealed_frame["weighted_portfolio_contribution"])
        performance = {
            "development": {"raw": development_raw, "weighted": development_weighted},
            "sealed": {"raw": sealed_raw, "weighted": sealed_weighted},
        }
        strategy_definition = {
            "candidate_id": definition["candidate_id"],
            "strategy": spec,
            "frozen_weight": weight,
            "fixed_portfolio_leverage": leverage,
            "one_way_cost_bps": float(definition["one_way_cost_bps"]),
            "signal_timing": definition["signal_timing"],
            "execution": definition["execution"],
            "overnight_exposure": False,
            "roll_gap_exposure": False,
            "source_run_id": run_id,
            "sealed_run_count": 1,
            "no_refit_or_reselection_during_archive": True,
        }
        (folder / "strategy_definition.json").write_text(
            json.dumps(strategy_definition, indent=2), encoding="utf-8"
        )
        (folder / "performance.json").write_text(
            json.dumps(performance, indent=2), encoding="utf-8"
        )
        temporal.loc[temporal["name"].eq(name)].to_csv(folder / "temporal_audit.csv", index=False)
        neighbors.loc[neighbors["selected_strategy"].eq(name)].to_csv(
            folder / "parameter_neighbor_audit.csv", index=False
        )
        _write_strategy_readme(
            folder / "README.md",
            spec=spec,
            weight=weight,
            leverage=leverage,
            development=performance["development"],
            sealed=performance["sealed"],
        )
        strategy_index.append(
            {
                "order": position,
                "name": name,
                "symbol": spec["symbol"],
                "family": spec["family"],
                "folder": folder.relative_to(master).as_posix(),
                "weight": weight,
                "sealed_total_return_contribution": sealed_weighted["total_return"],
                "sealed_sharpe": sealed_weighted["sharpe"],
            }
        )

    portfolio_files = (
        "frozen_candidate.json",
        "sealed_result.json",
        "post_sealed_audit.json",
        "stable_candidate_weights.csv",
        "sealed_portfolio_returns.parquet",
        "sealed_calendar_period_returns.csv",
        "AUDIT_REPORT.html",
        "audit_report_artifact.json",
    )
    for name in portfolio_files:
        _copy_if_present(master / name, portfolio_root / name)
    _copy_if_present(
        run_root / "stable_candidate_portfolio_returns.parquet",
        portfolio_root / "development_portfolio_returns.parquet",
    )

    evidence_files = (
        "run_manifest.json",
        "stable_candidate_audit_gates.json",
        "stable_candidate_bootstrap.json",
        "stable_candidate_cost_stress.csv",
        "stable_candidate_fold_audit.csv",
        "stable_candidate_neighbor_audit.csv",
        "stable_candidate_tear_sheet.html",
        "SEALED_EVALUATION_LOCK.json",
    )
    for name in evidence_files:
        _copy_if_present(master / name, evidence_root / name)
    for name in (
        "all_strategy_temporal_stability.parquet",
        "build_selection_summary.parquet",
        "strategy_registry.csv",
    ):
        _copy_if_present(run_root / name, evidence_root / name)

    source_files = (
        "src/systematic_research/daily_session_factory.py",
        "src/systematic_research/daily_session_sealed.py",
        "src/systematic_research/archive_daily_session_candidate.py",
        "tests/test_daily_session_factory.py",
        "pyproject.toml",
        "uv.lock",
        "AGENTS.md",
    )
    for relative in source_files:
        source = project_root / relative
        _copy_if_present(source, source_root / relative)

    master_index = {
        "candidate_id": definition["candidate_id"],
        "archive_layout_version": 1,
        "source_run_id": run_id,
        "sealed_run_count": 1,
        "strategy_count": len(strategy_index),
        "strategies": strategy_index,
        "portfolio_folder": "portfolio",
        "evidence_folder": "evidence",
        "source_snapshot_folder": "source_snapshot",
        "compatibility_note": "Original root artifacts were preserved; organized files are copies.",
    }
    (master / "MASTER_INDEX.json").write_text(json.dumps(master_index, indent=2), encoding="utf-8")
    markdown_rows = "\n".join(
        f"| {row['order']} | {row['symbol']} | {row['name']} | {row['family']} | "
        f"{float(row['weight']):.2%} | [{row['folder']}]({row['folder']}/README.md) |"
        for row in strategy_index
    )
    (master / "MASTER_INDEX.md").write_text(
        "# Stable Daily Session Portfolio Archive\n\n"
        "All original root artifacts remain in place. Each frozen strategy now has its own "
        "folder with definitions, returns, metrics, and audit evidence.\n\n"
        "| # | Product | Strategy | Family | Weight | Folder |\n"
        "|---:|---|---|---|---:|---|\n"
        f"{markdown_rows}\n\n"
        "- Portfolio package: [portfolio](portfolio)\n"
        "- Audit and search evidence: [evidence](evidence)\n"
        "- Exact source snapshot: [source_snapshot](source_snapshot)\n",
        encoding="utf-8",
    )

    checksum_path = master / "SHA256SUMS.json"
    checksums = {
        path.relative_to(master).as_posix(): sha256_file(path)
        for path in sorted(master.rglob("*"))
        if path.is_file()
        and path != checksum_path
        and ".tmp-" not in path.name
        and not path.name.endswith("verification-failure.png")
    }
    checksum_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    print(f"Archived {len(strategy_index)} strategies under {strategies_root}")
    print(f"Master index: {master / 'MASTER_INDEX.md'}")
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive the frozen daily-session candidate")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    archive_candidate(args.project_root.resolve())


if __name__ == "__main__":
    main()
