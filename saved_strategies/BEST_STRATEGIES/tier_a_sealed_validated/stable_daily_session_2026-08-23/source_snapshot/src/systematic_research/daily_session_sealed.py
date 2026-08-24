"""Freeze and evaluate the audited daily-session candidate exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from systematic_research.daily_session_factory import (
    DailySessionSpec,
    score_panels,
    session_panels,
    vectorbt_replay,
)
from systematic_research.saved_strategy_runner import performance_snapshot

FACTORY_ROOT = Path("data/processed/daily_session_factory")
DEVELOPMENT_PANEL = Path("data/processed/databento_research/development_session_panel.parquet")
SEALED_PANEL = Path("data/processed/databento_research/sealed_holdout_session_panel.parquet")
FROZEN_ROOT = Path("saved_strategies/stable_daily_session_2026-08-23")
SOURCE_MODULE = Path("src/systematic_research/daily_session_factory.py")
FROZEN_DEFINITION = "frozen_candidate.json"
SEALED_LOCK = "SEALED_EVALUATION_LOCK.json"
SEALED_RESULT = "sealed_result.json"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sealed_evaluation_unused(frozen_root: Path) -> None:
    """Refuse any second attempt, including one following an interrupted first attempt."""
    if (frozen_root / SEALED_LOCK).exists() or (frozen_root / SEALED_RESULT).exists():
        raise PermissionError("sealed holdout evaluation has already been attempted")


def freeze_candidate(project_root: Path, *, run_id: str | None = None) -> Path:
    """Persist the audited development candidate and all information needed to replay it."""
    factory_root = project_root / FACTORY_ROOT
    if run_id is None:
        latest = json.loads((factory_root / "latest_run.json").read_text(encoding="utf-8"))
        run_id = str(latest["run_id"])
    run_root = factory_root / "runs" / run_id
    manifest = cast(
        dict[str, Any], json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    )
    stable = cast(dict[str, Any], manifest["stable_candidate"])
    if bool(manifest["sealed_year_accessed"]):
        raise PermissionError("development run reports prior sealed-year access")
    if not bool(stable["audit_passed"]):
        raise PermissionError("candidate did not pass every development audit gate")

    registry = pd.read_csv(run_root / "strategy_registry.csv")
    selected_names = [str(name) for name in cast(list[str], stable["strategies"])]
    selected = registry.loc[registry["name"].isin(selected_names)].copy()
    selected = selected.set_index("name").loc[selected_names].reset_index()
    specs = [
        asdict(
            DailySessionSpec(
                name=str(row["name"]),
                symbol=str(row["symbol"]),
                family=str(row["family"]),
                lookback_sessions=int(row["lookback_sessions"]),
                entry_threshold=float(row["entry_threshold"]),
            )
        )
        for row in selected.to_dict("records")
    ]

    frozen_root = project_root / FROZEN_ROOT
    if frozen_root.exists():
        raise FileExistsError(f"frozen candidate already exists: {frozen_root}")
    frozen_root.mkdir(parents=True, exist_ok=False)
    evidence_names = (
        "run_manifest.json",
        "stable_candidate_audit_gates.json",
        "stable_candidate_bootstrap.json",
        "stable_candidate_cost_stress.csv",
        "stable_candidate_fold_audit.csv",
        "stable_candidate_neighbor_audit.csv",
        "stable_candidate_weights.csv",
        "stable_candidate_tear_sheet.html",
    )
    for name in evidence_names:
        shutil.copy2(run_root / name, frozen_root / name)

    definition = {
        "candidate_id": "stable_daily_session_2026-08-23",
        "status": "frozen_development_candidate_holdout_not_run",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source_run_id": run_id,
        "strategy_specs": specs,
        "weights": cast(dict[str, float], stable["weights"]),
        "fixed_leverage": float(stable["fixed_leverage"]),
        "one_way_cost_bps": float(manifest["one_way_cost_bps"]),
        "signal_timing": manifest["signal_timing"],
        "execution": manifest["execution"],
        "overnight_exposure": False,
        "roll_gap_exposure": False,
        "selection_rule": (
            "one highest-full-Sharpe strategy per product among strategies with full Sharpe >= 1 "
            "and positive Sharpe in every one of four chronological development folds; retain the "
            "top six, requiring at least three; capped inverse-volatility weights"
        ),
        "development_audit_passed": True,
        "development_panel": {
            "path": DEVELOPMENT_PANEL.as_posix(),
            "sha256": sha256_file(project_root / DEVELOPMENT_PANEL),
        },
        "execution_source": {
            "path": SOURCE_MODULE.as_posix(),
            "sha256": sha256_file(project_root / SOURCE_MODULE),
        },
        "sealed_policy": {
            "run_count": 0,
            "one_attempt_only": True,
            "sealed_file_not_opened_during_freeze": True,
        },
    }
    output = frozen_root / FROZEN_DEFINITION
    output.write_text(json.dumps(definition, indent=2), encoding="utf-8")
    print(f"Frozen candidate: {output}")
    return output


def evaluate_sealed_once(project_root: Path) -> Path:
    """Evaluate the frozen strategy once without refitting or reselection."""
    frozen_root = project_root / FROZEN_ROOT
    definition_path = frozen_root / FROZEN_DEFINITION
    definition = cast(dict[str, Any], json.loads(definition_path.read_text(encoding="utf-8")))
    assert_sealed_evaluation_unused(frozen_root)
    development_record = cast(dict[str, str], definition["development_panel"])
    source_record = cast(dict[str, str], definition["execution_source"])
    if sha256_file(project_root / development_record["path"]) != development_record["sha256"]:
        raise RuntimeError("development panel differs from the frozen input")
    if sha256_file(project_root / source_record["path"]) != source_record["sha256"]:
        raise RuntimeError("execution source differs from the frozen implementation")

    lock_path = frozen_root / SEALED_LOCK
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {
                "state": "started_before_sealed_file_access",
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
            handle,
            indent=2,
        )

    development = pd.read_parquet(project_root / development_record["path"])
    sealed_path = project_root / SEALED_PANEL
    sealed = pd.read_parquet(sealed_path)
    combined = pd.concat([development, sealed], ignore_index=True)
    combined["trading_date"] = pd.to_datetime(combined["trading_date"])
    combined["session_open_utc"] = pd.to_datetime(combined["session_open_utc"], utc=True)
    combined["session_close_utc"] = pd.to_datetime(combined["session_close_utc"], utc=True)
    combined = combined.sort_values(["symbol", "trading_date"], ignore_index=True)
    panels = session_panels(combined)
    scores = score_panels(panels["session_return"])
    specs = [DailySessionSpec(**raw) for raw in definition["strategy_specs"]]
    replay = vectorbt_replay(
        combined,
        specs,
        scores,
        one_way_cost_bps=float(definition["one_way_cost_bps"]),
    )
    weights = pd.Series(definition["weights"], dtype=float)
    portfolio = replay[weights.index].mul(weights, axis=1).sum(axis=1)
    portfolio *= float(definition["fixed_leverage"])
    holdout_start = pd.Timestamp(sealed["trading_date"].min())
    holdout_end = pd.Timestamp(sealed["trading_date"].max())
    sealed_returns = portfolio.loc[
        (portfolio.index >= holdout_start) & (portfolio.index <= holdout_end)
    ]
    snapshot, periods = performance_snapshot(sealed_returns)
    replay.loc[sealed_returns.index].to_parquet(frozen_root / "sealed_strategy_returns.parquet")
    pd.DataFrame({"portfolio_return": sealed_returns}).to_parquet(
        frozen_root / "sealed_portfolio_returns.parquet"
    )
    periods.to_csv(frozen_root / "sealed_calendar_period_returns.csv", index=False)
    result = {
        "candidate_id": definition["candidate_id"],
        "evaluation_type": "single_frozen_sealed_holdout",
        "sealed_holdout_sha256": sha256_file(sealed_path),
        "holdout_accessed": True,
        "holdout_run_count": 1,
        "no_refit_or_reselection": True,
        "sealed_result": snapshot,
    }
    output = frozen_root / SEALED_RESULT
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lock_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "result": SEALED_RESULT,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze or evaluate the stable daily candidate")
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.action == "freeze":
        freeze_candidate(project_root, run_id=args.run_id)
    else:
        evaluate_sealed_once(project_root)


if __name__ == "__main__":
    main()
