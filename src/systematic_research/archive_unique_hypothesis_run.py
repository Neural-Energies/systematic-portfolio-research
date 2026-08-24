"""Archive every unique-hypothesis variant in a reproducible master folder."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_run(project_root: Path, run_id: str) -> Path:
    """Save evidence plus one definition-and-return folder per tested variant."""
    run_root = project_root / "data/processed/unique_hypothesis_factory/runs" / run_id
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    research_pass = int(manifest.get("research_pass", 1))
    master = project_root / "saved_strategies" / f"unique_hypothesis_pass_{research_pass}_{run_id}"
    master.mkdir(parents=True, exist_ok=False)
    evidence = master / "evidence"
    evidence.mkdir()

    for source in sorted(run_root.iterdir()):
        if source.is_file():
            shutil.copy2(source, evidence / source.name)

    registry = pd.read_csv(run_root / "strategy_registry.csv")
    gross = pd.read_parquet(run_root / "all_strategy_gross_returns.parquet")
    net = pd.read_parquet(run_root / "all_strategy_net_returns.parquet")
    sides = pd.read_parquet(run_root / "all_strategy_trade_sides.parquet")
    strategy_index: list[dict[str, Any]] = []
    for row in registry.to_dict(orient="records"):
        name = str(row["name"])
        hypothesis = str(row["hypothesis_id"])
        folder = master / "strategies" / hypothesis / name
        folder.mkdir(parents=True)
        (folder / "strategy_definition.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        pd.DataFrame(
            {
                "gross_return": gross[name],
                "net_return": net[name],
                "trade_sides": sides[name],
            }
        ).to_parquet(folder / "daily_evidence.parquet")
        (folder / "README.md").write_text(
            "\n".join(
                [
                    f"# {name}",
                    "",
                    f"- Hypothesis family: {hypothesis}",
                    f"- Product: {row['symbol']}",
                    f"- Timeframe: {row['timeframe_minutes']} minutes",
                    "- Status: rejected development variant; not approved for live trading",
                    "- Evidence: `daily_evidence.parquet` contains gross, costed, "
                    "and turnover series.",
                    "",
                    "This parameter variant is not counted as an independent strategy idea.",
                ]
            ),
            encoding="utf-8",
        )
        strategy_index.append(
            {
                "strategy": name,
                "hypothesis_id": hypothesis,
                "symbol": row["symbol"],
                "folder": folder.relative_to(master).as_posix(),
            }
        )

    source_snapshot = master / "source_snapshot"
    source_snapshot.mkdir()
    for relative in (
        "src/systematic_research/unique_hypothesis_factory.py",
        "tests/test_unique_hypothesis_factory.py",
        "research_program/RESEARCH_LOOP_PROTOCOL.md",
        "research_program/hypothesis_registry.csv",
        "research_program/learning_log.md",
        "pyproject.toml",
        "uv.lock",
    ):
        source = project_root / relative
        if source.exists():
            target = source_snapshot / relative.replace("/", "__")
            shutil.copy2(source, target)

    pd.DataFrame(strategy_index).to_csv(master / "strategy_index.csv", index=False)
    (master / "MASTER_INDEX.md").write_text(
        "\n".join(
            [
                f"# Unique Hypothesis Pass {research_pass} — Rejected Master Archive",
                "",
                f"- Run: `{run_id}`",
                f"- Strategy variants: {len(strategy_index)}",
                f"- Unique ideas: {len(manifest['unique_hypotheses'])}",
                f"- Audit passed: {manifest['audit_passed']}",
                f"- Net Sharpe: {manifest['portfolio']['sharpe']:.4f}",
                f"- Sealed year accessed: {manifest['sealed_year_accessed']}",
                "",
                "Every tested variant has its own folder under `strategies/<hypothesis>/`.",
                "This archive preserves a negative result and must not be represented "
                "as live-ready.",
            ]
        ),
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(item for item in master.rglob("*") if item.is_file()):
        checksums.append({"path": path.relative_to(master).as_posix(), "sha256": _sha256(path)})
    pd.DataFrame(checksums).to_csv(master / "SHA256SUMS.csv", index=False)
    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive a unique-hypothesis research run")
    parser.add_argument("run_id")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    output = archive_run(arguments.project_root.resolve(), arguments.run_id)
    print(f"Archived run: {output}")


if __name__ == "__main__":
    main()
