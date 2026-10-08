from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from systematic_research.research_partitions import (
    ForwardHoldoutError,
    guard_research_sample,
    load_research_partitions,
    nq_calendar_split,
    unlock_forward_holdout,
)

FITTING = ("fit", "selection", "ranking", "threshold", "tuning")


def test_shared_config_defines_the_approved_partitions() -> None:
    partitions = load_research_partitions()
    assert partitions.timezone == "America/New_York"
    assert partitions.bar_timestamp_convention == "pending_human_confirmation"
    assert partitions.discovery_start == date(2022, 1, 1)
    assert partitions.discovery_end == date(2024, 12, 31)
    assert partitions.validation_start == date(2025, 1, 1)
    assert partitions.validation_end == date(2025, 12, 31)
    assert partitions.validation_already_used
    assert partitions.contaminated_start == date(2026, 1, 1)
    assert partitions.contaminated_end == date(2026, 10, 7)
    assert not partitions.contaminated_usable_as_holdout
    assert partitions.forward_holdout_after == date(2026, 10, 7)
    assert partitions.forward_holdout_sealed
    assert partitions.frozen_entries.endswith("LOCKED_ENTRIES.json")


def test_fitting_allows_discovery_validation_and_contaminated_dates() -> None:
    stamps = pd.DatetimeIndex(["2024-06-01", "2025-06-01", "2026-10-07 23:59"]).tz_localize(
        "America/New_York"
    )
    for purpose in FITTING:
        guard_research_sample(stamps, purpose)


def test_fitting_blocks_forward_holdout_including_the_utc_boundary() -> None:
    allowed = pd.DatetimeIndex([pd.Timestamp("2026-10-08 03:59", tz="UTC")])
    blocked = (
        pd.DatetimeIndex([pd.Timestamp("2026-10-08", tz="America/New_York")]),
        pd.DatetimeIndex([pd.Timestamp("2026-10-08 04:00", tz="UTC")]),
    )
    for purpose in FITTING:
        guard_research_sample(allowed, purpose)
        for stamps in blocked:
            with pytest.raises(ForwardHoldoutError, match="FORWARD_HOLDOUT"):
                guard_research_sample(stamps, purpose)


def test_a_forward_label_end_blocks_fitting() -> None:
    stamps = pd.DatetimeIndex(["2025-06-01", "2026-10-08 09:30"], tz="America/New_York")
    with pytest.raises(ForwardHoldoutError):
        guard_research_sample(stamps, "fit")


def test_one_forward_timestamp_blocks_ranking() -> None:
    stamps = pd.DatetimeIndex(["2024-01-02", "2026-10-08"], tz="America/New_York")
    with pytest.raises(ForwardHoldoutError):
        guard_research_sample(stamps, "ranking")


def test_empty_index_and_naive_calendar_dates() -> None:
    guard_research_sample(pd.DatetimeIndex([]), "fit")
    guard_research_sample(pd.to_datetime(["2026-10-07"]), "selection")
    guard_research_sample(pd.to_datetime(["2026-10-07 23:59"]), "tuning")
    with pytest.raises(ForwardHoldoutError):
        guard_research_sample(pd.to_datetime(["2026-10-08"]), "threshold")
    with pytest.raises(ValueError, match="Unknown research purpose"):
        guard_research_sample(pd.DatetimeIndex([]), "explore")


def test_seen_dates_do_not_touch_the_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    stamps = pd.DatetimeIndex(["2026-10-07 23:59"], tz="America/New_York")
    guard_research_sample(stamps, "evaluation", ledger_path=ledger)
    assert not ledger.exists()


def test_unlock_is_once_and_evaluation_requires_it(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    candidate = tmp_path / "frozen.json"
    candidate.write_text('{"frozen": true}\n', encoding="utf-8")
    holdout = pd.DatetimeIndex(["2026-10-08"], tz="America/New_York")
    with pytest.raises(ForwardHoldoutError, match="unlock_forward_holdout"):
        guard_research_sample(holdout, "evaluation", ledger_path=ledger)

    record = unlock_forward_holdout(ledger, candidate_path=candidate, commit="abc123")
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert record["commit"] == "abc123"
    assert record["candidate_sha256"] == candidate_sha
    assert (
        record["record_sha256"] == hashlib.sha256(f"{candidate_sha}\nabc123".encode()).hexdigest()
    )
    assert record["unlocked_at_utc"]
    assert record["evaluation_count"] == 0
    with pytest.raises(PermissionError, match="already recorded"):
        unlock_forward_holdout(ledger, candidate_path=candidate, commit="abc123")

    guard_research_sample(holdout, "evaluation", ledger_path=ledger)
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["evaluation_count"] == 1
    assert saved["evaluated_at_utc"]
    assert saved["candidate_sha256"] == candidate_sha
    with pytest.raises(ForwardHoldoutError, match="already been used"):
        guard_research_sample(holdout, "evaluation", ledger_path=ledger)
    with pytest.raises(ForwardHoldoutError, match="FORWARD_HOLDOUT"):
        guard_research_sample(holdout, "fit", ledger_path=ledger)
    for purpose in ("selection", "ranking", "threshold", "tuning"):
        with pytest.raises(ForwardHoldoutError):
            guard_research_sample(holdout, purpose, ledger_path=ledger)


def test_unlock_hashes_the_real_frozen_entries(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    root = Path(__file__).resolve().parents[1]
    candidate = root / "saved_strategies/NQ_RTH_100_LOCKED_20261007/LOCKED_ENTRIES.json"
    record = unlock_forward_holdout(ledger, candidate_path=candidate, project_root=root)
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    assert record["commit"] == expected_commit
    assert record["candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert not (root / "research_program" / "forward_holdout_ledger.json").exists()


def test_nq_calendar_split_matches_the_frozen_labels_then_seals() -> None:
    local = pd.DatetimeIndex(
        ["2022-01-01", "2024-06-03 14:00", "2025-03-03 14:00", "2026-10-07 23:59", "2026-10-08"]
    ).tz_localize("America/New_York")
    assert list(nq_calendar_split(local)) == [
        "in_sample",
        "in_sample",
        "validation",
        "final_confirmation",
        "forward_holdout",
    ]
    boundary = pd.DatetimeIndex(["2026-10-08 03:59", "2026-10-08 04:00"], tz="UTC")
    assert list(nq_calendar_split(boundary)) == ["final_confirmation", "forward_holdout"]
    assert list(nq_calendar_split(pd.DatetimeIndex([]))) == []
