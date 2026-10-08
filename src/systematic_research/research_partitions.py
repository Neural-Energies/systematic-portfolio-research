"""Explicit calendar boundaries for chronological research samples."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

FITTING_PURPOSES = frozenset({"fit", "selection", "ranking", "threshold", "tuning"})
EVALUATION_PURPOSE = "evaluation"


class ForwardHoldoutError(PermissionError):
    """A fitting or premature evaluation step touched the sealed forward sample."""


@dataclass(frozen=True)
class ResearchPartitions:
    """Calendar windows from ``config/research_splits.yaml``."""

    timezone: str
    embargo_cash_sessions: int
    bar_timestamp_convention: str
    frozen_entries: str
    discovery_start: date
    discovery_end: date
    validation_start: date
    validation_end: date
    validation_already_used: bool
    contaminated_start: date
    contaminated_end: date
    contaminated_usable_as_holdout: bool
    forward_holdout_after: date
    forward_holdout_sealed: bool


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    section = payload[name]
    if not isinstance(section, dict):
        raise ValueError(f"{name} partition must be a mapping")
    return section


def load_research_partitions(path: Path | None = None) -> ResearchPartitions:
    """Load the shared discovery, validation, contaminated, and forward windows."""
    source = path or (_project_root() / "config" / "research_splits.yaml")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("research splits config must be a mapping")
    partitions = loaded["partitions"]
    if not isinstance(partitions, dict):
        raise ValueError("research splits config is missing partitions")
    discovery = _section(partitions, "discovery")
    validation = _section(partitions, "validation")
    contaminated = _section(partitions, "contaminated_2026")
    forward = _section(partitions, "forward_holdout")
    return ResearchPartitions(
        timezone=str(loaded["timezone"]),
        embargo_cash_sessions=int(loaded["embargo_cash_sessions"]),
        bar_timestamp_convention=str(loaded["bar_timestamp_convention"]),
        frozen_entries=str(loaded["frozen_entries"]),
        discovery_start=_as_date(discovery["start"]),
        discovery_end=_as_date(discovery["end"]),
        validation_start=_as_date(validation["start"]),
        validation_end=_as_date(validation["end"]),
        validation_already_used=bool(validation["already_used_for_selection_and_exit_tuning"]),
        contaminated_start=_as_date(contaminated["start"]),
        contaminated_end=_as_date(contaminated["end"]),
        contaminated_usable_as_holdout=bool(contaminated["usable_as_holdout"]),
        forward_holdout_after=_as_date(forward["strictly_after"]),
        forward_holdout_sealed=bool(forward["sealed"]),
    )


def as_research_clock(
    stamps: pd.DatetimeIndex,
    timezone: str = "America/New_York",
) -> pd.DatetimeIndex:
    """Convert stamps to the research zone.

    Timezone-aware values are converted. Naive values are calendar labels in
    ``timezone`` (session dates), not UTC instants.
    """
    index = pd.DatetimeIndex(stamps)
    if index.tz is None:
        return index.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    return index.tz_convert(timezone)


def _exceeds_forward_holdout(index: pd.DatetimeIndex, partitions: ResearchPartitions) -> bool:
    if len(index) == 0:
        return False
    local = as_research_clock(index, partitions.timezone)
    boundary = pd.Timestamp(partitions.forward_holdout_after).tz_localize(partitions.timezone)
    return bool((local.normalize() > boundary).any())


def default_forward_holdout_ledger(project_root: Path | None = None) -> Path:
    """Ledger created only by a successful one-time unlock."""
    root = project_root or _project_root()
    return root / "research_program" / "forward_holdout_ledger.json"


def _read_unlock(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ForwardHoldoutError(
            "FORWARD_HOLDOUT evaluation requires unlock_forward_holdout() first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("unlocked_at_utc"):
        raise ForwardHoldoutError(
            "FORWARD_HOLDOUT evaluation requires unlock_forward_holdout() first"
        )
    return payload


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def guard_research_sample(
    stamps: pd.DatetimeIndex,
    purpose: str,
    ledger_path: Path | None = None,
) -> None:
    """Refuse fitting on forward-holdout dates, and allow evaluation only after unlock.

    Samples that contain no forward-holdout timestamp return immediately and do
    not read or write the unlock ledger. Fitting stays illegal after unlock.
    The first evaluation of a forward-holdout timestamp consumes that unlock.
    """
    if purpose not in FITTING_PURPOSES and purpose != EVALUATION_PURPOSE:
        raise ValueError(f"Unknown research purpose: {purpose}")
    index = pd.DatetimeIndex(stamps)
    if len(index) == 0:
        return
    partitions = load_research_partitions()
    if not _exceeds_forward_holdout(index, partitions):
        return
    if purpose in FITTING_PURPOSES:
        raise ForwardHoldoutError(
            "FORWARD_HOLDOUT dates cannot be read by fitting, selection, ranking, "
            "threshold, or tuning"
        )
    path = ledger_path or default_forward_holdout_ledger()
    payload = _read_unlock(path)
    if int(payload.get("evaluation_count", 0)) != 0:
        raise ForwardHoldoutError("FORWARD_HOLDOUT evaluation has already been used")
    payload["evaluation_count"] = 1
    payload["evaluated_at_utc"] = datetime.now(UTC).isoformat()
    _replace_json(path, payload)


def frame_timestamps(frame: pd.DataFrame) -> pd.DatetimeIndex | None:
    """Prefer bar instants, then session dates, then a datetime index."""
    if "timestamp_utc" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["timestamp_utc"], utc=True))
    if "trading_date" in frame.columns:
        return pd.DatetimeIndex(pd.to_datetime(frame["trading_date"]))
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    return None


def guard_frame(
    frame: pd.DataFrame,
    purpose: str,
    ledger_path: Path | None = None,
) -> None:
    """Apply :func:`guard_research_sample` to one loaded frame."""
    stamps = frame_timestamps(frame)
    if stamps is None:
        return
    guard_research_sample(stamps, purpose, ledger_path=ledger_path)


def guard_frames(
    frames: list[pd.DataFrame],
    purpose: str,
    ledger_path: Path | None = None,
) -> None:
    """Guard several frames as one sample so evaluation is consumed at most once."""
    pieces: list[pd.DatetimeIndex] = []
    for frame in frames:
        stamps = frame_timestamps(frame)
        if stamps is not None and len(stamps):
            pieces.append(as_research_clock(stamps))
    if not pieces:
        guard_research_sample(pd.DatetimeIndex([]), purpose, ledger_path=ledger_path)
        return
    encoded = [np.asarray(piece, dtype="datetime64[ns]").view(np.int64) for piece in pieces]
    combined = pd.DatetimeIndex(np.concatenate(encoded), tz=pieces[0].tz)
    guard_research_sample(combined, purpose, ledger_path=ledger_path)


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def unlock_forward_holdout(
    ledger_path: Path,
    candidate_path: Path | None = None,
    commit: str | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Record one unlock. A second call raises even if the first file is empty."""
    root = project_root or _project_root()
    partitions = load_research_partitions()
    candidate = candidate_path or (root / partitions.frozen_entries)
    resolved_commit = commit if commit is not None else _git_commit(root)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    record_sha256 = hashlib.sha256(f"{candidate_sha256}\n{resolved_commit}".encode()).hexdigest()
    payload: dict[str, Any] = {
        "unlocked_at_utc": datetime.now(UTC).isoformat(),
        "candidate_sha256": candidate_sha256,
        "commit": resolved_commit,
        "record_sha256": record_sha256,
        "candidate_path": candidate.as_posix(),
        "evaluation_count": 0,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        descriptor = os.open(ledger_path, flags, 0o644)
    except FileExistsError as exc:
        raise PermissionError("Forward holdout unlock already recorded") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return payload


def nq_calendar_split(index: pd.DatetimeIndex) -> NDArray[np.object_]:
    """Label NQ rows the way the frozen discovery scripts did, then seal later bars.

    Through 7 October 2026 the labels stay ``in_sample``, ``validation``, and
    ``final_confirmation``. Later bars are ``forward_holdout``. Callers still
    apply the cash-session embargo after this assignment.
    """
    partitions = load_research_partitions()
    local = as_research_clock(pd.DatetimeIndex(index), partitions.timezone)
    labels: NDArray[np.object_] = np.full(len(local), "forward_holdout", dtype=object)
    if len(local) == 0:
        return labels
    normalized = local.normalize()
    zone = partitions.timezone
    discovery_end = pd.Timestamp(partitions.discovery_end).tz_localize(zone)
    validation_start = pd.Timestamp(partitions.validation_start).tz_localize(zone)
    validation_end = pd.Timestamp(partitions.validation_end).tz_localize(zone)
    contaminated_start = pd.Timestamp(partitions.contaminated_start).tz_localize(zone)
    contaminated_end = pd.Timestamp(partitions.contaminated_end).tz_localize(zone)
    labels[np.asarray(normalized <= discovery_end)] = "in_sample"
    labels[np.asarray((normalized >= validation_start) & (normalized <= validation_end))] = (
        "validation"
    )
    labels[np.asarray((normalized >= contaminated_start) & (normalized <= contaminated_end))] = (
        "final_confirmation"
    )
    return labels


def purged_training_rows(
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    embargo_sessions: int = 5,
) -> NDArray[np.bool_]:
    """All earlier rows whose full labels finish before a cash-session embargo."""
    if starts.tz is None or ends.tz is None or sessions.tz is None or cutoff.tz is None:
        raise ValueError("Explicit timezone required")
    if len(starts) != len(ends) or embargo_sessions < 0:
        raise ValueError("Invalid label clock or embargo")
    prior = sessions[sessions < cutoff]
    if embargo_sessions and len(prior) < embargo_sessions:
        return np.zeros(len(starts), dtype=bool)
    boundary = prior[-embargo_sessions] if embargo_sessions else cutoff
    return np.asarray((starts < boundary) & (ends <= boundary), dtype=bool)


def require_before_lockout(index: pd.DatetimeIndex, lockout_start: pd.Timestamp) -> None:
    """Refuse a fit/evaluation input containing even one reserved timestamp."""
    if index.tz is None or lockout_start.tz is None:
        raise ValueError("Explicit timezone required")
    if len(index) and index.max() >= lockout_start:
        raise ValueError("Reserved lockout rows reached the component research input")
