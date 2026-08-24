from pathlib import Path

import pytest

from systematic_research.sealed_evaluator import assert_holdout_unused


def test_sealed_evaluator_rejects_second_run(tmp_path: Path) -> None:
    candidate = {
        "candidate": {"holdout_accessed": True},
        "holdout_policy": {"frozen": True, "holdout_run_count": 1},
    }
    with pytest.raises(PermissionError, match="already been evaluated"):
        assert_holdout_unused(candidate, tmp_path / "result.json")


def test_sealed_evaluator_requires_frozen_candidate(tmp_path: Path) -> None:
    candidate = {
        "candidate": {"holdout_accessed": False},
        "holdout_policy": {"frozen": False, "holdout_run_count": 0},
    }
    with pytest.raises(PermissionError, match="must be frozen"):
        assert_holdout_unused(candidate, tmp_path / "result.json")
