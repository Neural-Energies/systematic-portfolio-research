from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from systematic_research.temporal_validation import (
    SplitPolicy,
    create_temporal_split,
    load_development_panel,
    load_sealed_holdout,
    write_temporal_split,
)


def _sessions() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", "2026-08-20", freq="B")
    return pd.DataFrame(
        {
            "symbol": "ESU6",
            "trading_date": dates,
            "close_to_close_return": 0.001,
        }
    )


def test_holdout_is_twelve_months_and_strictly_after_development() -> None:
    development, holdout, _, boundaries = create_temporal_split(
        _sessions(), SplitPolicy(holdout_months=12, minimum_holdout_months=6, embargo_days=5)
    )
    assert boundaries.holdout_start == "2025-08-20"
    assert development["trading_date"].max() < holdout["trading_date"].min()
    assert set(development.index).isdisjoint(holdout.index)


def test_minimum_holdout_cannot_be_below_six_months() -> None:
    with pytest.raises(ValueError, match="at least six months"):
        SplitPolicy(holdout_months=5, minimum_holdout_months=5)


def test_holdout_access_is_denied_to_research_code(tmp_path: Path) -> None:
    write_temporal_split(_sessions(), tmp_path, tmp_path / "catalog", SplitPolicy())
    assert not load_development_panel(tmp_path).empty
    with pytest.raises(PermissionError, match="unavailable"):
        load_sealed_holdout(tmp_path)
    assert not load_sealed_holdout(tmp_path, sequential_evaluator=True).empty
