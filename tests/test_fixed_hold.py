"""Execution timing and missing-data decisions are separate from future outcomes."""

import numpy as np
import pandas as pd
import pytest

from systematic_research.fixed_hold import fixed_hold_outcomes, select_nonoverlapping


def quotes() -> pd.DataFrame:
    index = pd.date_range("2024-08-26 13:30", periods=30, freq="min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0}, index=index)


def test_last_held_minute_close_is_exit_not_following_minute() -> None:
    minute = quotes()
    minute.loc[minute.index[4], "close"] = 103
    minute.loc[minute.index[4], "high"] = 103
    minute.loc[minute.index[5], ["open", "close", "high"]] = 999
    entry = pd.DatetimeIndex(minute.index[:1])
    result = fixed_hold_outcomes(minute, entry, entry + pd.Timedelta(minutes=5))
    assert result.iloc[0].net_dollars == 35  # 3 points * $20 minus $25
    assert result.iloc[0].mfe_points == 3


def test_missing_future_quote_invalidates_outcome_but_keeps_position_schedule() -> None:
    minute = quotes().drop(quotes().index[2])
    entry = pd.date_range("2024-08-26 13:30", periods=3, freq="5min", tz="UTC")
    exits = entry + pd.Timedelta(minutes=15)
    result = fixed_hold_outcomes(minute, entry, exits)
    assert not result.iloc[0].scorable
    assert np.isnan(result.iloc[0].net_dollars)
    chosen = select_nonoverlapping(
        np.ones(3, dtype=bool),
        entry.to_numpy(dtype="datetime64[ns]").astype("int64"),
        exits.to_numpy(dtype="datetime64[ns]").astype("int64"),
    )
    assert np.array_equal(chosen, [0])  # cannot use the future gap to take the 13:35 trade instead


def test_nonoverlap_allows_new_entry_at_previous_exit() -> None:
    entry = pd.date_range("2024-08-26 13:30", periods=5, freq="5min", tz="UTC")
    chosen = select_nonoverlapping(
        np.ones(5, dtype=bool),
        entry.to_numpy(dtype="datetime64[ns]").astype("int64"),
        (entry + pd.Timedelta(minutes=10)).to_numpy(dtype="datetime64[ns]").astype("int64"),
    )
    assert np.array_equal(chosen, [0, 2, 4])


def test_invalid_schedule_rejected() -> None:
    with pytest.raises(ValueError, match="schedule"):
        select_nonoverlapping(
            np.array([True]), np.array([1], dtype=np.int64), np.array([1], dtype=np.int64)
        )
