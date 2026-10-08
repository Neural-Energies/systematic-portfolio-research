import numpy as np
import pandas as pd
import pytest

from systematic_research.auction_entry import (
    entry_path_diagnostics,
    profile_histogram,
    profile_levels,
)


def test_profile_preserves_volume_and_builds_contiguous_value() -> None:
    bars = pd.DataFrame(
        {
            "low": [100, 101, 102],
            "high": [100, 101, 102],
            "close": [100, 101, 102],
            "volume": [50, 30, 20],
        }
    )
    hist = profile_histogram(bars, 1, "uniform")
    assert sum(hist.values()) == 100
    assert profile_levels(hist, 1) == (100.5, 100, 102)
    wider = pd.DataFrame({"low": [100], "high": [102], "close": [102], "volume": [90]})
    assert profile_histogram(wider, 1, "uniform") == {100: 30, 101: 30, 102: 30}
    assert profile_histogram(wider, 1, "close") == {102: 90}


def test_path_orders_touches_and_marks_intraminute_ambiguity() -> None:
    clock = pd.date_range("2024-01-01", periods=5, freq="min", tz="UTC")
    quotes = pd.DataFrame(
        {"open": 100.0, "high": [101, 110, 101, 110, 101], "low": [99, 99, 90, 90, 99]}, index=clock
    )
    result = entry_path_diagnostics(
        quotes,
        pd.DatetimeIndex([clock[0], clock[3]]),
        pd.DatetimeIndex([clock[3], clock[4]]),
        np.array([10.0, 10.0]),
    )
    assert result.iloc[0].up_first == 1
    assert result.iloc[0].adverse_before_up_points == 1
    assert result.iloc[1].ambiguous == 1
    assert result.iloc[1].up_first == 0
    assert result.iloc[1].reached_up == 1


def test_gap_invalidates_path_and_nonpositive_distance_rejected() -> None:
    clock = pd.date_range("2024-01-01", periods=3, freq="min", tz="UTC")
    quotes = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0}, index=clock).drop(clock[1])
    a = pd.DatetimeIndex([clock[0]])
    b = pd.DatetimeIndex([clock[-1] + pd.Timedelta(minutes=1)])
    assert not entry_path_diagnostics(quotes, a, b, np.array([10.0])).iloc[0].path_scorable
    with pytest.raises(ValueError):
        entry_path_diagnostics(quotes, a, b, np.array([0.0]))


def test_wilder_seed_and_recursion() -> None:
    from systematic_research.auction_entry import wilder_mean

    values = wilder_mean(pd.Series([1.0, 2.0, 3.0, 4.0]), 3)
    assert values.iloc[:2].isna().all()
    assert values.iloc[2] == 2
    assert values.iloc[3] == 8 / 3


def test_known_touch_survives_later_missing_data() -> None:
    clock = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    quotes = pd.DataFrame(
        {"open": 100.0, "high": [100.0, 111.0, 100.0, 100.0], "low": 99.0}, index=clock
    ).drop(clock[2])
    result = entry_path_diagnostics(
        quotes,
        pd.DatetimeIndex([clock[0]]),
        pd.DatetimeIndex([clock[0] + pd.Timedelta(minutes=4)]),
        np.array([10.0]),
    )
    assert result.iloc[0].path_scorable
    assert result.iloc[0].up_first == 1
    assert result.iloc[0].reach_scorable


def test_gap_before_first_touch_has_unknown_order_but_known_reach() -> None:
    clock = pd.date_range("2024-01-01", periods=4, freq="min", tz="UTC")
    quotes = pd.DataFrame(
        {"open": 100.0, "high": [100.0, 100.0, 111.0, 100.0], "low": 99.0}, index=clock
    ).drop(clock[1])
    result = entry_path_diagnostics(
        quotes,
        pd.DatetimeIndex([clock[0]]),
        pd.DatetimeIndex([clock[0] + pd.Timedelta(minutes=4)]),
        np.array([10.0]),
    )
    assert not result.iloc[0].path_scorable
    assert result.iloc[0].reach_scorable
    assert result.iloc[0].reached_up == 1
