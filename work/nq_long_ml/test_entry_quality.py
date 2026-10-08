import numpy as np
import pandas as pd
from adaptive_entries import period_scope
from clock_entries import clock_signal
from entry_quality import event_metrics


def test_positive_excursion_is_not_direction_success():
    days = pd.date_range("2024-01-01", periods=30, tz="UTC")
    frame = pd.DataFrame(
        {
            "scorable": True,
            "anchor": days,
            "entry_open": 100.0,
            "exit_close": np.r_[np.full(20, 101.0), np.full(10, 99.0)],
            "control": 0.5,
            "mfe_points": 3.0,
            "mae_points": 1.0,
            "atr": 2.0,
        },
        index=days,
    )
    m = event_metrics(frame, days)
    assert m["direction_success"] == 20 / 30
    assert m["favorable_1atr_rate"] == 1
    assert m["sessions_without_signal"] == 0
    assert m["median_forward_points"] == 1


def test_recurring_entry_sees_no_future_quote():
    clock = pd.date_range("2024-01-01", periods=15, freq="5min", tz="UTC")
    frame = pd.DataFrame({"anchor": clock.normalize()}, index=clock)
    pd.testing.assert_series_equal(
        clock_signal(frame, 15).iloc[:8], clock_signal(frame.iloc[:8], 15)
    )
    assert list(frame.index[clock_signal(frame, 15)]) == [clock[3]]


def test_session_cutoff_excludes_partial_sessions():
    clock = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {"anchor": clock - pd.Timedelta(hours=1), "session_end": clock + pd.Timedelta(hours=1)},
        index=clock,
    )
    result = period_scope(frame, clock[0], clock[2])
    assert list(result.index) == [clock[1]]
