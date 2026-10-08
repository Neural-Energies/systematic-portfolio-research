import pandas as pd

from systematic_research.rth_bar_grid import completed_rth_bars, minute_start_clock, rth_bar_grid


def test_end_label_buys_actual_open_and_bar_is_known_only_at_close() -> None:
    labels = pd.date_range("2026-04-23 09:31", periods=5, freq="min", tz="America/New_York")
    clock = minute_start_clock(labels, labels_at_end=True)
    m = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": 110,
            "low": 99,
            "close": [101, 102, 103, 104, 105],
            "volume": 10,
        },
        index=clock,
    )
    schedule = pd.DataFrame({"open": [clock[0]], "close": [labels[-1]]})
    bars = completed_rth_bars(m, rth_bar_grid(schedule, 5))
    assert bars.index[0].strftime("%H:%M") == "09:30"
    assert bars.open.iloc[0] == 100
    assert bars.available_at.iloc[0].strftime("%H:%M") == "09:35"
    assert bars.close.iloc[0] == 105


def test_hour_and_four_hour_grids_anchor_at_0930_and_cap_close() -> None:
    opened = pd.Timestamp("2026-04-23 09:30", tz="America/New_York")
    closed = opened + pd.Timedelta(minutes=390)
    s = pd.DataFrame({"open": [opened], "close": [closed]})
    assert [len(rth_bar_grid(s, x)) for x in (5, 15, 60, 240)] == [78, 26, 7, 2]
    h = rth_bar_grid(s, 60)
    assert h.index[1].strftime("%H:%M") == "10:30"
    assert h.planned_exit.iloc[-1] == closed
    four = rth_bar_grid(s, 240)
    assert four.index[1].strftime("%H:%M") == "13:30"
    assert four.planned_exit.iloc[-1] == closed


def test_missing_minute_does_not_create_completed_bar() -> None:
    clock = pd.date_range("2026-04-23 09:30", periods=5, freq="min", tz="America/New_York")
    m = pd.DataFrame({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}, index=clock)
    s = pd.DataFrame({"open": [clock[0]], "close": [clock[-1] + pd.Timedelta(minutes=1)]})
    assert completed_rth_bars(m.drop(clock[2]), rth_bar_grid(s, 5)).empty
