import numpy as np
import pandas as pd
from adaptive_entries import period_trades, stable_score


def test_future_label_cannot_enter_past_selection():
    clock = pd.date_range("2024-01-01", periods=8, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "planned_exit": clock + pd.Timedelta(minutes=15),
            "scorable": True,
            "anchor": clock.normalize(),
            "split": "development",
            "quarter": 0,
            "slot": 0,
            "vol_bucket": 0,
            "gross_dollars": np.arange(8) * 1000.0,
        },
        index=clock,
    )
    signal = pd.Series(True, index=clock)
    a = period_trades(frame, signal, clock[0], clock[6])
    frame.loc[clock[4] :, "gross_dollars"] = 999999
    b = period_trades(frame, signal, clock[0], clock[6])
    assert list(a.index) == [clock[0], clock[3]]
    assert not a.scorable.iloc[-1]
    pd.testing.assert_series_equal(a.control, b.control)


def test_missing_one_session_disqualifies_selection():
    days = pd.date_range("2024-01-01", periods=30, tz="UTC")
    frame = pd.DataFrame(
        {
            "anchor": days[:-1],
            "scorable": True,
            "net_dollars": np.arange(29) + 1,
            "stress_dollars": 1,
            "delayed_net_dollars": 1,
        }
    )
    assert stable_score(frame, days) == -np.inf
