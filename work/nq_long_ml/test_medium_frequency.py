"""Frequency, holding, capacity and daily mark-to-market contracts."""

import numpy as np
import pandas as pd
from medium_frequency import (
    build_labels,
    daily_policy,
    hypothetical_exits,
    marked_equity,
    trade_summary,
)


def predictions():
    index = pd.date_range("2024-08-26 12:00", periods=7, freq="h", tz="UTC")
    session = pd.Timestamp("2024-08-25")
    return pd.DataFrame(
        {
            "session": session,
            "probability": 0.52,
            "selected_threshold": 0.66,
            "execution_valid": True,
            "entry_time": index + pd.Timedelta(minutes=1),
            "exit_time": index + pd.Timedelta(hours=5),
            "entry_price": 100.0,
            "exit_price": 102.0,
            "observed_minutes_held": 300,
            "fold": 1,
        },
        index=index,
    )


def test_daily_quota_uses_deadline_without_future_ranking():
    frame = predictions()
    trades = daily_policy(frame, 0.66, True)
    assert len(trades) == 1
    assert trades.iloc[0].decision_time == pd.Timestamp("2024-08-26 16:00", tz="UTC")
    assert trades.iloc[0].quota_fallback
    assert daily_policy(frame, 0.66, False).empty
    changed = frame.copy()
    changed.loc[changed.index[-1], "probability"] = 0.99
    pd.testing.assert_frame_equal(trades, daily_policy(changed, 0.66, True))


def test_gate_can_enter_before_deadline():
    frame = predictions()
    frame.loc[frame.index[0], "probability"] = 0.70
    result = daily_policy(frame, 0.66, True)
    assert result.iloc[0].decision_time == frame.index[0]
    assert not result.iloc[0].quota_fallback


def test_denominator_counts_days_without_trades():
    trades = daily_policy(predictions(), 0.66, True)
    days = pd.DatetimeIndex(["2024-08-25", "2024-08-26"])
    result = trade_summary(trades, days)
    assert result["trades_per_day"] == 0.5
    assert result["missing_days"] == 1
    assert result["net_dollars"] == 15  # $40 gross less $25 costs


def test_long_horizon_falls_back_during_expiry_month():
    times = pd.date_range("2024-03-05 12:00", periods=600, freq="min", tz="UTC")
    minute = pd.DataFrame({"open": 100.0, "close": 100.0}, index=times)
    grid = pd.DataFrame({"feature": 1.0}, index=pd.DatetimeIndex(["2024-03-05 13:00"], tz="UTC"))
    result = build_labels(minute, grid, 5)
    assert result.iloc[0].effective_days == 0
    assert result.iloc[0].entry_time == pd.Timestamp("2024-03-05 13:01", tz="UTC")
    assert result.iloc[0].max_exit_position - result.iloc[0].entry_position == 240


def test_exit_signal_cannot_close_before_four_trading_hours():
    times = pd.date_range("2024-08-26 12:00", periods=600, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": np.arange(600) + 100.0, "close": np.arange(600) + 100.0}, index=times
    )
    index = times[::60][:8]
    frame = pd.DataFrame(
        {
            "probability": 0.30,
            "execution_valid": True,
            "entry_position": np.arange(8) * 60 + 1,
            "max_exit_position": 550,
            "entry_time": times[np.arange(8) * 60 + 1],
        },
        index=index,
    )
    result = hypothetical_exits(frame, minute)
    assert result.iloc[0].observed_minutes_held == 240
    assert result.iloc[0].exit_time == times[241]


def test_marked_equity_includes_open_loss():
    t = pd.Timestamp("2024-08-26 20:00", tz="UTC")
    minute = pd.DataFrame({"close": [90.0]}, index=pd.DatetimeIndex([t]))
    trades = pd.DataFrame(
        {
            "entry_time": [t - pd.Timedelta(hours=1)],
            "exit_time": [t + pd.Timedelta(days=1)],
            "entry_price": [100.0],
            "exit_price": [105.0],
        }
    )
    equity = marked_equity(trades, minute)
    assert equity.iloc[0].equity == -212.5
    assert equity.iloc[0].drawdown == 212.5


def test_daily_quota_never_exceeds_five_open_contracts():
    frame = predictions().iloc[:1].copy()
    copies = []
    for offset in range(7):
        part = frame.copy()
        part.index = part.index + pd.Timedelta(days=offset)
        part["entry_time"] = part["entry_time"] + pd.Timedelta(days=offset)
        part["exit_time"] = part["exit_time"] + pd.Timedelta(days=100)
        part["session"] = part["session"] + pd.Timedelta(days=offset)
        part["probability"] = 0.70
        copies.append(part)
    trades = daily_policy(pd.concat(copies), 0.66, True)
    assert len(trades) == 5
    assert trades["concurrent_at_entry"].max() == 5


def test_missing_quote_keeps_decision_session_identity():
    times = pd.date_range("2025-01-09 23:01", periods=600, freq="min", tz="UTC")
    minute = pd.DataFrame({"open": 100.0, "close": 100.0}, index=times)
    grid = pd.DataFrame({"feature": 1.0}, index=pd.DatetimeIndex(["2025-01-09 15:00"], tz="UTC"))
    result = build_labels(minute, grid, 0)
    assert not result.iloc[0].execution_valid
    assert result.iloc[0].session == pd.Timestamp("2025-01-08")
    assert result.iloc[0].entry_time == pd.Timestamp("2025-01-09 23:01", tz="UTC")
