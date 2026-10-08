import numpy as np
import pandas as pd

from systematic_research.reference_backtest import backtest_reference


def fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    clock = pd.date_range("2024-01-02 14:30", periods=10, freq="min", tz="UTC")
    minute = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=clock)
    decisions = clock[[0, 2, 5]]
    opportunities = pd.DataFrame(
        {
            "entry_open": 100.0,
            "target_points": 2.0,
            "planned_exit": clock[[5, 7, 9]],
            "anchor": clock[0],
            "split": "test",
        },
        index=decisions,
    )
    return minute, opportunities, pd.Series(True, index=decisions)


def test_target_exit_releases_position_and_cost_is_applied_once() -> None:
    minute, opportunities, signals = fixture()
    minute.loc[minute.index[1], "high"] = 102.0
    trades, equity = backtest_reference(minute, opportunities, signals)
    assert len(trades) == 2
    assert trades.iloc[0].status == "target"
    assert trades.iloc[0].exit_time == minute.index[2]
    assert trades.iloc[0].net_dollars == 15.0
    assert trades.iloc[1].entry_time == minute.index[2]
    assert trades.iloc[1].net_dollars == -25.0
    assert equity.equity.iloc[-1] == -10.0


def test_missing_future_data_does_not_cancel_entry_or_free_position() -> None:
    minute, opportunities, signals = fixture()
    minute = minute.drop(minute.index[1])
    trades, _ = backtest_reference(minute, opportunities, signals)
    assert len(trades) == 2
    assert trades.iloc[0].status == "unresolved"
    assert trades.iloc[1].entry_time == opportunities.index[2]


def test_penetration_and_latency_do_not_move_frozen_target() -> None:
    minute, opportunities, signals = fixture()
    signals.iloc[1:] = False
    minute.loc[minute.index[1], ["open", "high", "close"]] = [101.0, 102.0, 101.0]
    trades, _ = backtest_reference(minute, opportunities, signals, latency_minutes=1)
    assert trades.iloc[0].target_price == 102.0
    assert trades.iloc[0].net_dollars == -5.0
    strict, _ = backtest_reference(minute, opportunities, signals, penetration=0.25)
    assert strict.iloc[0].status == "timeout"
    assert np.isfinite(strict.iloc[0].net_dollars)
