import numpy as np

from systematic_research.mae_brackets import bracket_outcomes


def test_same_minute_stop_and_target_is_stop_first() -> None:
    clock = np.array([0], dtype=np.int64)
    prices = np.array([[100.0, 110.0, 90.0, 106.0]])
    result = bracket_outcomes(
        clock,
        prices,
        clock,
        np.array([60_000_000_000]),
        np.array([105.0]),
        np.array([5.0]),
        0,
        25.0,
        0.0,
        0.25,
        0.0,
    )
    assert result[0, 11] == -1
    assert result[0, 5] == 94.75
    assert result[0, 1] == -130


def test_stop_gap_fills_at_worse_open() -> None:
    clock = np.array([0, 60_000_000_000], dtype=np.int64)
    prices = np.array([[100.0, 102.0, 99.0, 101.0], [90.0, 92.0, 88.0, 91.0]])
    result = bracket_outcomes(
        clock,
        prices,
        clock[:1],
        np.array([120_000_000_000]),
        np.array([110.0]),
        np.array([5.0]),
        0,
        25.0,
        0.0,
        0.25,
        0.0,
    )
    assert result[0, 5] == 89.75
    assert result[0, 1] == -230


def test_no_stop_reproduces_existing_exit_engine() -> None:
    from systematic_research.execution_stress import minute_outcomes

    clock = np.array([0, 60_000_000_000], dtype=np.int64)
    prices = np.array([[100.0, 102.0, 99.0, 101.0], [101.0, 106.0, 100.0, 105.0]])
    common = (clock, prices, clock[:1], np.array([120_000_000_000]), np.array([105.0]))
    baseline = minute_outcomes(*common, 0, 25.0, 0.0, 0.25, 0.0)
    brackets = bracket_outcomes(*common, np.array([np.inf]), 0, 25.0, 0.0, 0.25, 0.0)
    np.testing.assert_allclose(brackets[:, :11], baseline)


def test_target_gap_precedes_later_low_in_same_minute() -> None:
    clock = np.array([0, 60_000_000_000], dtype=np.int64)
    prices = np.array([[100.0, 102.0, 99.0, 101.0], [110.0, 112.0, 90.0, 91.0]])
    result = bracket_outcomes(
        clock,
        prices,
        clock[:1],
        np.array([120_000_000_000]),
        np.array([105.0]),
        np.array([5.0]),
        0,
        25.0,
        0.0,
        0.0,
        0.0,
    )
    assert result[0, 11] == 1
    assert result[0, 5] == 105


def test_stop_losses_are_included_in_trade_kpis() -> None:
    import pandas as pd

    from systematic_research.trade_report_metrics import trade_kpis

    days = pd.date_range("2025-01-06", periods=2, freq="B", tz="UTC")
    trades = pd.DataFrame(
        {
            "anchor": days,
            "status": ["stop", "target"],
            "net_dollars": [-100.0, 200.0],
            "holding_minutes": [2.0, 3.0],
            "target_points": [10.0, 10.0],
        }
    )
    result, _ = trade_kpis(trades, days)
    assert result["trades"] == 2
    assert result["net_profit"] == 100
    assert result["profit_factor"] == 2
