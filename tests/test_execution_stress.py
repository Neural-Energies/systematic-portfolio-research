import numpy as np

from systematic_research.execution_stress import marked_drawdowns, minute_outcomes, schedule_events


def test_delayed_limit_requires_penetration_and_market_entry_slippage() -> None:
    clock = np.arange(4, dtype=np.int64) * 60_000_000_000
    quotes = np.array(
        [
            [100, 101, 99, 100],
            [101, 103, 100, 102],
            [102, 103.25, 101, 102.5],
            [102.5, 104, 102, 103],
        ],
        dtype=float,
    )
    outcome = minute_outcomes(
        clock,
        quotes,
        np.array([0], dtype=np.int64),
        np.array([240_000_000_000], dtype=np.int64),
        np.array([103.0]),
        1,
        25,
        0.25,
        0.25,
        0.25,
    )[0]
    assert outcome[0] == 180_000_000_000
    assert outcome[1] == 10  # (103 - 101.25) * 20 - 25
    assert outcome[2] == 1 and outcome[3] == 1
    assert outcome[6] == 1.25


def test_missing_selected_path_stays_unknown_and_blocks_next_entry() -> None:
    clock = np.array([0, 120_000_000_000, 180_000_000_000], dtype=np.int64)
    quotes = np.array([[100, 101, 99, 100]] * 3, dtype=float)
    entries = np.array([0, 120_000_000_000], dtype=np.int64)
    ends = np.array([180_000_000_000, 240_000_000_000], dtype=np.int64)
    outcome = minute_outcomes(
        clock, quotes, entries, ends, np.array([110.0, 110.0]), 0, 25, 0, 0, 0
    )
    assert outcome[0, 3] == 0 and np.isnan(outcome[0, 1])
    assert schedule_events(np.array([True, True]), entries, outcome).tolist() == [0]


def test_timeout_costs_and_final_close_not_next_open() -> None:
    clock = np.arange(3, dtype=np.int64) * 60_000_000_000
    quotes = np.array([[100, 101, 99, 100], [100, 102, 99, 101], [150, 151, 149, 150]], dtype=float)
    outcome = minute_outcomes(
        clock,
        quotes,
        np.array([0], dtype=np.int64),
        np.array([120_000_000_000], dtype=np.int64),
        np.array([110.0]),
        0,
        25,
        0.25,
        0.25,
        0,
    )[0]
    assert outcome[5] == 100.75 and outcome[1] == -15
    assert outcome[2] == 0 and outcome[8] == 2


def test_marked_drawdown_carries_previous_trade_peak() -> None:
    clock = np.arange(4, dtype=np.int64) * 60_000_000_000
    quotes = np.array(
        [[100, 100, 99, 99], [99, 102, 98, 102], [102, 103, 99, 100], [100, 101, 95, 96]],
        dtype=float,
    )
    entries = np.array([0, 120_000_000_000], dtype=np.int64)
    ends = np.array([120_000_000_000, 240_000_000_000], dtype=np.int64)
    outcomes = minute_outcomes(
        clock, quotes, entries, ends, np.array([200.0, 200.0]), 0, 0, 0, 0, 0
    )
    chosen = schedule_events(np.array([True, True]), entries, outcomes)
    close_dd, low_bound = marked_drawdowns(clock, quotes, entries, outcomes, chosen, 0, 0)
    assert close_dd == 120 and low_bound == 140
