from __future__ import annotations

import pandas as pd

from systematic_research.saved_strategy_runner import performance_snapshot


def test_performance_snapshot_reports_calendar_win_rates() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-04-01"])
    returns = pd.Series([0.01, -0.005, -0.01, 0.02], index=index)
    snapshot, periods = performance_snapshot(returns)
    assert snapshot["win_days"] == 0.5
    assert snapshot["win_months"] == 2 / 3
    assert snapshot["win_quarters"] == 0.5
    assert set(periods["period_type"]) == {"month", "quarter", "year"}
