from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.intraday_mean_reversion import (
    build_intraday_features,
    hysteresis_positions,
)


def test_running_midpoint_never_uses_future_high_or_low() -> None:
    frame = pd.DataFrame(
        {
            "symbol": "A",
            "trading_date": [pd.Timestamp("2024-01-02")] * 3,
            "timestamp_utc": pd.date_range("2024-01-02 23:00", periods=3, freq="min", tz="UTC"),
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 102.0, 110.0],
            "low": [99.0, 98.0, 90.0],
            "close": [100.0, 101.0, 105.0],
            "volume": [10.0, 20.0, 30.0],
            "return_1m": [np.nan, 0.01, 0.04],
        }
    )
    result = build_intraday_features(frame)
    assert float(cast(Any, result.loc[0, "running_midpoint"])) == 100.0
    assert float(cast(Any, result.loc[1, "running_midpoint"])) == 100.0


def test_hysteresis_respects_entry_exit_and_holding_cap() -> None:
    score = pd.Series([0.0, 0.5, 0.4, 0.05, -0.5])
    dates = pd.Series([pd.Timestamp("2024-01-02")] * 5)
    minute = pd.Series([100, 101, 102, 103, 104])
    result = hysteresis_positions(score, dates, minute, entry=0.35, exit=0.10, maximum_minutes=10)
    assert result.tolist() == [0.0, 1.0, 1.0, 0.0, -1.0]
