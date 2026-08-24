from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from systematic_research.market_structure import (
    build_market_structure_features,
    build_tape_pace_features,
)


def _sessions(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    close = 100.0 * np.cumprod(1.0 + 0.0002 + index * 0.000001)
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "symbol": "NQU6",
            "trading_date": dates,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 10_000.0 + index * 10.0,
        }
    )


def test_market_structure_is_prefix_invariant() -> None:
    short = build_market_structure_features(_sessions(130))
    long_prefix = build_market_structure_features(_sessions(150)).iloc[:130].reset_index(drop=True)
    assert_frame_equal(short, long_prefix)


def test_percentiles_and_quartiles_have_valid_bounds() -> None:
    features = build_market_structure_features(_sessions(150)).iloc[-1]
    assert 0.0 <= features["range_percentile_120s"] <= 1.0
    assert features["range_quartile_60s"] in {1.0, 2.0, 3.0, 4.0}
    assert 0.0 <= features["trend_r_squared_120s"] <= 1.0


def test_tape_pace_uses_minute_observations() -> None:
    minute = pd.DataFrame(
        {
            "symbol": "NQU6",
            "trading_date": pd.to_datetime(["2024-01-02"] * 4),
            "return_1m": [np.nan, -0.01, 0.0, 0.02],
            "volume": [10.0, 20.0, 30.0, 40.0],
        }
    )
    result = build_tape_pace_features(minute).iloc[0]
    assert result["volume_per_observed_minute"] == 25.0
    assert result["zero_return_fraction"] == 1.0 / 3.0
    assert result["return_sign_imbalance"] == 0.0
