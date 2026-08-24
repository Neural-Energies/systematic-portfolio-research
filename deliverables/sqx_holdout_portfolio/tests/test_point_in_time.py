from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_futures.signals import generate_candidate_orders


def bars(size: int = 400) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=size, freq="h", tz="UTC")
    base = 100 + np.sin(np.arange(size) / 7) + np.arange(size) * 0.01
    return pd.DataFrame(
        {"open": base, "high": base + 0.5, "low": base - 0.5, "close": base + 0.1, "volume": 100},
        index=index,
    )


@pytest.mark.parametrize("symbol", ["6E", "6J", "CL", "ES", "GC", "HG", "NG"])
def test_appending_future_bars_does_not_change_old_orders(symbol: str) -> None:
    full = bars()
    cutoff = 300
    old = generate_candidate_orders(symbol, full.iloc[:cutoff])
    new = generate_candidate_orders(symbol, full)
    limit = pd.Timestamp(full.index[cutoff - 1].value + 3_600_000_000_000, tz="UTC")
    pd.testing.assert_frame_equal(
        old, new.loc[new["timestamp_utc"] <= limit].reset_index(drop=True)
    )


def test_decisions_follow_completed_bars() -> None:
    frame = bars()
    orders = generate_candidate_orders("ES", frame)
    assert orders["timestamp_utc"].min() > frame.index.min()
