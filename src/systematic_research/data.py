"""Market-data contracts that prevent common silent research errors."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "symbol", "close"}


def validate_market_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize long-form market data.

    The returned frame is sorted by symbol and UTC timestamp. Duplicate observations,
    non-positive prices, missing values, and timezone-naive timestamps are rejected.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = frame.copy()
    timestamps = pd.to_datetime(result["timestamp"], errors="raise")
    if timestamps.dt.tz is None:
        raise ValueError("Timestamps must be timezone-aware")
    result["timestamp"] = timestamps.dt.tz_convert("UTC")

    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required market-data fields cannot be null")
    if (result["close"] <= 0).any():
        raise ValueError("Prices must be strictly positive")
    if result.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("Duplicate symbol/timestamp observations detected")

    return result.sort_values(["symbol", "timestamp"], ignore_index=True)
