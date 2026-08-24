from __future__ import annotations

from pathlib import Path

import pandas as pd

MINUTE_COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def read_minute_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, names=MINUTE_COLUMNS, header=None)
    stamp = pd.to_datetime(frame.pop("date") + " " + frame.pop("time"), utc=True)
    frame.index = pd.DatetimeIndex(stamp, name="timestamp_utc")
    frame = frame.sort_index()
    if not frame.index.is_unique:
        raise ValueError("minute timestamps must be unique")
    if (frame[["open", "high", "low", "close"]].isna().any()).any():
        raise ValueError("OHLC data contains missing values")
    if (
        (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
        | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
    ).any():
        raise ValueError("invalid OHLC relationship")
    return frame


def to_hourly(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate UTC minute data into bars labelled by bar-open time."""
    hourly = minute.resample("1h", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_count=("close", "count"),
    )
    return hourly.dropna(subset=["open", "high", "low", "close"])
