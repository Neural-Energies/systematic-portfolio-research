"""Causal execution scheduling and complete-data outcomes for fixed long holds."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def select_nonoverlapping(
    signal: NDArray[np.bool_], entry_times: NDArray[np.int64], exit_times: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Decide entries without consulting future-price availability or profitability."""
    if len(signal) != len(entry_times) or len(signal) != len(exit_times):
        raise ValueError("Signal and schedule lengths differ")
    if np.any(np.diff(entry_times) <= 0) or np.any(exit_times <= entry_times):
        raise ValueError("Invalid chronological holding schedule")
    chosen: list[int] = []
    busy_until = np.iinfo(np.int64).min
    for position in np.flatnonzero(signal):
        if entry_times[position] >= busy_until:
            chosen.append(int(position))
            busy_until = int(exit_times[position])
    return np.asarray(chosen, dtype=np.int64)


def fixed_hold_outcomes(
    minutes: pd.DataFrame,
    entries: pd.DatetimeIndex,
    exits: pd.DatetimeIndex,
    multiplier: float = 20.0,
    round_trip_cost: float = 25.0,
) -> pd.DataFrame:
    """Buy minute open, sell final held minute close; coverage never controls entry choice."""
    if not minutes.index.is_unique or not minutes.index.is_monotonic_increasing:
        raise ValueError("Minutes must be unique and ordered")
    if len(entries) != len(exits):
        raise ValueError("Entry/exit lengths differ")
    duration = (
        exits.to_numpy(dtype="datetime64[ns]").astype("int64")
        - entries.to_numpy(dtype="datetime64[ns]").astype("int64")
    ) // (60 * 10**9)
    if np.any(duration <= 0) or np.any(duration % 5 != 0):
        raise ValueError("Fixed holds must be positive multiples of five minutes")
    quote_clock = pd.DatetimeIndex(minutes.index)
    keyed = minutes.assign(observed_at=quote_clock)
    atomic = keyed.groupby(quote_clock.floor("5min")).agg(
        high=("high", "max"),
        low=("low", "min"),
        count=("close", "size"),
        first=("observed_at", "min"),
        last=("observed_at", "max"),
    )
    complete = atomic["count"].eq(5)
    atomic_clock = pd.DatetimeIndex(atomic.index)
    complete &= atomic["first"].eq(pd.Series(atomic_clock, index=atomic.index))
    complete &= atomic["last"].eq(
        pd.Series(atomic_clock + pd.Timedelta(minutes=4), index=atomic.index)
    )
    atomic[["high", "low"]] = atomic[["high", "low"]].where(complete, np.nan, axis=0)
    grid = pd.date_range(atomic.index.min(), atomic.index.max(), freq="5min")
    atomic = atomic.reindex(grid)
    future_high = np.full(len(entries), np.nan)
    future_low = np.full(len(entries), np.nan)
    for raw_width in np.unique(duration // 5):
        width = int(raw_width)
        positions = duration // 5 == width
        high = atomic.high.iloc[::-1].rolling(width, min_periods=width).max().iloc[::-1]
        low = atomic.low.iloc[::-1].rolling(width, min_periods=width).min().iloc[::-1]
        future_high[positions] = high.reindex(entries[positions]).to_numpy()
        future_low[positions] = low.reindex(entries[positions]).to_numpy()
    opened = minutes.open.reindex(entries).to_numpy(dtype=float)
    closed = minutes.close.reindex(exits - pd.Timedelta(minutes=1)).to_numpy(dtype=float)
    delayed = minutes.open.reindex(entries + pd.Timedelta(minutes=1)).to_numpy(dtype=float)
    valid = np.isfinite(opened) & np.isfinite(closed) & np.isfinite(future_high)
    valid &= np.isfinite(future_low)
    result = pd.DataFrame(
        {
            "planned_exit": exits,
            "effective_minutes": duration,
            "entry_open": opened,
            "exit_close": closed,
            "scorable": valid,
        },
        index=entries,
    )
    result["gross_dollars"] = np.where(valid, (closed - opened) * multiplier, np.nan)
    result["net_dollars"] = result.gross_dollars - round_trip_cost
    result["stress_dollars"] = result.gross_dollars - 2 * round_trip_cost
    result["delayed_net_dollars"] = np.where(
        valid, (closed - delayed) * multiplier - round_trip_cost, np.nan
    )
    result["mfe_points"] = np.where(valid, future_high - opened, np.nan)
    result["mae_points"] = np.where(valid, opened - future_low, np.nan)
    return result
