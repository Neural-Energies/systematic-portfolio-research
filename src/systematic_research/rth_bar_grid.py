"""Cash-open anchored bars with explicit minute timestamp semantics."""

from __future__ import annotations

from typing import cast

import pandas as pd


def minute_start_clock(clock: pd.DatetimeIndex, *, labels_at_end: bool) -> pd.DatetimeIndex:
    """Normalize aware source minute labels to the time of the recorded open."""
    if clock.tz is None or not clock.is_unique or not clock.is_monotonic_increasing:
        raise ValueError("An aware, unique, chronological source clock is required")
    return clock - pd.Timedelta(minutes=1) if labels_at_end else clock.copy()


def rth_bar_grid(schedule: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Half-open holding windows, anchored to cash open and capped at cash close."""
    if minutes not in (5, 15, 60, 240):
        raise ValueError("Supported research durations: 5, 15, 60, 240 minutes")
    rows = []
    for session in schedule.itertuples():
        opened = cast(pd.Timestamp, session.open)
        closed = cast(pd.Timestamp, session.close)
        for start in pd.date_range(opened, closed, freq=f"{minutes}min", inclusive="left"):
            rows.append((start, min(start + pd.Timedelta(minutes=minutes), closed), opened))
    return pd.DataFrame(rows, columns=["entry_time", "planned_exit", "anchor"]).set_index(
        "entry_time"
    )


def completed_rth_bars(minutes: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    """Only fully observed bars; availability is the actual window end, never its open."""
    rows = []
    for key, row in grid.iterrows():
        start = cast(pd.Timestamp, key)
        path = minutes.iloc[
            minutes.index.searchsorted(start) : minutes.index.searchsorted(row.planned_exit)
        ]
        expected = pd.date_range(start, row.planned_exit, freq="min", inclusive="left")
        if not path.index.equals(expected):
            continue
        rows.append(
            {
                "bar_open": start,
                "open": path.open.iloc[0],
                "high": path.high.max(),
                "low": path.low.min(),
                "close": path.close.iloc[-1],
                "volume": path.volume.sum(),
                "available_at": row.planned_exit,
                "session": row.anchor,
            }
        )
    columns = ["bar_open", "open", "high", "low", "close", "volume", "available_at", "session"]
    return pd.DataFrame(rows, columns=columns).set_index("bar_open").sort_index()
