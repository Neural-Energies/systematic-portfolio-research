"""Explicit OHLCV profile approximations and minute-resolution entry path diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def wilder_mean(series: pd.Series, periods: int) -> pd.Series:
    """Seed with a full arithmetic average, then Wilder's recursive smoothing."""
    if periods < 1:
        raise ValueError("Invalid smoothing length")
    values = series.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    seed: list[float] = []
    previous = np.nan
    for i, value in enumerate(values):
        if not np.isfinite(value):
            seed = []
            previous = np.nan
            continue
        if not np.isfinite(previous):
            seed.append(float(value))
            if len(seed) < periods:
                continue
            previous = float(np.mean(seed))
        else:
            previous = (previous * (periods - 1) + value) / periods
        output[i] = previous
    return pd.Series(output, index=series.index, name=series.name)


def profile_histogram(bars: pd.DataFrame, width: float, allocation: str) -> dict[int, float]:
    if width <= 0 or allocation not in ("uniform", "close"):
        raise ValueError("Invalid profile approximation")
    hist: dict[int, float] = {}
    for low, high, close, volume in bars[["low", "high", "close", "volume"]].to_numpy(dtype=float):
        if volume < 0 or high < low:
            raise ValueError("Invalid price/volume")
        a, b = (
            (int(np.floor(low / width)), int(np.floor(high / width)))
            if allocation == "uniform"
            else (int(np.floor(close / width)), int(np.floor(close / width)))
        )
        mass = volume / (b - a + 1)
        for key in range(a, b + 1):
            hist[key] = hist.get(key, 0) + mass
    return hist


def profile_levels(
    hist: dict[int, float], width: float, fraction: float = 0.70
) -> tuple[float, float, float]:
    if width <= 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid value area")
    nonzero = {k: v for k, v in hist.items() if v > 0}
    if not nonzero:
        return np.nan, np.nan, np.nan
    keys = np.arange(min(nonzero), max(nonzero) + 1)
    mass = np.array([nonzero.get(int(k), 0) for k in keys])
    peak = int(np.argmax(mass))
    left = right = peak
    acc = mass[peak]
    target = mass.sum() * fraction
    while acc < target:
        below = mass[left - 1] if left > 0 else -1
        above = mass[right + 1] if right + 1 < len(mass) else -1
        if above >= below:
            right += 1
            acc += mass[right]
        else:
            left -= 1
            acc += mass[left]
    return (
        float((keys[peak] + 0.5) * width),
        float(keys[left] * width),
        float((keys[right] + 1) * width),
    )


def entry_path_diagnostics(
    minutes: pd.DataFrame,
    entries: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    distance: NDArray[np.float64],
) -> pd.DataFrame:
    """Observed barriers describe paths; this does not place targets, stops, or orders.

    Same-minute touches have unknown ordering and are conservatively not counted as
    favorable-first successes. A gap before the first touch makes its ordering unknown;
    a gap after a
    known touch does not erase it. An observed favorable touch remains known even
    if later coverage is incomplete.
    """
    if (
        len(entries) != len(ends)
        or len(entries) != len(distance)
        or np.any(distance <= 0)
        or not np.isfinite(distance).all()
    ):
        raise ValueError("Invalid entry path inputs")
    clock = pd.DatetimeIndex(minutes.index)
    stamps = clock.to_numpy(dtype="datetime64[ns]").astype("int64")
    opened = minutes.open.to_numpy(dtype=float)
    high = minutes.high.to_numpy(dtype=float)
    low = minutes.low.to_numpy(dtype=float)
    starts = clock.get_indexer(entries)
    rows = []
    for i, a in enumerate(starts):
        expected = int((ends[i] - entries[i]).total_seconds() / 60)
        b = int(clock.searchsorted(ends[i], side="left")) - 1
        if a < 0 or b < a or expected <= 0:
            rows.append((False, False, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        gaps = np.flatnonzero(np.diff(stamps[a : b + 1]) != 60 * 10**9)
        prefix = int(gaps[0]) + 1 if len(gaps) else b - a + 1
        complete = prefix == expected and b - a + 1 == expected
        up_all = np.flatnonzero(high[a : b + 1] >= opened[a] + distance[i])
        upside = np.flatnonzero(high[a : a + prefix] >= opened[a] + distance[i])
        downside = np.flatnonzero(low[a : a + prefix] <= opened[a] - distance[i])
        u = int(upside[0]) if len(upside) else prefix
        d = int(downside[0]) if len(downside) else prefix
        path_known = complete or min(u, d) < prefix
        reach_known = complete or len(up_all) > 0
        ambiguous = u == d and u < prefix
        adverse_before_up = opened[a] - low[a : a + u + 1].min() if u < prefix else np.nan
        rows.append(
            (
                path_known,
                reach_known,
                float(len(up_all) > 0) if reach_known else np.nan,
                float(u < d) if path_known else np.nan,
                float(d < u) if path_known else np.nan,
                float(ambiguous) if path_known else np.nan,
                float(u == prefix and d == prefix) if complete else np.nan,
                adverse_before_up,
            )
        )
    return pd.DataFrame(
        rows,
        index=entries,
        columns=[
            "path_scorable",
            "reach_scorable",
            "reached_up",
            "up_first",
            "down_first",
            "ambiguous",
            "neither",
            "adverse_before_up_points",
        ],
    )
