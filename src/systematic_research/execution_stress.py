"""Causal minute-open entry and frozen limit/timeout execution scenarios."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar, cast

import numpy as np
from numba import njit  # type: ignore[import-untyped]
from numpy.typing import NDArray

P = ParamSpec("P")
T = TypeVar("T")


def _compile_kernel[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    """Preserve the public Python signature across Numba's untyped dispatcher."""
    return cast(Callable[P, T], njit(cache=True)(function))


@_compile_kernel
def minute_outcomes(
    minute_ns: NDArray[np.int64],
    ohlc: NDArray[np.float64],
    entry_ns: NDArray[np.int64],
    deadline_ns: NDArray[np.int64],
    target_prices: NDArray[np.float64],
    delay: int,
    cost: float,
    entry_slippage: float,
    timeout_slippage: float,
    penetration: float,
) -> NDArray[np.float64]:
    """Rows: exit_ns,net,hit,status,entry,exit,MAE,MFE,hold,low_mark,close_mark.

    Status 1 resolved, 0 missing selected path, 2 delayed open already at target,
    3 delay consumes the window. Unknown outcomes occupy the deadline. A limit
    requires penetration but fills at the original target. Market exits incur
    timeout slippage. Entire exit-minute low is a conservative risk bound.
    """
    n = len(entry_ns)
    result = np.full((n, 11), np.nan)
    for row in range(n):
        decision = entry_ns[row] + delay * 60_000_000_000
        deadline = deadline_ns[row]
        result[row, 0] = deadline
        result[row, 3] = 0
        if decision >= deadline:
            result[row, 3] = 3
            continue
        start = np.searchsorted(minute_ns, decision)
        end = np.searchsorted(minute_ns, deadline)
        if start >= len(minute_ns) or minute_ns[start] != decision or end <= start:
            continue
        opened = ohlc[start, 0] + entry_slippage
        target = target_prices[row]
        result[row, 4] = opened
        if opened >= target:
            result[row, 3] = 2
            continue
        lowest = opened
        highest = opened
        peak_close = -cost / 2
        peak_low = peak_close
        dd_close = 0.0
        dd_low = 0.0
        bad = False
        for pos in range(start, end):
            if minute_ns[pos] != decision + (pos - start) * 60_000_000_000:
                bad = True
                break
            lowest = min(lowest, ohlc[pos, 2])
            highest = max(highest, ohlc[pos, 1])
            hit = ohlc[pos, 1] >= target + penetration
            last = pos == end - 1
            if hit or last:
                exit_ns = minute_ns[pos] + 60_000_000_000
                if not hit and exit_ns != deadline:
                    bad = True
                    break
                closed = target if hit else ohlc[pos, 3] - timeout_slippage
                pnl = (closed - opened) * 20 - cost
                close_mark = pnl
            else:
                close_mark = (ohlc[pos, 3] - opened) * 20 - cost / 2
            low_mark = (ohlc[pos, 2] - opened) * 20 - cost / 2
            peak_close = max(peak_close, close_mark)
            dd_close = max(dd_close, peak_close - close_mark)
            # No high-then-low order is invented inside an OHLC minute.
            dd_low = max(dd_low, peak_low - low_mark)
            peak_low = max(peak_low, close_mark)
            if hit or last:
                result[row, 0] = exit_ns
                result[row, 1] = pnl
                result[row, 2] = 1.0 if hit else 0.0
                result[row, 3] = 1
                result[row, 5] = closed
                result[row, 6] = opened - lowest
                result[row, 7] = highest - opened
                result[row, 8] = pos - start + 1
                result[row, 9] = dd_low
                result[row, 10] = dd_close
                break
        if bad:
            result[row, 0] = deadline
            result[row, 1:3] = np.nan
            result[row, 3] = 0
    return result


@_compile_kernel
def schedule_events(
    signals: NDArray[np.bool_],
    entries: NDArray[np.int64],
    outcomes: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Select chronologically; missing outcomes cannot retroactively cancel entries."""
    chosen = np.empty(len(entries), dtype=np.int64)
    count = 0
    busy = np.iinfo(np.int64).min
    for i in range(len(entries)):
        if not signals[i] or entries[i] < busy:
            continue
        status = outcomes[i, 3]
        if status == 2 or status == 3:
            continue
        chosen[count] = i
        count += 1
        busy = int(outcomes[i, 0])
    return chosen[:count]


@_compile_kernel
def marked_drawdowns(
    minute_ns: NDArray[np.int64],
    ohlc: NDArray[np.float64],
    entries: NDArray[np.int64],
    outcomes: NDArray[np.float64],
    chosen: NDArray[np.int64],
    delay: int,
    cost: float,
) -> tuple[float, float]:
    """Minute-close DD and conservative minute-low bound across selected positions."""
    realized = 0.0
    peak = 0.0
    dd_close = 0.0
    dd_low = 0.0
    for i in chosen:
        if outcomes[i, 3] != 1:
            return np.nan, np.nan
        start = np.searchsorted(minute_ns, entries[i] + delay * 60_000_000_000)
        end = np.searchsorted(minute_ns, int(outcomes[i, 0]))
        opened = outcomes[i, 4]
        for q in range(start, end):
            low_mark = realized + (ohlc[q, 2] - opened) * 20 - cost / 2
            if q == end - 1:
                low_mark = min(low_mark, realized + outcomes[i, 1])
                close_mark = realized + outcomes[i, 1]
            else:
                close_mark = realized + (ohlc[q, 3] - opened) * 20 - cost / 2
            dd_low = max(dd_low, peak - low_mark)
            peak = max(peak, close_mark)
            dd_close = max(dd_close, peak - close_mark)
        realized += outcomes[i, 1]
    return dd_close, dd_low
