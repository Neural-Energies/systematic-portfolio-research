from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from systematic_research.execution_stress import _compile_kernel


@_compile_kernel
def bracket_outcomes(
    minute_ns: NDArray[np.int64],
    ohlc: NDArray[np.float64],
    entry_ns: NDArray[np.int64],
    deadline_ns: NDArray[np.int64],
    target_prices: NDArray[np.float64],
    stop_distances: NDArray[np.float64],
    delay: int,
    cost: float,
    entry_slippage: float,
    timeout_slippage: float,
    penetration: float,
) -> NDArray[np.float64]:
    """Rows: exit_ns,net,hit,status,entry,exit,MAE,MFE,hold,low_mark,close_mark.

    Same-minute stop/target ambiguity resolves stop first. Gap stops fill at worse open.
    Column11: target1, stop-1, time0. Status 1 resolved, 0 missing selected path,
    2 delayed open already at target,
    3 delay consumes the window. Unknown outcomes occupy the deadline. A limit
    requires penetration but fills at the original target. Market exits incur
    timeout slippage. Entire exit-minute low is a conservative risk bound.
    """
    n = len(entry_ns)
    result = np.full((n, 12), np.nan)
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
        stop_price = opened - stop_distances[row]
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
            # An opening gap has known ordering; only intraminute dual touches
            # need the conservative stop-first assumption.
            target_at_open = ohlc[pos, 0] >= target + penetration
            stopped = ohlc[pos, 2] <= stop_price and not target_at_open
            hit = hit and not stopped
            last = pos == end - 1
            if hit or stopped or last:
                exit_ns = minute_ns[pos] + 60_000_000_000
                if not hit and not stopped and exit_ns != deadline:
                    bad = True
                    break
                closed = target if hit else ohlc[pos, 3] - timeout_slippage
                if stopped:
                    closed = min(stop_price, ohlc[pos, 0]) - timeout_slippage
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
            if hit or stopped or last:
                result[row, 0] = exit_ns
                result[row, 1] = pnl
                result[row, 2] = 1.0 if hit else 0.0
                result[row, 3] = 1
                result[row, 11] = -1.0 if stopped else (1.0 if hit else 0.0)
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
