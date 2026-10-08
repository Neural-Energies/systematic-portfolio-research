"""Deterministic canonical rule search and accelerated causal entry diagnostics."""

from __future__ import annotations

import numpy as np
from numba import njit, prange  # type: ignore[import-untyped]
from numpy.typing import NDArray


def hit_tier(rate: float) -> str:
    """User-defined hit tiers; distinct from statistical validation status."""
    if rate == 1.0:
        return "tier_3_100"
    if rate >= 0.9:
        return "tier_2_90"
    if rate >= 0.7:
        return "tier_1_70"
    if rate > 0.53:
        return "saved_above_53"
    if rate >= 0.5:
        return "watch_50_to_53"
    return "failed"


@njit(cache=True)  # type: ignore[untyped-decorator]
def _popcount(value: np.uint64) -> int:
    value -= (value >> np.uint64(1)) & np.uint64(0x5555555555555555)
    value = (value & np.uint64(0x3333333333333333)) + (
        (value >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    value = (value + (value >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=True, parallel=True)  # type: ignore[untyped-decorator]
def screen_rules(
    atoms: NDArray[np.uint64],
    definitions: NDArray[np.int64],
    masks: NDArray[np.uint64],
    hit_masks: NDArray[np.uint64],
    groups: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Counts only frozen IS/validation stages; no confirmation mask supplied."""
    stride = hit_masks.shape[0] + 1
    output = np.zeros((len(definitions), 2 * stride + 1), dtype=np.int64)
    for i in prange(len(definitions)):
        a, b, c = definitions[i]
        if groups[a] == groups[b] or groups[a] == groups[c] or groups[b] == groups[c]:
            continue
        output[i, 2 * stride] = 1
        for word in range(atoms.shape[1]):
            signal = atoms[a, word] & atoms[b, word] & atoms[c, word]
            for stage in range(2):
                scoped = signal & masks[stage, word]
                output[i, stage * stride] += _popcount(scoped)
                for reference in range(hit_masks.shape[0]):
                    output[i, stage * stride + reference + 1] += _popcount(
                        scoped & hit_masks[reference, word]
                    )
    return output


@njit(cache=True)  # type: ignore[untyped-decorator]
def unrank_triples(ranks: NDArray[np.int64], count: int) -> NDArray[np.int64]:
    """One-to-one lexicographic unranking; avoids a giant random-definition set."""
    result = np.empty((len(ranks), 3), dtype=np.int64)
    total = count * (count - 1) * (count - 2) // 6
    for i in range(len(ranks)):
        rank = ranks[i]
        lo, hi = 0, count - 3
        while lo < hi:
            mid = (lo + hi + 1) // 2
            remaining = count - mid
            preceding = total - remaining * (remaining - 1) * (remaining - 2) // 6
            if preceding <= rank:
                lo = mid
            else:
                hi = mid - 1
        a = lo
        remaining = count - a
        rank -= total - remaining * (remaining - 1) * (remaining - 2) // 6
        b = a + 1
        while rank >= count - b - 1:
            rank -= count - b - 1
            b += 1
        result[i, 0], result[i, 1], result[i, 2] = a, b, b + 1 + rank
    return result


@njit(cache=True)  # type: ignore[untyped-decorator]
def execution_metrics(
    signal: NDArray[np.uint64],
    stage: NDArray[np.int64],
    starts: NDArray[np.int64],
    exits: NDArray[np.int64],
    pnl: NDArray[np.float64],
    mae: NDArray[np.float64],
    mfe: NDArray[np.float64],
    days: NDArray[np.int64],
    target_hit: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Greedy single-position execution; unresolved trades still occupy their window.

    Return per-stage counts, PNL, closed-trade drawdown, full-window excursions,
    and session coverage. These are not tick-resolved or minute-equity metrics.
    """
    output = np.zeros((2, 21), dtype=np.float64)
    busy = -1
    equity = np.zeros(2)
    peak = np.zeros(2)
    seen_days = np.zeros((2, int(days.max()) + 1), dtype=np.bool_)
    for i in range(len(starts)):
        if stage[i] < 0:
            continue
        if not (signal[i // 64] & (np.uint64(1) << np.uint64(i % 64))):
            continue
        if starts[i] < busy:
            continue
        s = stage[i]
        output[s, 0] += 1
        seen_days[s, days[i]] = True
        busy = exits[i]
        if not np.isfinite(pnl[i]):
            output[s, 1] += 1
            continue
        output[s, 2] += pnl[i]
        output[s, 11] += pnl[i] > 0
        output[s, 12] += pnl[i] * pnl[i]
        if output[s, 0] - output[s, 1] == 1:
            output[s, 13] = pnl[i]
            output[s, 14] = pnl[i]
        else:
            output[s, 13] = min(output[s, 13], pnl[i])
            output[s, 14] = max(output[s, 14], pnl[i])
        output[s, 19] += (exits[i] - starts[i]) / (60 * 10**9)
        if target_hit is not None:
            output[s, 20] += target_hit[i]
        equity[s] += pnl[i]
        peak[s] = max(peak[s], equity[s])
        output[s, 3] = max(output[s, 3], peak[s] - equity[s])
        output[s, 4] += max(0.0, pnl[i])
        output[s, 5] += min(0.0, pnl[i])
        if np.isfinite(mae[i]):
            output[s, 6] += mae[i]
            output[s, 9] += 1
            output[s, 15] += mae[i] * mae[i]
            output[s, 17] = max(output[s, 17], mae[i])
        if np.isfinite(mfe[i]):
            output[s, 7] += mfe[i]
            output[s, 10] += 1
            output[s, 16] += mfe[i] * mfe[i]
            output[s, 18] = max(output[s, 18], mfe[i])
    for s in range(2):
        output[s, 8] = seen_days[s].sum()
    return output
