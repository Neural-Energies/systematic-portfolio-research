"""Single-position long execution against frozen upward-excursion references."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_reference(
    minutes: pd.DataFrame,
    opportunities: pd.DataFrame,
    signals: pd.Series,
    cost: float = 25.0,
    latency_minutes: int = 0,
    penetration: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One NQ contract; frozen target; close at deadline; never select on future coverage.

    A target exit frees the position at the next minute boundary. Missing minutes
    before exit make the selected trade unresolved and occupy the whole planned
    window. Equity on unresolved paths is unknown rather than an invented zero.
    Exit-minute lows are excluded from exact MAE because OHLC cannot reveal order.
    Conservative MAE including that entire minute is returned separately.
    """
    if not minutes.index.is_unique or not minutes.index.is_monotonic_increasing:
        raise ValueError("Minute data must be unique and chronological")
    if not opportunities.index.is_unique or not opportunities.index.is_monotonic_increasing:
        raise ValueError("Opportunities must be unique and chronological")
    if cost < 0 or latency_minutes < 0 or penetration < 0:
        raise ValueError("Invalid execution settings")
    if opportunities.target_points.le(0).any():
        raise ValueError("Target distance must be positive")
    clock = pd.DatetimeIndex(minutes.index)
    busy_until = pd.Timestamp.min.tz_localize("UTC")
    trades: list[dict[str, object]] = []
    marks: list[dict[str, object]] = []
    realized = 0.0
    active = signals.reindex(opportunities.index, fill_value=False).fillna(False)
    for decision, item in opportunities.loc[active].iterrows():
        if not isinstance(decision, pd.Timestamp):
            raise ValueError("Opportunity index must contain timestamps")
        entry = decision + pd.Timedelta(minutes=latency_minutes)
        deadline = pd.Timestamp(item.planned_exit)
        if decision < busy_until or entry >= deadline:
            continue
        target = float(item.entry_open + item.target_points)
        start = clock.searchsorted(entry)
        end = clock.searchsorted(deadline)
        path = minutes.iloc[start:end]
        base: dict[str, object] = {
            "decision": decision,
            "entry_time": entry,
            "planned_exit": deadline,
            "anchor": item.anchor,
            "split": item.split,
            "target_price": target,
            "target_points": float(item.target_points),
        }
        if path.empty or path.index[0] != entry:
            trades.append({**base, "status": "unresolved", "exit_time": deadline})
            busy_until = deadline
            continue
        opened = float(path.open.iloc[0])
        if opened >= target:
            # At delayed entry, this is observable already; no profitable fill invented.
            trades.append({**base, "entry_price": opened, "status": "already_at_target"})
            continue
        touching = np.flatnonzero(path.high.to_numpy() >= target + penetration)
        hit = bool(len(touching))
        stop = int(touching[0]) if hit else len(path) - 1
        held = path.iloc[: stop + 1]
        expected_end = held.index[-1] + pd.Timedelta(minutes=1) if hit else deadline
        complete = len(held) == int((expected_end - entry).total_seconds() / 60)
        complete &= held.index[-1] == expected_end - pd.Timedelta(minutes=1)
        complete &= bool(np.isfinite(held[["open", "high", "low", "close"]]).all().all())
        if not complete:
            trades.append(
                {**base, "entry_price": opened, "status": "unresolved", "exit_time": deadline}
            )
            busy_until = deadline
            continue
        exit_price = target if hit else float(held.close.iloc[-1])
        pnl = (exit_price - opened) * 20 - cost
        previous = held.iloc[:-1] if hit else held
        mae_lower = max(0.0, opened - float(previous.low.min())) if len(previous) else 0.0
        mae_upper = max(0.0, opened - float(held.low.min()))
        for instant, quote in held.iloc[:-1].iterrows():
            if not isinstance(instant, pd.Timestamp):
                raise ValueError("Minute index must contain timestamps")
            marks.append(
                {
                    "time": instant + pd.Timedelta(minutes=1),
                    "equity": realized + (quote.close - opened) * 20 - cost / 2,
                    "intraminute_low_equity": realized + (quote.low - opened) * 20 - cost / 2,
                }
            )
        marks.append(
            {
                "time": expected_end,
                "equity": realized + pnl,
                "intraminute_low_equity": min(
                    realized + pnl, realized + (float(held.low.iloc[-1]) - opened) * 20 - cost
                ),
            }
        )
        realized += pnl
        trades.append(
            {
                **base,
                "entry_price": opened,
                "exit_time": expected_end,
                "exit_price": exit_price,
                "status": "target" if hit else "timeout",
                "net_dollars": pnl,
                "gross_points": (exit_price - opened),
                "mae_lower_points": mae_lower,
                "mae_upper_points": mae_upper,
                "holding_minutes": len(held),
            }
        )
        busy_until = expected_end
    return pd.DataFrame(trades), pd.DataFrame(marks)
