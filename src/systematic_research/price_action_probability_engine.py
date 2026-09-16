"""Session-reference price-action event research.

The module tests observable market structure rather than technical indicators.
Definitions are evaluated on a frozen development sample in chronological halves.
The reversion diagnostics follow the statistical checks described by Ernest Chan:
stationarity, Hurst exponent, and AR(1) half-life are evidence about a *distance
series*, not evidence that a tradable strategy already exists.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

NEW_YORK = "America/New_York"


@dataclass(frozen=True)
class PriceActionEvent:
    """One objectively timestamped event and its directional reference level."""

    name: str
    direction: int
    timestamp_utc: pd.Timestamp
    target: float
    session: pd.Timestamp


def rth_session(timestamp_utc: pd.Series) -> pd.Series:
    """Return the CME-style session date; 18:00 ET belongs to the next session."""
    local = timestamp_utc.dt.tz_convert(NEW_YORK)
    return (
        local.dt.normalize() + pd.to_timedelta((local.dt.hour >= 18).astype(int), unit="D")
    ).dt.date


def add_session_context(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Attach expanding Globex levels available at each timestamp without look-ahead."""
    bars = minute_bars.copy().sort_values("timestamp_utc").reset_index(drop=True)
    bars["session"] = rth_session(bars["timestamp_utc"])
    local = bars["timestamp_utc"].dt.tz_convert(NEW_YORK)
    minute = local.dt.hour * 60 + local.dt.minute
    bars["is_globex"] = (minute >= 18 * 60) | (minute < 9 * 60 + 30)
    bars["is_rth"] = (minute >= 9 * 60 + 30) & (minute <= 16 * 60)
    globex = bars[bars["is_globex"]].copy()
    grouped = globex.groupby("session", observed=True)
    summary = grouped.agg(
        globex_high=("high", "max"),
        globex_low=("low", "min"),
        globex_volume=("volume", "sum"),
    )
    summary["globex_vwap"] = grouped.apply(
        lambda x: float(np.average(x["close"], weights=x["volume"])), include_groups=False
    )
    return bars.join(summary, on="session")


def first_globex_breaks(context: pd.DataFrame) -> list[PriceActionEvent]:
    """First RTH breach of either completed Globex extreme, one event per direction/session."""
    events: list[PriceActionEvent] = []
    for session, day in context[context["is_rth"]].groupby("session", observed=True):
        if day.empty or day["globex_high"].isna().all():
            continue
        high_level = float(day["globex_high"].iloc[0])
        low_level = float(day["globex_low"].iloc[0])
        above = day[day["high"] > high_level]
        below = day[day["low"] < low_level]
        if not above.empty:
            row = above.iloc[0]
            events.append(
                PriceActionEvent(
                    "first_globex_high_break_to_vwap",
                    1,
                    row.timestamp_utc,
                    float(day["globex_vwap"].iloc[0]),
                    session,
                )
            )
        if not below.empty:
            row = below.iloc[0]
            events.append(
                PriceActionEvent(
                    "first_globex_low_break_to_vwap",
                    -1,
                    row.timestamp_utc,
                    float(day["globex_vwap"].iloc[0]),
                    session,
                )
            )
    return events


def failed_globex_breaks(
    context: pd.DataFrame, reentry_minutes: int = 30
) -> list[PriceActionEvent]:
    """Break an extreme, then re-enter its range within a fixed intraday window."""
    events: list[PriceActionEvent] = []
    rth_days = {
        session: day
        for session, day in context[context["is_rth"]].groupby("session", observed=True)
    }
    for event in first_globex_breaks(context):
        day = rth_days[event.session]
        after = day[day["timestamp_utc"] > event.timestamp_utc].head(reentry_minutes)
        break_level = float(
            day["globex_high"].iloc[0] if event.direction == 1 else day["globex_low"].iloc[0]
        )
        if event.direction == 1:
            reentry = after[after["close"] <= break_level]
        else:
            reentry = after[after["close"] >= break_level]
        if not reentry.empty:
            row = reentry.iloc[0]
            events.append(
                PriceActionEvent(
                    "failed_globex_break",
                    -event.direction,
                    row.timestamp_utc,
                    float(day["globex_vwap"].iloc[0]),
                    event.session,
                )
            )
    return events


def evaluate_target_touches(context: pd.DataFrame, events: list[PriceActionEvent]) -> pd.DataFrame:
    """Measure whether and when a known price target is touched after the event."""
    rows: list[dict[str, object]] = []
    rth_days = {
        session: day
        for session, day in context[context["is_rth"]].groupby("session", observed=True)
    }
    for event in events:
        after = rth_days[event.session]
        after = after[after["timestamp_utc"] > event.timestamp_utc]
        hit = after[(after["low"] <= event.target) & (after["high"] >= event.target)]
        rows.append(
            {
                "event": event.name,
                "direction": event.direction,
                "timestamp_utc": event.timestamp_utc,
                "session": event.session,
                "target": event.target,
                "target_touched": not hit.empty,
                "minutes_to_target": (
                    (hit.iloc[0].timestamp_utc - event.timestamp_utc).total_seconds() / 60
                    if not hit.empty
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def chan_reversion_diagnostics(context: pd.DataFrame) -> pd.DataFrame:
    """Stationarity, Hurst, and half-life diagnostics for RTH price-to-Globex-VWAP distance."""
    rth = context.loc[context["is_rth"], ["timestamp_utc", "close", "globex_vwap"]].dropna()
    # The research horizon is 15 minutes or longer. Sampling at that cadence
    # avoids overweighting microstructure noise and keeps the ADF test tractable.
    rth = rth.set_index("timestamp_utc").resample("15min").last().dropna()
    distance = rth["close"] - rth["globex_vwap"]
    if len(distance) < 100:
        return pd.DataFrame()
    lagged = distance.shift(1).dropna()
    delta = distance.diff().dropna().loc[lagged.index]
    beta = float(np.linalg.lstsq(np.c_[np.ones(len(lagged)), lagged], delta, rcond=None)[0][1])
    half_life = float(-np.log(2) / beta) if beta < 0 else np.nan
    lags = np.arange(2, min(100, len(distance) // 4))
    tau = np.array([np.sqrt(np.std(distance.diff(int(lag)).dropna())) for lag in lags])
    hurst = float(np.polyfit(np.log(lags), np.log(tau), 1)[0] * 2) if len(lags) > 2 else np.nan
    return pd.DataFrame(
        [
            {
                "series": "15m_rth_close_minus_completed_globex_vwap",
                "observations": len(distance),
                "adf_pvalue": float(adfuller(distance, autolag="AIC")[1]),
                "hurst": hurst,
                "ar1_half_life_minutes": half_life,
            }
        ]
    )


def summarize_by_half(touches: pd.DataFrame) -> pd.DataFrame:
    """Chronological split only; the second half is never used to select definitions."""
    midpoint = touches["timestamp_utc"].median()
    touches = touches.assign(
        split=np.where(touches["timestamp_utc"] <= midpoint, "first_half", "second_half")
    )
    return (
        touches.groupby(["event", "direction", "split"], observed=True)
        .agg(
            events=("target_touched", "size"),
            target_touch_probability=("target_touched", "mean"),
            median_minutes_to_target=("minutes_to_target", "median"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run development-only session price-action research."
    )
    parser.add_argument("--symbols", nargs="+", default=["ES", "NQ", "CL", "GC", "6E"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/price_action"))
    args = parser.parse_args()
    root = Path("data/processed/databento_research/development_minute_returns")
    summaries: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for symbol in args.symbols:
        path = root / f"symbol={symbol}" / "returns.parquet"
        if not path.exists():
            continue
        context = add_session_context(pd.read_parquet(path))
        events = first_globex_breaks(context) + failed_globex_breaks(context)
        touches = evaluate_target_touches(context, events)
        if not touches.empty:
            summary = summarize_by_half(touches)
            summary.insert(0, "symbol", symbol)
            summaries.append(summary)
        diagnostic = chan_reversion_diagnostics(context)
        if not diagnostic.empty:
            diagnostic.insert(0, "symbol", symbol)
            diagnostics.append(diagnostic)
    pd.concat(summaries, ignore_index=True).to_csv(
        args.output_dir / "target_touch_summary.csv", index=False
    )
    pd.concat(diagnostics, ignore_index=True).to_csv(
        args.output_dir / "chan_reversion_diagnostics.csv", index=False
    )


if __name__ == "__main__":
    main()
