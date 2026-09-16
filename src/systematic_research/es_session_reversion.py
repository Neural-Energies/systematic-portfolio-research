"""Independent ES session-reference mean-reversion event search."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from systematic_research.price_action_probability_engine import add_session_context


def scan(
    context: pd.DataFrame, thresholds: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
) -> pd.DataFrame:
    """Test first independent RTH extension from completed Globex VWAP per session/side."""
    rows: list[dict[str, object]] = []
    rth = context.loc[context["is_rth"]].copy()
    rth["globex_range"] = (rth["globex_high"] - rth["globex_low"]).replace(0.0, pd.NA)
    rth["vwap_distance_range"] = (rth["close"] - rth["globex_vwap"]) / rth["globex_range"]
    for threshold, direction in itertools.product(thresholds, (-1, 1)):
        events: list[dict[str, object]] = []
        for _session, day in rth.groupby("session", observed=True):
            trigger = (
                day[day["vwap_distance_range"] <= -threshold]
                if direction == 1
                else day[day["vwap_distance_range"] >= threshold]
            )
            if trigger.empty:
                continue
            entry = trigger.iloc[0]
            after = day[day["timestamp_utc"] > entry.timestamp_utc]
            target = float(entry.globex_vwap)
            hit = after[(after["low"] <= target) & (after["high"] >= target)]
            events.append(
                {
                    "timestamp_utc": entry.timestamp_utc,
                    "target_hit": not hit.empty,
                    "minutes_to_target": (
                        (hit.iloc[0].timestamp_utc - entry.timestamp_utc).total_seconds() / 60
                        if not hit.empty
                        else pd.NA
                    ),
                }
            )
        event_frame = pd.DataFrame(events)
        if event_frame.empty:
            continue
        midpoint = event_frame["timestamp_utc"].median()
        for split, sample in (
            ("discovery", event_frame[event_frame["timestamp_utc"] <= midpoint]),
            ("confirmation", event_frame[event_frame["timestamp_utc"] > midpoint]),
        ):
            rows.append(
                {
                    "threshold_globex_ranges": threshold,
                    "side": "long_from_below" if direction == 1 else "short_from_above",
                    "split": split,
                    "independent_sessions": len(sample),
                    "target_touch_probability": sample["target_hit"].mean(),
                    "median_minutes_to_target": sample.loc[
                        sample["target_hit"], "minutes_to_target"
                    ].median(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan ES Globex-VWAP reversion events on development data only."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/es_session_reversion/event_scan.csv")
    )
    args = parser.parse_args()
    bars = pd.read_parquet(
        "data/processed/databento_research/development_minute_returns/symbol=ES/returns.parquet"
    )
    result = scan(add_session_context(bars))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
