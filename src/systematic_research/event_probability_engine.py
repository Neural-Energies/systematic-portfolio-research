"""Extensible development-only event probability research engine.

Add an event by writing an EventBuilder that returns event timestamps and a
direction (+1 continuation/up, -1 continuation/down). The engine supplies
forward returns, hit probabilities, excursions, and split-sample stability.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EventDefinition:
    name: str
    direction: int
    mask: pd.Series


EventBuilder = Callable[[pd.DataFrame], list[EventDefinition]]
HORIZONS = {"15m": 1, "1h": 4, "4h": 16, "1d": 96}


def resample_15m(minute_bars: pd.DataFrame) -> pd.DataFrame:
    frame = minute_bars.set_index("timestamp_utc").sort_index()
    bars = (
        frame.resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trading_date=("trading_date", "last"),
        )
        .dropna(subset=["close"])
        .reset_index()
    )
    bars["return"] = bars["close"].pct_change(fill_method=None)
    bars["session_vwap"] = (
        bars.groupby("trading_date", observed=True)
        .apply(
            lambda x: (x["close"] * x["volume"]).cumsum() / x["volume"].cumsum(),
            include_groups=False,
        )
        .reset_index(level=0, drop=True)
    )
    bars["range_atr"] = (bars["high"] - bars["low"]).rolling(20, min_periods=10).mean()
    return bars


def rolling_breakouts(bars: pd.DataFrame) -> list[EventDefinition]:
    prior_high = bars["high"].rolling(20, min_periods=20).max().shift(1)
    prior_low = bars["low"].rolling(20, min_periods=20).min().shift(1)
    return [
        EventDefinition("20_bar_high_break", 1, bars["close"].gt(prior_high)),
        EventDefinition("20_bar_low_break", -1, bars["close"].lt(prior_low)),
    ]


def vwap_stretches(bars: pd.DataFrame) -> list[EventDefinition]:
    distance = (bars["close"] - bars["session_vwap"]) / bars["range_atr"]
    return [
        EventDefinition("vwap_stretch_up_1atr", -1, distance.ge(1)),
        EventDefinition("vwap_stretch_down_1atr", 1, distance.le(-1)),
    ]


EVENT_BUILDERS: dict[str, EventBuilder] = {
    "rolling_breakouts": rolling_breakouts,
    "vwap_stretches": vwap_stretches,
}


def evaluate_events(bars: pd.DataFrame, builders: list[EventBuilder]) -> pd.DataFrame:
    rows = []
    midpoint = bars["timestamp_utc"].median()
    for definition in [event for builder in builders for event in builder(bars)]:
        for label, steps in HORIZONS.items():
            future = bars["close"].shift(-steps) / bars["close"] - 1
            favorable = definition.direction * future
            for split, mask in {
                "all": definition.mask,
                "first_half": definition.mask & bars["timestamp_utc"].le(midpoint),
                "second_half": definition.mask & bars["timestamp_utc"].gt(midpoint),
            }.items():
                values = favorable[mask].dropna()
                if len(values) < 30:
                    continue
                rows.append(
                    {
                        "event": definition.name,
                        "direction": definition.direction,
                        "horizon": label,
                        "split": split,
                        "events": len(values),
                        "hit_probability": float((values > 0).mean()),
                        "mean_forward_return": float(values.mean()),
                        "median_forward_return": float(values.median()),
                        "p05": float(values.quantile(0.05)),
                        "p95": float(values.quantile(0.95)),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run extensible event probability research on development futures bars."
    )
    parser.add_argument("--symbols", nargs="+", default=["ES", "NQ", "CL", "GC", "6E"])
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/event_probability_results.csv")
    )
    args = parser.parse_args()
    root = Path("data/processed/databento_research/development_minute_returns")
    outputs = []
    for symbol in args.symbols:
        path = root / f"symbol={symbol}" / "returns.parquet"
        if not path.exists():
            continue
        bars = resample_15m(pd.read_parquet(path))
        result = evaluate_events(bars, list(EVENT_BUILDERS.values()))
        result.insert(0, "symbol", symbol)
        outputs.append(result)
    combined = pd.concat(outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
