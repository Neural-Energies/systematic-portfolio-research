"""Data-driven intraday state discovery using observable Markov states.

States are learned from lagged market features, never from forward returns. A
candidate must exceed the probability threshold independently in both halves.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

HORIZONS = {"15m": 1, "1h": 4, "4h": 16}


def bars(path: Path) -> pd.DataFrame:
    x = (
        pd.read_parquet(path)
        .set_index("timestamp_utc")
        .resample("15min")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trading_date=("trading_date", "last"),
        )
        .dropna()
        .reset_index()
    )
    x["r"] = x.close.pct_change(fill_method=None)
    x["r1"] = x.r.shift(1)
    x["r4"] = x.r.rolling(4).sum().shift(1)
    x["r16"] = x.r.rolling(16).sum().shift(1)
    x["vol"] = x.r.rolling(20).std().shift(1)
    x["range"] = (x.high - x.low).div(x.close).rolling(20).mean().shift(1)
    x["vwap"] = (
        x.groupby("trading_date", observed=True)
        .apply(lambda g: (g.close * g.volume).cumsum() / g.volume.cumsum(), include_groups=False)
        .reset_index(level=0, drop=True)
    )
    x["vwap_dist"] = (x.close - x.vwap).div(x.close).shift(1)
    x["volu"] = np.log1p(x.volume).shift(1)
    x["hour_sin"] = np.sin(2 * np.pi * x.timestamp_utc.dt.hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * x.timestamp_utc.dt.hour / 24)
    return x


def scan(x: pd.DataFrame, components: int) -> pd.DataFrame:
    cols = ["r1", "r4", "r16", "vol", "range", "vwap_dist", "volu", "hour_sin", "hour_cos"]
    x = x.dropna(subset=cols).copy()
    mid = x.timestamp_utc.median()
    train = x.timestamp_utc.le(mid)
    scaler = StandardScaler().fit(x.loc[train, cols])
    model = GaussianMixture(
        n_components=components, covariance_type="full", random_state=7, n_init=3
    ).fit(scaler.transform(x.loc[train, cols]))
    x["state"] = model.predict(scaler.transform(x[cols]))
    rows = []
    for h, steps in HORIZONS.items():
        target = np.sign(x.close.shift(-steps) / x.close - 1)
        first_branch: dict[int, dict[int, float]] = {1: {}, -1: {}}
        for state in range(components):
            for direction, label in [(1, "up"), (-1, "down")]:
                probs = []
                counts = []
                for _split, mask in [("first", train), ("second", ~train)]:
                    vals = target[mask & x.state.eq(state)].dropna()
                    counts.append(len(vals))
                    probs.append(float((vals.eq(direction)).mean()) if len(vals) else np.nan)
                if min(counts) >= 100 and min(probs) >= 0.66:
                    rows.append(
                        {
                            "state_count": components,
                            "state": state,
                            "horizon": h,
                            "direction": label,
                            "first_probability": probs[0],
                            "second_probability": probs[1],
                            "first_events": counts[0],
                            "second_events": counts[1],
                            "min_probability": min(probs),
                        }
                    )
                if counts[0] >= 100 and pd.notna(probs[0]):
                    first_branch[direction][state] = probs[0]
        # OR discovery: form unions only from first-half branches that each
        # have at least a 55% same-direction probability. The second half is
        # held out for validation of the union, not used for selection.
        for direction, label in [(1, "up"), (-1, "down")]:
            eligible = [
                state
                for state, probability in first_branch[direction].items()
                if probability >= 0.55
            ]
            for union in combinations(eligible, 2):
                first_values = target[train & x.state.isin(union)].dropna()
                second_values = target[(~train) & x.state.isin(union)].dropna()
                first_probability = (
                    float(first_values.eq(direction).mean()) if len(first_values) else np.nan
                )
                second_probability = (
                    float(second_values.eq(direction).mean()) if len(second_values) else np.nan
                )
                if (
                    len(first_values) >= 100
                    and len(second_values) >= 100
                    and first_probability >= 0.66
                    and second_probability >= 0.66
                ):
                    rows.append(
                        {
                            "state_count": components,
                            "state": " OR ".join(map(str, union)),
                            "horizon": h,
                            "direction": label,
                            "first_probability": first_probability,
                            "second_probability": second_probability,
                            "first_events": len(first_values),
                            "second_events": len(second_values),
                            "min_probability": min(first_probability, second_probability),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["ES", "NQ", "CL", "GC", "6E"])
    p.add_argument("--output", type=Path, default=Path("outputs/dynamic_state_candidates.csv"))
    a = p.parse_args()
    out = []
    for s in a.symbols:
        path = (
            Path("data/processed/databento_research/development_minute_returns")
            / f"symbol={s}"
            / "returns.parquet"
        )
        for n in range(3, 13):
            r = scan(bars(path), n)
            if not r.empty:
                r.insert(0, "symbol", s)
                out.append(r)
    result = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False)
    print(result.sort_values("min_probability", ascending=False).head(50).to_string(index=False))


if __name__ == "__main__":
    main()
