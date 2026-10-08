"""Past-session, same-clock upward-excursion summaries with explicit eligibility."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def excursion_summary(
    values: NDArray[np.float64], minimum: int = 5, trim_iqr: bool = True
) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if np.any(x < 0):
        raise ValueError("High minus open must be nonnegative")
    result = {"qualified_samples": float(len(x)), "retained_samples": 0.0, "removed_outliers": 0.0}
    fields = (
        "mean",
        "median",
        "std",
        "q10",
        "q25",
        "q50",
        "q70",
        "q75",
        "q90",
        "iqr",
        "lower_fence",
        "upper_fence",
        "mean70",
    )
    result.update({key: np.nan for key in fields})
    if len(x) < minimum:
        return result
    q1, q3 = np.quantile(x, [0.25, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    kept = x[(x >= lo) & (x <= hi)] if trim_iqr else x
    result.update(
        {
            "retained_samples": float(len(kept)),
            "removed_outliers": float(len(x) - len(kept)),
            "lower_fence": float(lo),
            "upper_fence": float(hi),
        }
    )
    if len(kept) < minimum:
        return result
    qs = np.quantile(kept, [0.1, 0.25, 0.5, 0.7, 0.75, 0.9])
    result.update(
        dict(zip(("q10", "q25", "q50", "q70", "q75", "q90"), map(float, qs), strict=True))
    )
    result.update(
        {
            "mean": float(kept.mean()),
            "median": float(np.median(kept)),
            "std": float(kept.std(ddof=1)),
            "iqr": float(qs[4] - qs[1]),
            "mean70": float(0.7 * kept.mean()),
        }
    )
    return result


def same_clock_history(
    frame: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    lookback: int = 20,
    minimum: int = 5,
    trim_iqr: bool = True,
) -> pd.DataFrame:
    """Previous 20 session positions, NOT the last 20 qualifying observations.

    Required columns: anchor, slot, excursion, opening_up, scorable. Excludes the
    current session completely; never strips future outcome outliers.
    """
    if lookback < 1 or minimum < 2 or minimum > lookback:
        raise ValueError("Invalid historical window")
    rows = []
    for _, g in frame.groupby("slot", sort=False):
        if g.anchor.duplicated().any():
            raise ValueError("Duplicate session/clock observations")
        values = (
            g.set_index("anchor")
            .excursion.where(g.set_index("anchor").opening_up & g.set_index("anchor").scorable)
            .reindex(sessions)
        )
        lookup = {stamp: i for i, stamp in enumerate(sessions)}
        for stamp, r in g.iterrows():
            i = lookup[r.anchor]
            if i < lookback:
                continue
            stats = excursion_summary(
                values.iloc[i - lookback : i].to_numpy(dtype=float), minimum, trim_iqr
            )
            rows.append({"timestamp": stamp, "past_sessions": lookback, **stats})
    if not rows:
        return pd.DataFrame(index=frame.index)
    return pd.DataFrame(rows).set_index("timestamp").reindex(frame.index)
