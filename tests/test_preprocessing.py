from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from systematic_research.preprocessing import (
    build_feature_registry,
    point_in_time_robust_scale,
)


def _features(rows: int) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    dates = pd.date_range("2023-01-01", periods=rows, freq="B")
    return pd.DataFrame(
        {
            "symbol": "NQU6",
            "trading_date": dates,
            "feature_available_at_utc": dates.tz_localize("UTC"),
            "feature_a": np.sin(index / 10.0) + index / 100.0,
            "feature_b": np.cos(index / 15.0),
        }
    )


def test_point_in_time_scaling_is_prefix_invariant() -> None:
    short = point_in_time_robust_scale(_features(180), ["feature_a", "feature_b"])
    long = point_in_time_robust_scale(_features(220), ["feature_a", "feature_b"])
    assert_frame_equal(short, long.iloc[:180].reset_index(drop=True))


def test_current_outlier_does_not_set_its_own_winsorization_bounds() -> None:
    frame = _features(180)
    frame.loc[179, "feature_a"] = 1_000_000.0
    transformed = point_in_time_robust_scale(frame, ["feature_a"])
    value = transformed["feature_a"].iloc[179]
    assert isinstance(value, (float, np.floating))
    assert value < 100.0


def test_registry_only_drops_declared_exact_duplicates() -> None:
    metadata = {
        "feature_columns": ["return", "negative_return", "volatility"],
        "feature_families": {"signals": ["return", "negative_return", "volatility"]},
    }
    redundancy = pd.DataFrame(
        {"feature_a": ["return"], "feature_b": ["volatility"], "correlation": [0.99]}
    )
    registry = build_feature_registry(metadata, redundancy, {"negative_return": "return"})
    assert registry.loc[registry["selected"], "feature"].tolist() == ["return", "volatility"]
