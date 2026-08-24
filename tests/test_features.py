from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from systematic_research.features import (
    FEATURE_COLUMNS,
    build_daily_realized_measures,
    build_session_features,
)


def _panel(rows: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    path_returns = -0.002 + np.arange(rows) * 0.00005
    close = 100.0 * np.cumprod(1.0 + path_returns)
    return pd.DataFrame(
        {
            "symbol": "NQ",
            "trading_date": dates,
            "session_close_utc": dates.tz_localize("UTC") + pd.Timedelta(hours=21),
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.arange(rows) + 1_000,
            "close_to_close_return": pd.Series(close).pct_change(),
            "realized_volatility_1m": 0.005 + np.arange(rows) * 0.0001,
        }
    )


def test_features_are_prefix_invariant_when_future_data_is_appended() -> None:
    short = build_session_features(_panel(80))
    long = build_session_features(_panel(100)).iloc[:80].reset_index(drop=True)
    assert_frame_equal(short, long)


def test_features_have_availability_timestamp_and_no_targets() -> None:
    features = build_session_features(_panel())
    assert "feature_available_at_utc" in features
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert not any("target" in column.lower() for column in features.columns)
    assert pd.notna(features.loc[features.index[-1], "momentum_60s"])


def test_daily_realized_measures_match_independent_calculations() -> None:
    arithmetic_returns = np.array([0.01, -0.02, 0.03])
    log_returns = np.log1p(arithmetic_returns)
    minute = pd.DataFrame(
        {
            "symbol": "NQ",
            "trading_date": pd.Timestamp("2024-01-02"),
            "timestamp_utc": pd.date_range("2024-01-02", periods=3, freq="min", tz="UTC"),
            "return_1m": arithmetic_returns,
        }
    )

    result = build_daily_realized_measures(minute).iloc[0]

    expected_variance = np.square(log_returns).sum()
    expected_bipower = np.pi / 2.0 * np.abs(log_returns[1:] * log_returns[:-1]).sum()
    assert np.isclose(result["realized_variance"], expected_variance)
    assert np.isclose(result["downside_semivariance"], log_returns[1] ** 2)
    assert np.isclose(result["upside_semivariance"], log_returns[0] ** 2 + log_returns[2] ** 2)
    assert np.isclose(result["bipower_variation"], expected_bipower)
    assert np.isclose(result["jump_variation"], max(expected_variance - expected_bipower, 0.0))


def test_standardized_returns_are_scale_free() -> None:
    features = build_session_features(_panel())
    last = features.iloc[-1]
    assert np.isfinite(last["standardized_return_5s"])
    assert np.isfinite(last["standardized_return_20s"])
    assert np.isfinite(last["standardized_return_60s"])
