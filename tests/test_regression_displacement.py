from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.regression_displacement import (
    REGRESSION_FEATURES,
    aggregate_hourly_bars,
    build_regression_features,
    build_regression_targets,
    fit_walk_forward_models,
)


def _minute_bars(hours: int = 30) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-02", periods=hours * 60, freq="min", tz="UTC")
    close = 100.0 * np.exp(np.arange(len(timestamps)) * 0.00001)
    return pd.DataFrame(
        {
            "symbol": "A",
            "trading_date": timestamps.tz_localize(None).normalize(),
            "timestamp_utc": timestamps,
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1.0,
        }
    )


def test_hourly_aggregation_uses_only_observations_in_each_hour() -> None:
    minute = _minute_bars(hours=2)
    result = aggregate_hourly_bars(minute)
    assert len(result) == 2
    assert result.loc[0, "feature_available_at_utc"] == minute.loc[59, "timestamp_utc"]
    assert result.loc[0, "close"] == minute.loc[59, "close"]
    assert result.loc[0, "minute_observations"] == 60


def test_linear_price_path_has_zero_regression_displacement() -> None:
    hours = pd.date_range("2024-01-02", periods=180, freq="h", tz="UTC")
    log_close = np.log(100.0) + np.arange(len(hours)) * 0.001
    hourly = pd.DataFrame(
        {
            "symbol": "A",
            "trading_date": hours.tz_localize(None).normalize(),
            "hour_start_utc": hours,
            "feature_available_at_utc": hours + pd.Timedelta(minutes=59),
            "open": np.exp(log_close),
            "high": np.exp(log_close) * 1.001,
            "low": np.exp(log_close) * 0.999,
            "close": np.exp(log_close),
            "volume": 100.0,
            "minute_observations": 60,
        }
    )
    features = build_regression_features(hourly)
    mature = features.iloc[140:]
    assert mature["regression_displacement_24h"].abs().max() < 1e-10
    assert np.allclose(mature["regression_slope_24h"], 0.001, atol=1e-10)
    assert mature["regression_r_squared_24h"].min() > 0.999999


def test_target_timestamps_are_future_and_features_remain_predeclared() -> None:
    hourly = aggregate_hourly_bars(_minute_bars(hours=150))
    features = build_regression_features(hourly)
    targets = build_regression_targets(features, horizons=(1, 8))
    assert 10 <= len(REGRESSION_FEATURES) <= 30
    assert len(REGRESSION_FEATURES) == len(set(REGRESSION_FEATURES))
    valid = targets["label_available_at_8h_utc"].notna()
    assert (
        targets.loc[valid, "label_available_at_8h_utc"]
        > features.loc[valid, "feature_available_at_utc"]
    ).all()


def test_walk_forward_predictions_are_validation_only() -> None:
    dates = pd.date_range("2020-01-01", periods=70, freq="D")
    rows: list[dict[str, object]] = []
    for hour, date in enumerate(dates):
        row: dict[str, object] = {
            "symbol": "A",
            "trading_date": date,
            "hour_start_utc": date.tz_localize("UTC"),
            "feature_available_at_utc": date.tz_localize("UTC"),
            "future_log_return_8h": hour * 0.0001,
            "label_available_at_8h_utc": date.tz_localize("UTC") + pd.Timedelta(hours=8),
            "realized_volatility_24h": 0.01,
        }
        row.update({feature: float(hour) for feature in REGRESSION_FEATURES})
        rows.append(row)
    folds = pd.DataFrame(
        [
            {
                "fold": 1,
                "train_end": "2020-02-09",
                "validation_start": "2020-02-15",
                "validation_end": "2020-02-20",
            }
        ]
    )
    forecasts, coefficients = fit_walk_forward_models(pd.DataFrame(rows), folds, horizons=(8,))
    assert forecasts["trading_date"].min() == pd.Timestamp("2020-02-15")
    assert forecasts["trading_date"].max() == pd.Timestamp("2020-02-20")
    assert set(forecasts["horizon_hours"]) == {8}
    assert not coefficients.empty
