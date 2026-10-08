"""Future-high alignment, strict availability and forecast error contracts."""

import numpy as np
import pandas as pd
from hour_high import (
    clustered_columns,
    corrected_quantiles,
    feature_table,
    high_labels,
    regression_metrics,
    temporal_parts,
)


def test_high_uses_exact_future_hour_and_reference_excludes_future_open():
    index = pd.date_range("2024-08-26 11:59", periods=122, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100, "close": 100, "high": 101, "low": 99, "volume": 10}, index=index
    )
    minute.loc[index[1], "open"] = 140
    minute.loc[index[1], "high"] = 141
    minute.loc[index[60], "high"] = 150
    minute.loc[index[61], "high"] = 999  # 13:00 belongs to the following hour
    clock = pd.DatetimeIndex([index[1], index[61]])
    x = pd.DataFrame({"scale": 20}, index=clock)
    labels = high_labels(minute, x)
    assert labels.iloc[0].reference == 100
    assert labels.iloc[0].actual_high == 150
    assert labels.iloc[0].target == 1.25  # scale is twice previous 15m range
    assert labels.iloc[1].actual_high == 999
    assert (labels.known_at <= labels.index).all()


def test_incomplete_future_hour_is_not_an_observed_label():
    index = pd.date_range("2024-08-26 11:59", periods=122, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100, "close": 100, "high": 101, "low": 99, "volume": 10}, index=index
    ).drop(index[25])
    x = pd.DataFrame({"scale": 20}, index=pd.DatetimeIndex([index[1], index[61]]))
    labels = high_labels(minute, x)
    assert index[1] not in labels.index
    assert index[61] in labels.index


def test_error_metrics_are_move_based_and_in_points():
    actual = np.array([10010, 20030, 30050])
    predicted = actual + np.array([-10, 0, 20])
    stats = regression_metrics(actual, predicted, np.array([10000, 20000, 30000]))
    assert stats["mae_points"] == 10
    assert np.isclose(stats["rmse_points"], np.sqrt(500 / 3))
    assert stats["within_10_points"] == 2 / 3
    assert stats["within_20_points"] == 1
    assert stats["mse_points_squared"] == 500 / 3
    assert stats["move_r2"] < 0.5


def test_fit_only_clustering_collapses_redundant_features():
    rng = np.random.default_rng(5)
    a = rng.normal(size=100)
    fit = pd.DataFrame(
        {"a": a, "duplicate": -a, "independent": rng.normal(size=100), "future": np.nan}
    )
    chosen = clustered_columns(fit)
    assert len(chosen) == 2
    assert "independent" in chosen
    assert "future" not in chosen


def test_quantile_shifts_use_calibration_residuals():
    predicted = np.zeros((5, 2))
    shifts = corrected_quantiles(predicted, np.arange(5), np.array([0.5, 0.8]))
    assert np.array_equal(shifts, [2, 4])


def test_hourly_label_purging_excludes_boundary():
    index = pd.date_range("2024-01-01", periods=230 * 24, freq="h", tz="UTC")
    data = pd.DataFrame({"label_end": index + pd.Timedelta(hours=1)}, index=index)
    parts = temporal_parts(
        data, pd.Timestamp("2024-06-01", tz="UTC"), pd.Timestamp("2024-07-01", tz="UTC")
    )
    for before, after in zip(parts[:-1], parts[1:], strict=True):
        assert before.label_end.max() < after.index.min()


def test_features_at_forecast_are_unchanged_after_removing_future_minutes():
    index = pd.date_range("2024-08-26 00:00", periods=3 * 1440, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100, "close": 100.1, "high": 101, "low": 99, "volume": 10}, index=index
    )
    snapshots = index[::15] + pd.Timedelta(minutes=15)
    previous = pd.DataFrame(
        {"scale": 2.0, "nq_feature": np.arange(len(snapshots)), "market_ES_return": 0.01},
        index=snapshots,
    )
    cutoff = pd.Timestamp("2024-08-27 14:00", tz="UTC")
    clock = pd.DatetimeIndex([cutoff])
    full = feature_table(minute, previous, clock)
    truncated = feature_table(
        minute.loc[minute.index < cutoff], previous.loc[previous.index <= cutoff], clock
    )
    pd.testing.assert_frame_equal(full, truncated)
