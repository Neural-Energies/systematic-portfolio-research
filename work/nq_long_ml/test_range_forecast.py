"""Causal alignment, mature labels, executable targets and calendar regression checks."""

import numpy as np
import pandas as pd
from medium_frequency import build_labels, daily_policy
from range_forecast import (
    carry_completed,
    daily_bars,
    excursion_labels,
    pca_matrices,
    quantile_predict,
    split_training,
    target_exits,
)
from test_medium_frequency import predictions


def test_completed_alignment_never_reads_future_or_stale_values():
    index = pd.date_range("2024-01-02", periods=3, freq="h", tz="UTC")
    values = pd.DataFrame({"value": [10.0, 999.0]}, index=index[[0, 2]])
    aligned = carry_completed(values, index, 0.5)
    assert aligned.iloc[0, 0] == 10
    assert np.isnan(aligned.iloc[1, 0])
    changed = values.copy()
    changed.iloc[1, 0] = -999
    pd.testing.assert_frame_equal(aligned.iloc[:2], carry_completed(changed, index, 0.5).iloc[:2])


def test_daily_features_wait_for_scheduled_close():
    index = pd.date_range("2024-03-11 00:00", periods=300, freq="min", tz="UTC")
    minutes = pd.DataFrame(
        {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 5}, index=index
    )
    bars = daily_bars(minutes)
    assert bars.iloc[0].available_at == pd.Timestamp("2024-03-11 21:00", tz="UTC")


def test_future_excursion_excludes_exit_minute_high():
    index = pd.date_range("2024-08-26 12:00", periods=600, freq="min", tz="UTC")
    minutes = pd.DataFrame({"open": 100.0, "high": 120.0, "low": 95.0, "close": 100.0}, index=index)
    grid = pd.DataFrame({"feature": 1}, index=index[:1])
    frame = build_labels(minutes, grid, 0)
    minutes.loc[index[241], "high"] = 999
    result = excursion_labels(frame, minutes, pd.Series(10.0, index=grid.index))
    assert result.iloc[0].mfe == 2
    assert result.iloc[0].mae == 0.5


def test_target_execution_uses_predicted_target_after_minimum_hold():
    index = pd.date_range("2024-08-26 12:00", periods=600, freq="min", tz="UTC")
    minutes = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index)
    minutes.loc[index[100], "high"] = 999  # early touch must not close the position
    minutes.loc[index[300], "high"] = 999  # later touch fills predicted 105, not 999
    frame = pd.DataFrame(
        {
            "probability": 0.8,
            "execution_valid": True,
            "entry_position": 1,
            "max_exit_position": 550,
            "entry_time": index[1],
            "scale": 10.0,
            "q50": 0.5,
        },
        index=index[:1],
    )
    result = target_exits(frame, minutes, 50)
    assert result.iloc[0].exit_time == index[300]
    assert result.iloc[0].exit_price == 105
    assert result.iloc[0].observed_minutes_held == 299
    assert result.iloc[0].exit_reason == "predicted_target"


def test_quantiles_are_nonnegative_and_ordered():
    x = pd.DataFrame({"x": [1, 2]})
    result = quantile_predict([3.0, 1.0, -1.0], x, np.zeros(3))
    np.testing.assert_array_equal(result, [[0, 1, 3], [0, 1, 3]])


def test_shortened_horizon_is_not_a_multi_day_range_label():
    index = pd.date_range("2024-08-26 12:00", periods=600, freq="min", tz="UTC")
    minutes = pd.DataFrame({"open": 100.0, "high": 120.0, "low": 95.0, "close": 100.0}, index=index)
    grid = pd.DataFrame({"feature": 1}, index=index[:1])
    frame = build_labels(minutes, grid, 5, conservative_roll_guard=False)
    assert frame.iloc[0].effective_days == 0
    result = excursion_labels(frame, minutes, pd.Series(10.0, index=grid.index), required_days=5)
    assert result["mfe"].isna().all()


def test_split_purges_labels_crossing_each_following_segment():
    index = pd.date_range("2023-01-01", periods=1000, freq="h", tz="UTC")
    frame = pd.DataFrame({"mfe": 1.0, "label_end": index + pd.Timedelta(hours=60)}, index=index)
    parts = split_training(frame, index[-1] + pd.Timedelta(hours=200), "mfe")
    for earlier, later in zip(parts[:-1], parts[1:], strict=True):
        assert earlier.label_end.max() < later.index.min()


def test_es_ablation_preserves_regression_features_and_excludes_es_from_pca():
    index = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    rng = np.random.default_rng(10)
    factors = pd.DataFrame(
        rng.normal(size=(100, 5)), columns=["ES", "ZN", "GC", "CL", "6J"], index=index
    )
    x = pd.DataFrame(
        {"regression_slope_4": 1.0, "es_relative_strength4": 2.0, "factor_beta_ES": 3.0},
        index=index,
    )
    matrices, _, markets = pca_matrices(x, factors, [x], False)
    assert "ES" not in markets
    assert "regression_slope_4" in matrices[0]
    assert "es_relative_strength4" not in matrices[0]
    assert "factor_beta_ES" not in matrices[0]


def test_carter_closure_deadline_is_early_and_session_is_not_consumed_tomorrow():
    frame = predictions().iloc[:3].copy()
    frame.index = pd.date_range("2025-01-09 13:00", periods=3, freq="2h", tz="UTC")
    frame["entry_time"] = frame.index + pd.Timedelta(minutes=1)
    frame["session"] = pd.Timestamp("2025-01-08")
    result = daily_policy(frame, 0.66, True, conservative_roll_guard=False)
    assert result.iloc[0].decision_time == frame.index[0]
    assert result.iloc[0].quota_fallback
