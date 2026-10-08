"""Next-bar event, separate cost outcome, causal rolls and calibrated-probability contracts."""

import numpy as np
import pandas as pd
from direction_expanded import (
    ProbabilityCalibration,
    contract_safe_bars,
    feature_matrices,
    literal_bar_labels,
    metrics,
    observed_predictor_bars,
)


def test_probability_event_is_bar_up_not_costed_profit():
    index = pd.date_range("2024-08-26 12:00", periods=180, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "close": 100.5, "high": 101.0, "low": 99.0, "volume": 10}, index=index
    )
    clock = index[::60]
    x = pd.DataFrame({"momentum_64": [1, 2, 3], "feature": [10, 20, 30]}, index=clock)
    result = literal_bar_labels(minute, x, 60)
    assert result.target.eq(1).all()  # small price increase is UP
    assert result.net_dollars.eq(-15).all()  # $10 increase does not cover $25 costs
    assert result.iloc[1].feature == 20
    assert result.iloc[1].label_end == index[120]


def test_next_hour_label_is_missing_if_future_bar_has_a_gap():
    index = pd.date_range("2024-08-26 12:00", periods=180, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0, "volume": 10}, index=index
    ).drop(index[90])
    x = pd.DataFrame({"momentum_64": 1}, index=index[::60])
    result = literal_bar_labels(minute, x, 60)
    assert index[60] not in result.index


def test_rolling_inputs_exclude_bar_that_crosses_a_contract_change():
    index = pd.date_range("2024-08-26 12:00", periods=120, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {
            "open": 100.0,
            "close": 101.0,
            "high": 102.0,
            "low": 99.0,
            "volume": 10,
            "_contract": "NQU4",
        },
        index=index,
    )
    minute.loc[index[30] :, "_contract"] = "NQZ4"
    bars = contract_safe_bars(minute, 60)
    assert index[0] not in bars.index
    assert index[60] in bars.index


def test_unavailable_market_is_excluded_from_training_pca():
    index = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"known": rng.normal(size=100), "future_market": np.nan}, index=index)
    markets = pd.DataFrame({"ES": rng.normal(size=100), "RTY": np.nan}, index=index)
    matrices, _, observed = feature_matrices([x], x, markets)
    assert observed == ["ES"]
    assert "future_market" not in matrices[0]


def test_selective_accuracy_is_distinct_from_probability_and_overall_accuracy():
    data = pd.DataFrame({"target": [1, 1, 0, 0], "net_dollars": [10, 20, -30, -40]})
    p = np.array([0.68, 0.68, 0.68, 0.55])
    score = metrics(data, p)
    assert score["signals_066"] == 3
    assert np.isclose(score["signal_accuracy"], 2 / 3)
    assert np.isclose(score["mean_signal_probability"], 0.68)
    assert score["accuracy"] == 0.5
    assert score["signal_net_dollars"] == 0


def test_no_high_confidence_scores_produces_no_signals():
    data = pd.DataFrame({"target": [1, 0, 1, 0], "net_dollars": [10, -20, 30, -40]})
    score = metrics(data, np.array([0.55, 0.48, 0.51, 0.50]))
    assert score["signals_066"] == 0
    assert np.isnan(score["signal_accuracy"])


def test_binned_calibration_is_monotone_and_within_probability_bounds():
    score = np.linspace(-2, 2, 300)
    y = pd.Series((score > 0).astype(int))
    calibration = ProbabilityCalibration("binned_isotonic").fit(score, y)
    p = calibration.predict(np.linspace(-10, 10, 100))
    assert np.all(np.diff(p) >= 0)
    assert ((p > 0) & (p < 1)).all()
    assert p[0] < 0.5 < p[-1]


def test_completed_sparse_predictor_has_coverage_and_quote_age():
    index = pd.date_range("2024-08-26 12:00", periods=60, freq="min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "close": 101.0, "high": 102.0, "low": 99.0, "volume": 10}, index=index
    )
    sparse = minute.drop(index[30])
    result = observed_predictor_bars(sparse, 60)
    assert len(result) == 1
    assert result.iloc[0].available_at == index[0] + pd.Timedelta(hours=1)
    assert result.iloc[0].coverage_fraction == 59 / 60
    assert result.iloc[0].quote_age_minutes == 0
    assert observed_predictor_bars(minute.iloc[:40], 60).empty
