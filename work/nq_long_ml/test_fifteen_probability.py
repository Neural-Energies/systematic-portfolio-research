"""Causality and exact execution contracts for the narrowed 15-minute study."""

import numpy as np
import pandas as pd
from direction_expanded import screen_features
from experiment import aggregate
from fifteen_probability import (
    exact_labels,
    intraday_features,
    measure,
    quantile_probability,
    slot_history,
    time_parts,
)


def minute_fixture():
    index = pd.date_range("2024-08-26 00:00", periods=3 * 1440, freq="min", tz="UTC")
    rng = np.random.default_rng(14)
    price = 10000 + np.cumsum(rng.normal(0, 0.5, len(index)))
    return pd.DataFrame(
        {"open": price, "close": price + 0.1, "high": price + 1, "low": price - 1, "volume": 10},
        index=index,
    )


def test_all_features_match_when_future_minutes_are_removed():
    minute = minute_fixture()
    full = intraday_features(minute, aggregate(minute, 15))
    # Include the 09:30 decision, 09:45 and opening-range availability at 10:00.
    for end in ("2024-08-27 13:30", "2024-08-27 13:45", "2024-08-27 14:00", "2024-08-27 18:00"):
        cutoff = pd.Timestamp(end, tz="UTC")
        prefix = minute.loc[minute.index < cutoff]
        truncated = intraday_features(prefix, aggregate(prefix, 15))
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)


def test_next_bar_literal_open_close_not_delayed_and_cost_not_label():
    minute = minute_fixture()
    clock = minute.index[::15]
    minute.loc[clock, "open"] = 100
    minute.loc[clock + pd.Timedelta(minutes=1), "open"] = 101
    minute.loc[clock + pd.Timedelta(minutes=14), "close"] = 100.5
    x = pd.DataFrame({"scale": 2, "momentum64": 1}, index=clock)
    labels = exact_labels(minute, x)
    assert labels.target.eq(1).all()
    assert labels.net_dollars.eq(-15).all()
    assert labels.delayed_net_dollars.eq(-35).all()
    assert (labels.label_end - labels.index).eq(pd.Timedelta(minutes=15)).all()


def test_missing_target_minute_excludes_trade():
    minute = minute_fixture().iloc[:180].drop(minute_fixture().index[25])
    x = pd.DataFrame({"scale": 2, "momentum64": 1}, index=minute.index[::15])
    assert minute.index[15] not in exact_labels(minute, x).index


def test_same_clock_history_excludes_current_observation():
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    values = pd.Series(np.arange(10.0), index=index)
    result = slot_history(values, np.zeros(10), 5)
    assert result.iloc[5] == 2
    changed = values.copy()
    changed.iloc[5:] = 1000
    assert slot_history(changed, np.zeros(10), 5).iloc[5] == result.iloc[5]


def test_quantile_cdf_and_threshold_boundary():
    q = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(
        quantile_probability(
            np.array([[-2, -1, 0, 1, 2], [1, 2, 3, 4, 5], [-5, -4, -3, -2, -1]]), q
        ),
        [0.5, 0.9, 0.1],
    )
    data = pd.DataFrame(
        {
            "target": [1, 0],
            "net_dollars": [5, -10],
            "stress_dollars": [-20, -35],
            "delayed_net_dollars": [1, -20],
        }
    )
    assert measure(data, np.array([0.66, 0.659999]))["trades"] == 1


def test_temporal_partitions_purge_label_boundary():
    index = pd.date_range("2024-01-01", periods=200 * 96, freq="15min", tz="UTC")
    data = pd.DataFrame({"label_end": index + pd.Timedelta(minutes=15)}, index=index)
    start = pd.Timestamp("2024-06-01", tz="UTC")
    parts = time_parts(data, start, start + pd.Timedelta(days=30))
    for a, b in zip(parts[:-1], parts[1:], strict=True):
        assert a.label_end.max() < b.index.min()


def test_market_absent_in_earliest_screen_keeps_feature_schema():
    rng = np.random.default_rng(15)
    fit = pd.DataFrame({"known": rng.normal(size=100), "new_market": np.nan})
    validation = pd.DataFrame({"known": rng.normal(size=100), "new_market": rng.normal(size=100)})
    ranking = screen_features(
        fit, pd.Series(np.arange(100) % 2), validation, pd.Series(np.arange(100) % 2)
    )
    assert set(ranking.index) == set(fit.columns)
    assert ranking.new_market == 0
