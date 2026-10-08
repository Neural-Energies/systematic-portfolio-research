"""Temporal and economic contract tests using independently specified outcomes."""

import numpy as np
import pandas as pd
from experiment import aggregate, features, labeled_frame, metrics, signal_backtest


def minute_data():
    index = pd.date_range("2024-01-02 23:00", periods=120, freq="min", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1}, index=index
    )


def test_missing_minute_rejects_whole_bar():
    data = minute_data().drop(pd.Timestamp("2024-01-02 23:07", tz="UTC"))
    bars = aggregate(data, 15)
    assert len(bars) == 7
    assert pd.Timestamp("2024-01-02 23:00", tz="UTC") not in bars.index
    assert bars.iloc[0]["available_at"] == pd.Timestamp("2024-01-02 23:30", tz="UTC")


def test_future_mutation_does_not_change_past_features():
    bars = aggregate(minute_data(), 15)
    factors = pd.DataFrame(
        {s: np.arange(len(bars)) * 0.001 for s in ("ES", "ZN", "GC", "CL", "6J")}, index=bars.index
    )
    before = features(bars, factors)
    changed = bars.copy()
    changed.loc[changed.index[-1], "close"] = 10000
    after = features(changed, factors)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])
    assert len(before.columns) == 90


def test_label_requires_contiguous_next_bar():
    bars = aggregate(minute_data(), 15)
    bars = bars.drop(bars.index[3])
    x = pd.DataFrame({"momentum_64": 1.0}, index=bars.index)
    labeled = labeled_frame(bars, x, 15)
    assert bars.index[2] not in labeled.index
    # A one-point gain is $20, below the declared $25 round-trip cost.
    assert labeled["target"].eq(0).all()


def predictions():
    time = pd.date_range("2024-01-03", periods=4, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "available_at": time,
            "label_end": time + pd.Timedelta(hours=1),
            "session": time[0],
            "probability": [0.66, 0.7, 0.49, 0.1],
            "target": [1, 1, 0, 0],
            "trade_open": [100, 102, 104, 103],
            "trade_close": [102, 104, 103, 102],
            "ema_distance_4": 1,
            "regression_slope_8": 1,
        }
    )


def test_threshold_cost_and_signal_exit():
    frame = predictions()
    result = metrics(frame)
    assert result["trades"] == 2
    assert result["net_dollars_1_contract"] == 30  # two $40 gross gains, $25 each
    exit_result = signal_backtest(frame, "none")
    assert exit_result["trades"] == 1
    assert exit_result["net_dollars"] == 55  # enter 100, exit 104 when probability falls
    assert signal_backtest(frame, "none", stress=2)["net_dollars"] == 30


def test_zero_trade_results_are_retained():
    frame = predictions()
    frame["probability"] = 0.6
    assert metrics(frame)["trades"] == 0
    assert metrics(frame)["net_dollars_1_contract"] == 0
    assert signal_backtest(frame, "none")["trades"] == 0


def test_factor_beta_uses_paired_observations():
    index = pd.date_range("2024-01-03", periods=100, freq="h", tz="UTC")
    market = pd.Series(np.sin(np.arange(100)) * 0.001, index=index)
    r = market * 2 + 0.0001
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "close": np.exp(r) * 100,
            "high": 102.0,
            "low": 98.0,
            "volume": np.arange(100) + 1,
            "session": index.floor("D"),
        },
        index=index,
    )
    factors = pd.DataFrame({s: market for s in ("ES", "ZN", "GC", "CL", "6J")})
    factors.iloc[::3] = np.nan
    generated = features(bars, factors)
    assert abs(generated["factor_beta_ES"].iloc[-1] - 2) < 1e-10
    assert abs(generated["factor_alpha_ES"].iloc[-1] - 0.0001) < 1e-10
    changed = bars.copy()
    changed.loc[index[-1], "close"] = 10000
    pd.testing.assert_frame_equal(generated.iloc[:-1], features(changed, factors).iloc[:-1])
