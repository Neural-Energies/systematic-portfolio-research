from pathlib import Path

import numpy as np
import pandas as pd

from systematic_research.intraday_strategy_search import resample_bars
from systematic_research.strategy_factory import ProductStrategy
from systematic_research.vectorbt_strategy_factory import (
    DEVELOPMENT_MINUTE_ROOT,
    delayed_signals,
    optimized_portfolios,
    run_vectorbt_batch,
)


def _spec(name: str = "test") -> ProductStrategy:
    return ProductStrategy(
        name=name,
        symbol="ES",
        timeframe_minutes=30,
        signal_family="momentum",
        lookback_bars=2,
        direction=1,
        entry_threshold=1.0,
        exit_threshold=0.1,
        maximum_hold_bars=2,
        volatility_filter="all",
        time_rule="all",
        stop_loss=0.5,
        profit_target=0.5,
    )


def test_factory_points_only_to_databento_development_data() -> None:
    assert (
        Path("data/processed/databento_research/development_minute_returns")
        == DEVELOPMENT_MINUTE_ROOT
    )
    assert "sealed" not in DEVELOPMENT_MINUTE_ROOT.as_posix()


def test_signals_are_delayed_to_the_next_bar() -> None:
    index = pd.date_range("2024-01-02 14:00Z", periods=5, freq="30min")
    score = pd.Series([2.0, 2.0, 0.0, -2.0, -2.0], index=index)
    eligible = pd.Series(True, index=index)
    long_entries, long_exits, short_entries, _ = delayed_signals(score, eligible, _spec())
    assert not long_entries.iloc[0]
    assert long_entries.iloc[1]
    assert long_exits.iloc[3]
    assert short_entries.iloc[4]


def test_vectorbt_batch_produces_trade_ledger_and_costed_returns() -> None:
    index = pd.date_range("2024-01-02 14:00Z", periods=6, freq="30min")
    bars = pd.DataFrame(
        {
            "timestamp_utc": index,
            "symbol": "ES",
            "trading_date": pd.Timestamp("2024-01-02"),
            "open": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0],
            "high": [100.1, 100.1, 101.1, 102.1, 101.1, 100.1],
            "low": [99.9, 99.9, 100.9, 101.9, 100.9, 99.9],
            "close": [100.0, 100.0, 101.0, 102.0, 101.0, 100.0],
            "volume": 1,
            "return": [np.nan, 0.0, 0.01, 0.0099, -0.0098, -0.0099],
        }
    )
    score = pd.DataFrame({"ES": [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]}, index=index)
    eligible = pd.DataFrame({"ES": True}, index=index)
    scores = {("momentum", 2): score}
    eligibility = {("all", "all"): eligible}
    free, _, _ = run_vectorbt_batch(bars, [_spec()], scores, eligibility, cost_bps=0.0)
    costed, entries, records = run_vectorbt_batch(
        bars, [_spec()], scores, eligibility, cost_bps=5.0
    )
    assert not records.empty
    assert entries["test"].sum() == 1
    assert costed["test"].iloc[0] < free["test"].iloc[0]
    assert records["Entry Fees"].sum() > 0.0


def test_pypfopt_weights_are_fitted_only_to_supplied_allocation_rows() -> None:
    index = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {
            "a": rng.normal(0.0010, 0.01, len(index)),
            "b": rng.normal(0.0008, 0.008, len(index)),
            "c": rng.normal(0.0006, 0.006, len(index)),
        },
        index=index,
    )
    summary = pd.DataFrame(
        {
            "strategy": ["a", "b", "c"],
            "selection_annualized_return": [0.25, 0.20, 0.15],
        }
    )
    portfolios, weights, statuses = optimized_portfolios(
        returns, summary, np.arange(len(index)) < 60
    )
    assert "pypfopt_hrp" in portfolios
    assert statuses["pypfopt_hrp"] == "ok"
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_resample_uses_nominal_clock_not_each_products_last_print() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["ES", "ES", "ZN", "ZN"],
            "trading_date": "2024-01-02",
            "timestamp_utc": pd.to_datetime(
                ["2024-01-02 14:00Z", "2024-01-02 14:28Z", "2024-01-02 14:00Z", "2024-01-02 14:29Z"]
            ),
            "open": [100.0, 101.0, 110.0, 111.0],
            "high": [100.0, 101.0, 110.0, 111.0],
            "low": [100.0, 101.0, 110.0, 111.0],
            "close": [100.0, 101.0, 110.0, 111.0],
            "volume": 1,
        }
    )
    bars = resample_bars(raw, 30)
    assert bars["timestamp_utc"].nunique() == 1
    assert bars["timestamp_utc"].iloc[0] == pd.Timestamp("2024-01-02 14:00Z")


def test_fifteen_minute_bars_are_supported() -> None:
    raw = pd.DataFrame(
        {
            "symbol": "ES",
            "trading_date": "2024-01-02",
            "timestamp_utc": pd.to_datetime(
                ["2024-01-02 14:00Z", "2024-01-02 14:14Z", "2024-01-02 14:15Z"]
            ),
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 102.0],
            "low": [100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0],
            "volume": 1,
        }
    )
    bars = resample_bars(raw, 15)
    assert bars["timestamp_utc"].tolist() == [
        pd.Timestamp("2024-01-02 14:00Z"),
        pd.Timestamp("2024-01-02 14:15Z"),
    ]


def test_four_hour_bars_do_not_collide_at_the_futures_session_boundary() -> None:
    raw = pd.DataFrame(
        {
            "symbol": "ES",
            "trading_date": ["2024-01-02", "2024-01-03"],
            "timestamp_utc": pd.to_datetime(["2024-01-02 20:59Z", "2024-01-02 23:00Z"]),
            "open": [100.0, 101.0],
            "high": [100.0, 101.0],
            "low": [100.0, 101.0],
            "close": [100.0, 101.0],
            "volume": 1,
        }
    )
    bars = resample_bars(raw, 240)
    assert not bars.duplicated(["timestamp_utc", "symbol"]).any()


def test_batch_delays_on_the_products_own_clock_without_filled_prices() -> None:
    times = pd.to_datetime(
        ["2024-01-02 14:00Z", "2024-01-02 14:30Z", "2024-01-02 15:00Z", "2024-01-02 15:30Z"]
    )
    bars = pd.DataFrame(
        {
            "timestamp_utc": [times[0], times[2], times[3], times[1]],
            "symbol": ["ES", "ES", "ES", "ZN"],
            "trading_date": pd.Timestamp("2024-01-02"),
            "open": [100.0, 101.0, 102.0, 110.0],
            "high": [100.1, 101.1, 102.1, 110.1],
            "low": [99.9, 100.9, 101.9, 109.9],
            "close": [100.0, 101.0, 102.0, 110.0],
            "volume": 1,
            "return": [np.nan, 0.01, 0.0099, np.nan],
        }
    )
    score = pd.DataFrame({"ES": [2.0, np.nan, 2.0, 0.0]}, index=times)
    eligible = pd.DataFrame({"ES": True}, index=times)
    _, _, records = run_vectorbt_batch(
        bars,
        [_spec("own_clock")],
        {("momentum", 2): score},
        {("all", "all"): eligible},
        cost_bps=0.0,
    )
    assert pd.Timestamp(records.iloc[0]["Entry Timestamp"]) == times[2]


def test_stops_are_anchored_to_fill_price_not_future_close() -> None:
    index = pd.date_range("2024-01-02 14:00Z", periods=3, freq="30min")
    bars = pd.DataFrame(
        {
            "timestamp_utc": index,
            "symbol": "ES",
            "trading_date": pd.Timestamp("2024-01-02"),
            "open": [100.0, 100.0, 120.0],
            "high": [100.0, 121.0, 120.0],
            "low": [100.0, 99.0, 120.0],
            "close": [100.0, 120.0, 120.0],
            "volume": 1,
            "return": [np.nan, 0.20, 0.0],
        }
    )
    score = pd.DataFrame({"ES": [2.0, 2.0, 2.0]}, index=index)
    eligible = pd.DataFrame({"ES": True}, index=index)
    _, _, records = run_vectorbt_batch(
        bars,
        [_spec("fill_stop")],
        {("momentum", 2): score},
        {("all", "all"): eligible},
        cost_bps=0.0,
    )
    assert float(records.iloc[0]["Return"]) > 0.19
