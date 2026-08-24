import numpy as np
import pandas as pd

from systematic_research.intraday_strategy_search import (
    IntradaySpec,
    apply_costs,
    build_score,
    resample_bars,
    simulate,
    strategy_specs,
)


def test_registry_covers_requested_intraday_dimensions() -> None:
    specs = strategy_specs()
    assert len(specs) == 576
    assert {spec.timeframe_minutes for spec in specs} == {15, 30, 60, 240}
    assert {spec.theme for spec in specs} == {"mean_reversion", "relative_value", "lead_lag"}
    assert {spec.volatility_filter for spec in specs} >= {"below_median", "above_median"}
    assert {spec.time_rule for spec in specs} >= {"us_core", "overnight"}
    assert any(spec.stop_loss is not None for spec in specs)
    assert max(spec.maximum_hold_bars for spec in specs) == 24


def test_resample_never_crosses_trading_date() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 4,
            "trading_date": pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2),
            "timestamp_utc": pd.to_datetime(
                ["2024-01-02 23:00Z", "2024-01-02 23:01Z", "2024-01-03 23:00Z", "2024-01-03 23:01Z"]
            ),
            "open": [100.0, 101.0, 200.0, 202.0],
            "high": [101.0, 102.0, 202.0, 203.0],
            "low": [99.0, 100.0, 199.0, 201.0],
            "close": [101.0, 102.0, 202.0, 203.0],
            "volume": [1, 2, 3, 4],
        }
    )
    result = resample_bars(frame, 30)
    assert len(result) == 2
    assert result["open"].tolist() == [100.0, 200.0]
    assert result["volume"].tolist() == [3, 7]


def test_simulation_delays_entries_and_applies_costs() -> None:
    index = pd.date_range("2024-01-02 14:00Z", periods=4, freq="30min")
    scores = pd.DataFrame({"A": [2.0, 0.5, 0.0, 0.0]}, index=index)
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.00]}, index=index)
    dates = pd.Series(pd.Timestamp("2024-01-02"), index=index)
    spec = IntradaySpec("x", "mean_reversion", 30, 2, 1.0, 0.1, 2, "all", "all", None)
    gross, turnover = simulate(scores, returns, dates, spec)
    assert np.isclose(gross.iloc[0], 0.01)
    assert np.isclose(turnover.iloc[0], 2.0)
    assert np.isclose(apply_costs(gross, turnover, 5.0).iloc[0], 0.009)


def test_lead_lag_score_preserves_the_asset_panel() -> None:
    index = pd.date_range("2024-01-02", periods=50, freq="30min", tz="UTC")
    returns = pd.DataFrame(
        {"A": np.linspace(-0.01, 0.01, 50), "B": np.linspace(0.02, -0.01, 50)},
        index=index,
    )
    close = (1.0 + returns).cumprod()
    score = build_score(close, returns, "lead_lag", 2)
    assert score.columns.tolist() == ["A", "B"]
    assert score.shape == returns.shape


def test_position_can_persist_across_sessions() -> None:
    index = pd.to_datetime(["2024-01-02 21:00Z", "2024-01-03 00:00Z", "2024-01-03 21:00Z"])
    scores = pd.DataFrame({"A": [2.0, 0.5, 0.5]}, index=index)
    returns = pd.DataFrame({"A": [0.0, 0.01, 0.02]}, index=index)
    dates = pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]), index=index)
    spec = IntradaySpec("x", "mean_reversion", 240, 2, 1.0, 0.1, 4, "all", "all", None)
    gross, _ = simulate(scores, returns, dates, spec)
    assert np.isclose(gross.loc[pd.Timestamp("2024-01-03")], 0.03)
