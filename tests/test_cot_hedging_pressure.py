import pandas as pd

from systematic_research.cot_hedging_pressure import (
    build_signal_panels,
    portfolio_returns,
)


def _reports() -> pd.DataFrame:
    values = {"CL": 0.1, "GC": 0.2, "HG": 0.3, "NG": 0.4, "ZC": 0.5}
    return pd.DataFrame(
        {
            "symbol": list(values),
            "report_date": pd.Timestamp("2024-01-02"),
            "release_date": pd.Timestamp("2024-01-05"),
            "hedging_pressure": list(values.values()),
        }
    )


def test_signal_waits_until_first_session_after_friday_release() -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08"]))
    panel = build_signal_panels(_reports(), dates)[1.0]
    assert panel.loc["2024-01-02"].abs().sum() == 0.0
    assert panel.loc["2024-01-05"].abs().sum() == 0.0
    assert panel.loc["2024-01-08"].abs().sum() == 2.0


def test_signal_longs_high_and_shorts_low_hedging_pressure() -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-05", "2024-01-08"]))
    panel = build_signal_panels(_reports(), dates)[1.0]
    assert panel.loc["2024-01-08", "ZC"] == 1.0
    assert panel.loc["2024-01-08", "CL"] == -1.0
    assert panel.loc["2024-01-08", ["GC", "HG", "NG"]].abs().sum() == 0.0


def test_portfolio_cost_is_full_entry_exit_round_trip() -> None:
    dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-08"]))
    positions = pd.DataFrame({"CL": [-1.0], "ZC": [1.0]}, index=dates)
    returns = pd.DataFrame({"CL": [0.0], "ZC": [0.0]}, index=dates)
    portfolio, legs = portfolio_returns(positions, returns, one_way_cost_bps=1.0)
    assert abs(portfolio.iloc[0] + 0.0002) < 1e-12
    assert (legs.iloc[0] == -0.0002).all()
