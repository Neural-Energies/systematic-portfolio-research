import numpy as np
import pandas as pd

from systematic_research.trade_report_metrics import trade_kpis


def test_trade_pf_and_cash_calendar_returns_are_not_daily_pf() -> None:
    days = pd.date_range("2026-01-05", periods=3, freq="B", tz="UTC")
    trades = pd.DataFrame(
        {
            "anchor": [days[0], days[0], days[2]],
            "status": ["target", "timeout", "target"],
            "net_dollars": [200.0, -100.0, 50.0],
            "holding_minutes": [1, 60, 2],
            "target_points": [12.0, 12.0, 4.0],
        }
    )
    result, curve = trade_kpis(trades, days, capital=1000.0)
    assert result["profit_factor"] == 2.5
    assert result["net_profit"] == 150.0
    assert result["session_coverage"] == 2 / 3
    np.testing.assert_allclose(curve["return"], [0.1, 0.0, 50 / 1100])
    expected = curve["return"].mean() / curve["return"].std(ddof=1) * np.sqrt(252)
    assert result["sharpe"] == expected


def test_initial_account_high_is_included_in_drawdown() -> None:
    days = pd.date_range("2026-01-05", periods=2, freq="B", tz="UTC")
    trades = pd.DataFrame(
        {
            "anchor": days,
            "status": ["timeout", "target"],
            "net_dollars": [-100.0, 200.0],
            "holding_minutes": [60, 1],
            "target_points": [10.0, 10.0],
        }
    )
    result, _ = trade_kpis(trades, days, capital=1000.0)
    assert result["max_daily_drawdown_dollars"] == 100.0
    assert result["max_daily_drawdown_pct"] == 0.1
