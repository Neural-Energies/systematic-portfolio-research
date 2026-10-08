"""Trade KPIs and calendar-complete daily account returns for research reports."""

from __future__ import annotations

import numpy as np
import pandas as pd


def trade_kpis(
    trades: pd.DataFrame, sessions: pd.DatetimeIndex, capital: float = 100_000.0
) -> tuple[dict[str, float], pd.DataFrame]:
    """Constant one-contract exposure, account returns, zero-trade sessions included.

    Profit factor here uses individual net trade dollars, not aggregated daily
    returns. Sharpe is mean daily account return / sample SD * sqrt(252), rf=0.
    End-of-day drawdown differs from minute-marked or intraminute drawdown.
    """
    if capital <= 0 or not len(sessions):
        raise ValueError("Positive reference capital and nonempty session calendar required")
    valid = trades.loc[trades.status.isin(["target", "timeout", "stop"])].copy()
    pnl = valid.net_dollars.astype(float)
    daily = valid.groupby("anchor").net_dollars.sum().reindex(sessions, fill_value=0.0)
    balance = capital + daily.cumsum()
    previous = balance.shift(fill_value=capital)
    returns = daily / previous
    insolvent = bool((balance <= 0).any() or (previous <= 0).any())
    if insolvent:
        returns[:] = np.nan
    curve = pd.DataFrame(
        {"pnl_dollars": daily, "balance": balance, "return": returns}, index=sessions
    )
    equity = np.r_[capital, balance.to_numpy()]
    peaks = np.maximum.accumulate(equity)
    dd_dollars = peaks - equity
    dd_pct = dd_dollars / peaks
    curve["drawdown_pct"] = dd_pct[1:]
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    gain = float(winners.sum())
    loss = float(-losers.sum())
    sd = float(returns.std(ddof=1))
    mean = float(returns.mean())
    downside = float(np.sqrt(np.minimum(returns.to_numpy(), 0.0).__pow__(2).mean()))
    sharpe = mean / sd * np.sqrt(252) if sd > 0 else np.nan
    sortino = mean / downside * np.sqrt(252) if downside > 0 else np.nan
    daily_cagr = (
        (float(balance.iloc[-1]) / capital) ** (252 / len(sessions)) - 1
        if not insolvent
        else np.nan
    )
    longest_loss = longest_win = current_loss = current_win = 0
    for value in pnl:
        current_loss = current_loss + 1 if value < 0 else 0
        current_win = current_win + 1 if value > 0 else 0
        longest_loss = max(longest_loss, current_loss)
        longest_win = max(longest_win, current_win)
    result = {
        "trades": float(len(valid)),
        "unknown_trades": float(trades.status.eq("unresolved").sum()),
        "net_profit": float(pnl.sum()),
        "return_pct": float(pnl.sum() / capital),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "daily_volatility": float(sd * np.sqrt(252)),
        "profit_factor": gain / loss if loss > 0 else np.nan,
        "net_win_rate": float(pnl.gt(0).mean()),
        "target_hit_rate": float(valid.status.eq("target").mean()),
        "daily_win_rate": float(daily.gt(0).mean()),
        "expectancy_dollars": float(pnl.mean()),
        "average_win": float(winners.mean()),
        "average_loss": float(losers.mean()),
        "payoff_ratio": float(winners.mean() / -losers.mean()) if loss > 0 else np.nan,
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
        "net_pnl_std": float(pnl.std(ddof=1)),
        "max_daily_drawdown_dollars": float(dd_dollars.max()),
        "max_daily_drawdown_pct": float(dd_pct.max()),
        "annualized_return": float(daily_cagr) if not insolvent else np.nan,
        "calmar_daily": float(daily_cagr / dd_pct.max()) if dd_pct.max() > 0 else np.nan,
        "session_coverage": float(valid.anchor.nunique() / len(sessions)),
        "sessions": float(len(sessions)),
        "average_hold_minutes": float(valid.holding_minutes.mean()),
        "mean_target_points": float(valid.target_points.mean()),
        "mean_mae_points": float(valid.mae_upper_points.mean())
        if "mae_upper_points" in valid
        else np.nan,
        "mean_full_window_mfe_points": float(valid.full_window_mfe_points.mean())
        if "full_window_mfe_points" in valid
        else np.nan,
        "longest_losing_streak": float(longest_loss),
        "longest_winning_streak": float(longest_win),
        "insolvent_reference_account": float(insolvent),
    }
    return result, curve
