from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_VERSION, STRATEGIES
from .indicators import atr, bollinger, keltner, macd_signal, parabolic_sar, tema


def _daily_value(bars: pd.DataFrame, column: str, reducer: str, day_shift: int) -> pd.Series:
    grouped = bars[column].resample("1D").agg(reducer)
    bar_index = pd.DatetimeIndex(bars.index)
    day = bar_index.floor("D") - pd.to_timedelta(day_shift, unit="D")
    return pd.Series(grouped.reindex(day).to_numpy(), index=bars.index)


def _rules(symbol: str, b: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, str]:
    r = b["high"] - b["low"]
    if symbol == "6E":
        upper, lower = bollinger(b["close"], 10, 2.6)
        long = b["close"].shift(1) > upper.shift(1)
        short = (b["close"].shift(1) < lower.shift(1)) & ~long
        psar = parabolic_sar(b["high"], b["low"], 0.02, 0.2)
        smallest = r.rolling(25, min_periods=1).min()
        buy = psar.shift(1) + 2.8 * smallest.shift(3)
        sell = psar.shift(1) - 2.8 * smallest.shift(3)
        reason = "Bollinger close breakout"
    elif symbol == "6J":
        first = macd_signal(b["close"], 24, 17, 9)
        second = macd_signal(b["close"], 3, 26, 9)
        long = (first > first.shift(1)) & (second.shift(1) > second.shift(2))
        short = (first < first.shift(1)) & (second.shift(1) < second.shift(2)) & ~long
        buy = sell = tema(b["close"], 14).shift(1)
        reason = "Dual MACD signal slope"
    elif symbol == "CL":
        u30, l30 = keltner(b, 30, 1.5)
        u20, l20 = keltner(b, 20, 2.0)
        long = (
            (u30.shift(1) < u30.shift(2))
            & (b["open"].shift(1) < l20.shift(1))
            & (b["open"].shift(2) > l20.shift(2))
        )
        short = (
            (l30.shift(1) > l30.shift(2))
            & (b["open"].shift(1) > u20.shift(1))
            & (b["open"].shift(2) < u20.shift(2))
            & ~long
        )
        ue, le = keltner(b, 20, 2.25)
        buy = ue.shift(2) + 0.3 * r.shift(1)
        sell = le.shift(2) - 0.3 * r.shift(1)
        reason = "Keltner reversal"
    elif symbol == "ES":
        upper, _ = bollinger(b["high"], 20, 1.9)
        _, lower = bollinger(b["low"], 20, 1.9)
        long = b["open"] < upper
        short = (b["open"] > lower) & ~long
        buy = sell = _daily_value(b, "open", "first", 3)
        reason = "Bollinger open location"
    elif symbol == "GC":
        upper, lower = keltner(b, 20, 2.5)
        long = upper.shift(1) > upper.shift(2)
        short = (lower.shift(1) < lower.shift(2)) & ~long
        day_open = _daily_value(b, "open", "first", 0)
        atr20 = _wilder_atr(b, 20)
        buy = day_open + 2.1 * atr20.shift(1)
        sell = day_open - 2.1 * atr20.shift(1)
        reason = "Keltner slope"
    elif symbol == "HG":
        upper, lower = keltner(b, 20, 2.0)
        long = lower.shift(1) < lower.shift(2)
        short = (upper.shift(1) > upper.shift(2)) & ~long
        buy = _daily_value(b, "high", "max", 2) + 1.4 * r.shift(1)
        sell = _daily_value(b, "low", "min", 2) - 1.4 * r.shift(1)
        reason = "Keltner slope with prior-day range"
    elif symbol == "NG":
        upper, lower = keltner(b, 20, 1.5)
        long = lower < lower.shift(1)
        short = (upper > upper.shift(1)) & ~long
        psar = parabolic_sar(b["high"], b["low"], 0.01, 0.2)
        biggest = r.rolling(14, min_periods=1).max()
        buy = psar.shift(1) - 0.7 * biggest.shift(1)
        sell = psar.shift(1) + 0.7 * biggest.shift(1)
        reason = "Keltner slope with volatility offset"
    else:
        raise KeyError(f"unsupported symbol: {symbol}")
    return long.fillna(False), short.fillna(False), buy, sell, reason


def _wilder_atr(bars: pd.DataFrame, period: int) -> pd.Series:
    previous = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous).abs(),
            (bars["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=1).mean()


def generate_candidate_orders(
    symbol: str, hourly_bars: pd.DataFrame, quantity: float = 1.0
) -> pd.DataFrame:
    """Generate orders using completed H1 bars only.

    An H1 bar labelled 10:00 contains observations from 10:00 through 10:59;
    the corresponding decision timestamp is 11:00 UTC.
    """
    symbol = symbol.upper()
    cfg = STRATEGIES[symbol]
    long, short, buy, sell, reason = _rules(symbol, hourly_bars)
    side = np.where(long, "BUY", np.where(short, "SELL", ""))
    price = np.where(long, buy, np.where(short, sell, np.nan))
    mask = (side != "") & np.isfinite(price)
    decision_time = pd.DatetimeIndex(hourly_bars.index, name="timestamp_utc") + pd.Timedelta(
        hours=1
    )
    rounded_price = np.round(price[mask] / cfg.tick_size) * cfg.tick_size
    pt_distance = (
        atr(hourly_bars, cfg.profit_target_atr_period).to_numpy()[mask]
        * cfg.profit_target_atr_multiple
    )
    sl_distance = (
        atr(hourly_bars, cfg.stop_loss_atr_period).to_numpy()[mask] * cfg.stop_loss_atr_multiple
    )
    direction = np.where(side[mask] == "BUY", 1.0, -1.0)
    target_price = (
        np.round((rounded_price + direction * pt_distance) / cfg.tick_size) * cfg.tick_size
    )
    stop_loss_price = (
        np.round((rounded_price - direction * sl_distance) / cfg.tick_size) * cfg.tick_size
    )
    result = pd.DataFrame(
        {
            "timestamp_utc": decision_time[mask],
            "model_version": MODEL_VERSION,
            "strategy_id": cfg.strategy_id,
            "symbol": symbol,
            "action": "PLACE_OR_REPLACE",
            "side": side[mask],
            "order_type": "STOP",
            "quantity": quantity,
            "stop_price": rounded_price,
            "valid_bars": cfg.bars_valid,
            "valid_until_utc": decision_time[mask] + pd.to_timedelta(cfg.bars_valid, unit="h"),
            "profit_target_atr_multiple": cfg.profit_target_atr_multiple,
            "profit_target_atr_period": cfg.profit_target_atr_period,
            "profit_target_distance": pt_distance,
            "profit_target_price": target_price,
            "stop_loss_atr_multiple": cfg.stop_loss_atr_multiple,
            "stop_loss_atr_period": cfg.stop_loss_atr_period,
            "stop_loss_distance": sl_distance,
            "stop_loss_price": stop_loss_price,
            "exit_after_bars": cfg.exit_after_bars,
            "friday_exit_time_utc": "20:40:00",
            "reason": reason,
        }
    )
    return result.reset_index(drop=True)
