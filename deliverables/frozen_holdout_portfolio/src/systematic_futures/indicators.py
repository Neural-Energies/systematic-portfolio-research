from __future__ import annotations

import numpy as np
import pandas as pd


def ema(values: pd.Series, period: int) -> pd.Series:
    return values.ewm(span=period, adjust=False, min_periods=1).mean()


def tema(values: pd.Series, period: int) -> pd.Series:
    first = ema(values, period)
    second = ema(first, period)
    third = ema(second, period)
    return 3.0 * first - 3.0 * second + third


def macd_signal(close: pd.Series, fast: int, slow: int, smooth: int) -> pd.Series:
    return ema(ema(close, fast) - ema(close, slow), smooth)


def true_range(bars: pd.DataFrame) -> pd.Series:
    previous = bars["close"].shift(1)
    return pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous).abs(),
            (bars["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(bars: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR, retained for protective-order calculation and audit."""
    return true_range(bars).ewm(alpha=1.0 / period, adjust=False, min_periods=1).mean()


def bollinger(values: pd.Series, period: int, deviation: float) -> tuple[pd.Series, pd.Series]:
    middle = values.rolling(period, min_periods=1).mean()
    width = values.rolling(period, min_periods=1).std(ddof=0).fillna(0.0) * deviation
    return middle + width, middle - width


def keltner(bars: pd.DataFrame, period: int, deviation: float) -> tuple[pd.Series, pd.Series]:
    """Frozen research definition: SMA(typical) +/- multiplier*SMA(high-low)."""
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    middle = typical.rolling(period, min_periods=1).mean()
    offset = (bars["high"] - bars["low"]).rolling(period, min_periods=1).mean() * deviation
    return middle + offset, middle - offset


def parabolic_sar(high: pd.Series, low: pd.Series, step: float, maximum: float) -> pd.Series:
    """Causal Parabolic SAR using only observations at or before each output."""
    if len(high) == 0:
        return pd.Series(dtype=float, index=high.index)
    h = high.to_numpy(float)
    low_values = low.to_numpy(float)
    out = np.full(len(h), np.nan)
    long = False
    sar = h[0]
    extreme_high = h[0]
    extreme_low = low_values[0]
    previous_high = extreme_high
    previous_low = extreme_low
    acceleration = step
    for i in range(len(h)):
        if i == 0:
            out[i] = sar
            continue
        extreme_high = max(extreme_high, h[i])
        extreme_low = min(extreme_low, low_values[i])
        if long:
            if low_values[i] <= sar:
                long = False
                close_sar = extreme_high
                extreme_high, extreme_low = h[i], low_values[i]
                acceleration = step
                sar = max(close_sar + acceleration * (extreme_low - close_sar), h[i], h[i - 1])
            else:
                if extreme_high > previous_high:
                    acceleration = min(acceleration + step, maximum)
                sar = min(
                    sar + acceleration * (extreme_high - sar),
                    low_values[i],
                    low_values[i - 1],
                )
        else:
            if h[i] >= sar:
                long = True
                close_sar = extreme_low
                extreme_high, extreme_low = h[i], low_values[i]
                acceleration = step
                sar = min(
                    close_sar + acceleration * (extreme_high - close_sar),
                    low_values[i],
                    low_values[i - 1],
                )
            else:
                if extreme_low < previous_low:
                    acceleration = min(acceleration + step, maximum)
                sar = max(sar + acceleration * (extreme_low - sar), h[i], h[i - 1])
        previous_high, previous_low = extreme_high, extreme_low
        out[i] = sar
    return pd.Series(out, index=high.index, name="parabolic_sar")
