"""Point-in-time market-structure and tape-pacing features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MARKET_STRUCTURE_FAMILIES = {
    "range_atr_and_estimators": [
        "true_range_fraction",
        "atr_5s_fraction",
        "atr_14s_fraction",
        "atr_30s_fraction",
        "parkinson_volatility_5s",
        "parkinson_volatility_20s",
        "parkinson_volatility_30s",
        "garman_klass_volatility_5s",
        "garman_klass_volatility_20s",
        "garman_klass_volatility_30s",
        "overnight_gap_fraction",
        "range_percentile_20s",
        "range_percentile_60s",
        "range_percentile_120s",
        "range_quartile_60s",
    ],
    "ohlcv_history": [
        "open_to_previous_close",
        "high_to_previous_close",
        "low_to_previous_close",
        "close_return_lag_1s",
        "close_return_lag_2s",
        "close_return_lag_5s",
        "close_return_lag_10s",
        "close_return_lag_20s",
        "close_return_lag_30s",
        "volume_change_1s",
        "volume_change_5s",
        "volume_change_20s",
        "volume_percentile_20s",
        "volume_percentile_60s",
        "volume_percentile_120s",
        "volume_quartile_60s",
        "close_percentile_20s",
        "close_percentile_60s",
        "close_percentile_120s",
    ],
    "moving_average_structure": [
        "price_to_sma_5s",
        "price_to_sma_30s",
        "price_to_sma_120s",
        "sma_gap_5_20s",
        "sma_gap_20_60s",
        "sma_gap_30_120s",
        "price_to_ema_10s",
        "price_to_ema_30s",
        "price_to_ema_60s",
        "price_to_ema_120s",
    ],
    "linear_regression_structure": [
        "trend_slope_30s",
        "trend_slope_120s",
        "trend_r_squared_20s",
        "trend_r_squared_60s",
        "trend_r_squared_120s",
        "trend_residual_zscore_20s",
        "trend_residual_zscore_60s",
        "trend_slope_acceleration_20s",
    ],
}
MARKET_STRUCTURE_COLUMNS = [
    column for family in MARKET_STRUCTURE_FAMILIES.values() for column in family
]

TAPE_PACE_COLUMNS = [
    "volume_per_observed_minute",
    "absolute_return_per_observed_minute",
    "zero_return_fraction",
    "positive_return_fraction",
    "return_sign_imbalance",
    "intraday_return_q10",
    "intraday_return_q25",
    "intraday_return_median",
    "intraday_return_q75",
    "intraday_return_q90",
    "intraday_return_iqr",
    "intraday_tail_imbalance",
]


def _percentile_of_latest(values: NDArray[np.float64]) -> float:
    return float(np.mean(values <= values[-1]))


def _normalized_slope(values: NDArray[np.float64]) -> float:
    mean = float(np.mean(values))
    if mean == 0.0:
        return np.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0] / mean)


def _regression_r_squared(values: NDArray[np.float64]) -> float:
    x = np.arange(len(values), dtype=float)
    fitted = np.polyval(np.polyfit(x, values, 1), x)
    total = float(np.square(values - np.mean(values)).sum())
    residual = float(np.square(values - fitted).sum())
    return 1.0 - residual / total if total > 0.0 else np.nan


def _latest_residual_zscore(values: NDArray[np.float64]) -> float:
    x = np.arange(len(values), dtype=float)
    residuals = values - np.polyval(np.polyfit(x, values, 1), x)
    scale = float(np.std(residuals, ddof=1))
    return float(residuals[-1] / scale) if scale > 0.0 else np.nan


def build_market_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create scale-aware daily structure features without future observations."""
    frame = panel.sort_values(["symbol", "trading_date"], ignore_index=True).copy()
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        group = group.copy()
        open_price = group["open"].astype(float)
        high = group["high"].astype(float)
        low = group["low"].astype(float)
        close = group["close"].astype(float)
        volume = group["volume"].astype(float)
        previous_close = close.shift(1)
        raw_range = high - low
        true_range = pd.concat(
            [raw_range, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
        ).max(axis=1)
        group["true_range_fraction"] = true_range.div(previous_close.where(previous_close.ne(0)))
        for window in (5, 14, 30):
            group[f"atr_{window}s_fraction"] = (
                true_range.rolling(window, min_periods=window).mean().div(close.where(close.ne(0)))
            )

        log_high_low = pd.Series(np.log(high.div(low.where(low.gt(0)))), index=group.index)
        log_close_open = pd.Series(
            np.log(close.div(open_price.where(open_price.gt(0)))), index=group.index
        )
        parkinson_variance = log_high_low.pow(2).div(4.0 * np.log(2.0))
        gk_variance = 0.5 * log_high_low.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_close_open.pow(2)
        for window in (5, 20, 30):
            group[f"parkinson_volatility_{window}s"] = np.sqrt(
                parkinson_variance.rolling(window, min_periods=window).mean().clip(lower=0.0)
            )
            group[f"garman_klass_volatility_{window}s"] = np.sqrt(
                gk_variance.rolling(window, min_periods=window).mean().clip(lower=0.0)
            )
        group["overnight_gap_fraction"] = open_price.div(previous_close).sub(1.0)

        range_fraction = raw_range.div(close.where(close.ne(0)))
        for window in (20, 60, 120):
            group[f"range_percentile_{window}s"] = range_fraction.rolling(
                window, min_periods=window
            ).apply(_percentile_of_latest, raw=True)
            group[f"volume_percentile_{window}s"] = volume.rolling(
                window, min_periods=window
            ).apply(_percentile_of_latest, raw=True)
            group[f"close_percentile_{window}s"] = close.rolling(window, min_periods=window).apply(
                _percentile_of_latest, raw=True
            )
        group["range_quartile_60s"] = np.ceil(group["range_percentile_60s"] * 4.0)
        group["volume_quartile_60s"] = np.ceil(group["volume_percentile_60s"] * 4.0)

        group["open_to_previous_close"] = open_price.div(previous_close).sub(1.0)
        group["high_to_previous_close"] = high.div(previous_close).sub(1.0)
        group["low_to_previous_close"] = low.div(previous_close).sub(1.0)
        close_returns = close.pct_change(fill_method=None)
        for lag in (1, 2, 5, 10, 20, 30):
            group[f"close_return_lag_{lag}s"] = close_returns.shift(lag)
        for lag in (1, 5, 20):
            group[f"volume_change_{lag}s"] = volume.pct_change(lag, fill_method=None)

        simple_averages = {
            window: close.rolling(window, min_periods=window).mean()
            for window in (5, 20, 30, 60, 120)
        }
        for window in (5, 30, 120):
            group[f"price_to_sma_{window}s"] = close.div(simple_averages[window]).sub(1.0)
        group["sma_gap_5_20s"] = simple_averages[5].div(simple_averages[20]).sub(1.0)
        group["sma_gap_20_60s"] = simple_averages[20].div(simple_averages[60]).sub(1.0)
        group["sma_gap_30_120s"] = simple_averages[30].div(simple_averages[120]).sub(1.0)
        for window in (10, 30, 60, 120):
            ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
            group[f"price_to_ema_{window}s"] = close.div(ema).sub(1.0)

        slopes: dict[int, pd.Series] = {}
        for window in (20, 30, 60, 120):
            slopes[window] = close.rolling(window, min_periods=window).apply(
                _normalized_slope, raw=True
            )
        group["trend_slope_30s"] = slopes[30]
        group["trend_slope_120s"] = slopes[120]
        for window in (20, 60, 120):
            group[f"trend_r_squared_{window}s"] = close.rolling(window, min_periods=window).apply(
                _regression_r_squared, raw=True
            )
        for window in (20, 60):
            group[f"trend_residual_zscore_{window}s"] = close.rolling(
                window, min_periods=window
            ).apply(_latest_residual_zscore, raw=True)
        group["trend_slope_acceleration_20s"] = slopes[20].diff(5)
        pieces.append(group[["symbol", "trading_date"] + MARKET_STRUCTURE_COLUMNS])
    return pd.concat(pieces, ignore_index=True)


def build_tape_pace_features(minute_returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize observable one-minute activity; no tick-count claim is made."""
    frame = minute_returns.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    rows: list[dict[str, object]] = []
    for (symbol, trading_date), group in frame.groupby(
        ["symbol", "trading_date"], sort=True, observed=True
    ):
        returns = group["return_1m"].dropna().to_numpy(dtype=float)
        observations = len(group)
        quantiles = (
            np.quantile(returns, [0.1, 0.25, 0.5, 0.75, 0.9]) if len(returns) else [np.nan] * 5
        )
        negative_tail = abs(float(quantiles[0]))
        positive_tail = abs(float(quantiles[4]))
        rows.append(
            {
                "symbol": symbol,
                "trading_date": trading_date,
                "volume_per_observed_minute": float(group["volume"].sum() / observations),
                "absolute_return_per_observed_minute": float(np.abs(returns).sum() / observations),
                "zero_return_fraction": float(np.mean(returns == 0.0)),
                "positive_return_fraction": float(np.mean(returns > 0.0)),
                "return_sign_imbalance": float(np.mean(np.sign(returns))),
                "intraday_return_q10": quantiles[0],
                "intraday_return_q25": quantiles[1],
                "intraday_return_median": quantiles[2],
                "intraday_return_q75": quantiles[3],
                "intraday_return_q90": quantiles[4],
                "intraday_return_iqr": quantiles[3] - quantiles[1],
                "intraday_tail_imbalance": positive_tail - negative_tail,
            }
        )
    return pd.DataFrame(rows)
