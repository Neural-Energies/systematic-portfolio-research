"""Point-in-time feature construction using development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from systematic_research.market_structure import (
    MARKET_STRUCTURE_COLUMNS,
    MARKET_STRUCTURE_FAMILIES,
    TAPE_PACE_COLUMNS,
    build_market_structure_features,
    build_tape_pace_features,
)
from systematic_research.temporal_validation import load_development_panel

FEATURE_FAMILIES = {
    "momentum_reversal": [
        "log_return_1s",
        "momentum_2s",
        "momentum_5s",
        "momentum_10s",
        "momentum_20s",
        "momentum_60s",
        "momentum_120s",
        "standardized_return_5s",
        "standardized_return_20s",
        "standardized_return_60s",
        "reversal_1s",
        "return_zscore_20s",
    ],
    "trend": [
        "price_to_sma_10s",
        "price_to_sma_20s",
        "price_to_sma_60s",
        "ema_gap_5_20s",
        "ema_gap_10_60s",
        "trend_slope_20s",
        "trend_slope_60s",
        "trend_efficiency_20s",
        "trend_efficiency_60s",
    ],
    "volatility_liquidity_regime": [
        "realized_volatility_5s",
        "realized_volatility_20s",
        "return_volatility_5s",
        "return_volatility_20s",
        "return_volatility_60s",
        "volatility_ratio_5_20s",
        "volatility_ratio_20_60s",
        "downside_volatility_20s",
        "upside_downside_volatility_ratio_20s",
        "drawdown_20s",
        "drawdown_60s",
        "range_fraction",
        "range_zscore_20s",
        "close_location",
        "log_volume",
        "volume_zscore_20s",
        "return_autocorrelation_20s",
        "return_autocorrelation_60s",
    ],
    "cross_market_lead_lag": [
        "peer_return_lag1s",
        "peer_momentum_5s_lag1s",
        "peer_breadth_lag1s",
        "leadlag_correlation_20s",
        "leadlag_correlation_60s",
        "leadlag_beta_60s",
    ],
}
FEATURE_COLUMNS = [column for family in FEATURE_FAMILIES.values() for column in family]

REALIZED_MEASURE_COLUMNS = [
    "realized_variance",
    "realized_volatility",
    "downside_semivariance",
    "upside_semivariance",
    "bipower_variation",
    "jump_variation",
    "realized_quarticity",
]


def _normalized_slope(values: NDArray[np.float64]) -> float:
    mean = float(np.mean(values))
    if mean == 0.0:
        return np.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values, 1)[0] / mean)


def _trend_efficiency(values: NDArray[np.float64]) -> float:
    path = float(np.abs(np.diff(values)).sum())
    return float(abs(values[-1] - values[0]) / path) if path > 0.0 else np.nan


def _autocorrelation(values: NDArray[np.float64]) -> float:
    return float(pd.Series(values).autocorr(lag=1))


def build_daily_realized_measures(minute_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate gap-aware minute returns into non-parametric daily risk measures."""
    required = {"symbol", "trading_date", "timestamp_utc", "return_1m"}
    missing = required.difference(minute_returns.columns)
    if missing:
        raise ValueError(f"Development minute returns missing columns: {sorted(missing)}")
    if minute_returns.duplicated(["symbol", "timestamp_utc"]).any():
        raise ValueError("Development minute symbol/timestamp keys must be unique")

    frame = minute_returns.sort_values(["symbol", "trading_date", "timestamp_utc"]).copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame["log_return_1m"] = np.log1p(frame["return_1m"].astype(float))
    rows: list[dict[str, object]] = []
    for (symbol, trading_date), group in frame.groupby(
        ["symbol", "trading_date"], sort=True, observed=True
    ):
        return_series = group["log_return_1m"]
        returns = return_series.dropna().to_numpy(dtype=float)
        squared = np.square(returns)
        realized_variance = float(squared.sum())
        adjacent_products = return_series.abs().mul(return_series.shift().abs()).dropna()
        bipower = float(np.pi / 2.0 * adjacent_products.sum()) if len(adjacent_products) else np.nan
        rows.append(
            {
                "symbol": symbol,
                "trading_date": trading_date,
                "realized_variance": realized_variance,
                "realized_volatility": float(np.sqrt(realized_variance)),
                "downside_semivariance": float(squared[returns < 0.0].sum()),
                "upside_semivariance": float(squared[returns >= 0.0].sum()),
                "bipower_variation": bipower,
                "jump_variation": max(realized_variance - bipower, 0.0)
                if np.isfinite(bipower)
                else np.nan,
                "realized_quarticity": float(len(returns) / 3.0 * np.power(returns, 4).sum())
                if len(returns) > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_session_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build features known at each session close without targets or future values."""
    required = {
        "symbol",
        "trading_date",
        "session_close_utc",
        "high",
        "low",
        "close",
        "volume",
        "close_to_close_return",
        "realized_volatility_1m",
    }
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Development panel missing columns: {sorted(missing)}")
    if panel.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("Development symbol/trading_date keys must be unique")

    frame = panel.sort_values(["symbol", "trading_date"], ignore_index=True).copy()
    return_matrix = frame.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    peer_return = return_matrix.apply(
        lambda column: return_matrix.drop(columns=column.name).mean(axis=1), axis=0
    )
    peer_momentum_source = (1.0 + return_matrix).rolling(5, min_periods=5).apply(np.prod) - 1.0
    peer_momentum = peer_momentum_source.apply(
        lambda column: peer_momentum_source.drop(columns=column.name).mean(axis=1), axis=0
    )
    peer_breadth = return_matrix.apply(
        lambda column: return_matrix.drop(columns=column.name).gt(0.0).mean(axis=1), axis=0
    )
    pieces: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=True, observed=True):
        group = group.copy()
        returns = pd.Series(
            np.log1p(group["close_to_close_return"].to_numpy(dtype=float)),
            index=group.index,
            dtype=float,
        )
        close = group["close"].astype(float)
        realized = group["realized_volatility_1m"].astype(float)
        volume = group["volume"].astype(float)
        price_range = group["high"].astype(float) - group["low"].astype(float)

        group["log_return_1s"] = returns
        for window in (2, 5, 10, 20, 60, 120):
            cumulative_return = returns.rolling(window, min_periods=window).sum()
            group[f"momentum_{window}s"] = np.expm1(cumulative_return)
        for window in (5, 20, 60):
            cumulative_return = returns.rolling(window, min_periods=window).sum()
            trailing_risk = returns.rolling(window, min_periods=window).std(ddof=1)
            group[f"standardized_return_{window}s"] = cumulative_return.div(
                trailing_risk.mul(np.sqrt(float(window))).where(trailing_risk.ne(0.0))
            )
        group["reversal_1s"] = -returns
        return_mean_20 = returns.rolling(20, min_periods=20).mean()
        return_std_20 = returns.rolling(20, min_periods=20).std(ddof=1)
        group["return_zscore_20s"] = returns.sub(return_mean_20).div(
            return_std_20.where(return_std_20.ne(0.0))
        )

        for window in (10, 20, 60):
            average = close.rolling(window, min_periods=window).mean()
            group[f"price_to_sma_{window}s"] = close.div(average).sub(1.0)
        ema_5 = close.ewm(span=5, adjust=False, min_periods=5).mean()
        ema_10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
        ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema_60 = close.ewm(span=60, adjust=False, min_periods=60).mean()
        group["ema_gap_5_20s"] = ema_5.div(ema_20).sub(1.0)
        group["ema_gap_10_60s"] = ema_10.div(ema_60).sub(1.0)
        for window in (20, 60):
            group[f"trend_slope_{window}s"] = close.rolling(window, min_periods=window).apply(
                _normalized_slope, raw=True
            )
            group[f"trend_efficiency_{window}s"] = close.rolling(window, min_periods=window).apply(
                _trend_efficiency, raw=True
            )

        group["realized_volatility_5s"] = realized.rolling(5, min_periods=5).mean()
        group["realized_volatility_20s"] = realized.rolling(20, min_periods=20).mean()
        group["return_volatility_5s"] = returns.rolling(5, min_periods=5).std(ddof=1)
        group["return_volatility_20s"] = returns.rolling(20, min_periods=20).std(ddof=1)
        group["return_volatility_60s"] = returns.rolling(60, min_periods=60).std(ddof=1)
        group["volatility_ratio_5_20s"] = group["return_volatility_5s"].div(
            group["return_volatility_20s"].where(group["return_volatility_20s"].ne(0.0))
        )
        group["volatility_ratio_20_60s"] = group["return_volatility_20s"].div(
            group["return_volatility_60s"].where(group["return_volatility_60s"].ne(0.0))
        )
        downside = returns.clip(upper=0.0)
        upside = returns.clip(lower=0.0)
        group["downside_volatility_20s"] = downside.rolling(20, min_periods=20).std(ddof=1)
        upside_volatility = upside.rolling(20, min_periods=20).std(ddof=1)
        group["upside_downside_volatility_ratio_20s"] = upside_volatility.div(
            group["downside_volatility_20s"].where(group["downside_volatility_20s"].ne(0.0))
        )
        for window in (20, 60):
            rolling_high = close.rolling(window, min_periods=window).max()
            group[f"drawdown_{window}s"] = close.div(rolling_high).sub(1.0)
        group["range_fraction"] = price_range.div(close.where(close.ne(0.0)))
        range_mean = group["range_fraction"].rolling(20, min_periods=20).mean()
        range_std = group["range_fraction"].rolling(20, min_periods=20).std(ddof=1)
        group["range_zscore_20s"] = (
            group["range_fraction"].sub(range_mean).div(range_std.where(range_std.ne(0.0)))
        )
        group["close_location"] = close.sub(group["low"].astype(float)).div(
            price_range.where(price_range.ne(0.0))
        )
        group["log_volume"] = np.log1p(volume.clip(lower=0.0))
        volume_mean = group["log_volume"].rolling(20, min_periods=20).mean()
        volume_std = group["log_volume"].rolling(20, min_periods=20).std(ddof=1)
        group["volume_zscore_20s"] = (
            group["log_volume"].sub(volume_mean).div(volume_std.where(volume_std.ne(0.0)))
        )
        for window in (20, 60):
            group[f"return_autocorrelation_{window}s"] = returns.rolling(
                window, min_periods=window
            ).apply(_autocorrelation, raw=True)

        symbol = str(group["symbol"].iloc[0])
        dates = group["trading_date"]
        group["peer_return_lag1s"] = dates.map(peer_return[symbol].shift(1))
        group["peer_momentum_5s_lag1s"] = dates.map(peer_momentum[symbol].shift(1))
        group["peer_breadth_lag1s"] = dates.map(peer_breadth[symbol].shift(1))
        for window in (20, 60):
            group[f"leadlag_correlation_{window}s"] = returns.rolling(
                window, min_periods=window
            ).corr(group["peer_return_lag1s"])
        peer_variance = group["peer_return_lag1s"].rolling(60, min_periods=60).var(ddof=1)
        group["leadlag_beta_60s"] = (
            returns.rolling(60, min_periods=60)
            .cov(group["peer_return_lag1s"])
            .div(peer_variance.where(peer_variance.ne(0.0)))
        )
        pieces.append(group)

    features = pd.concat(pieces, ignore_index=True)
    structure = build_market_structure_features(frame)
    features = features.merge(
        structure, on=["symbol", "trading_date"], how="left", validate="one_to_one"
    )
    features = features.rename(columns={"session_close_utc": "feature_available_at_utc"})
    identifiers = ["symbol", "trading_date", "feature_available_at_utc"]
    all_columns = FEATURE_COLUMNS + MARKET_STRUCTURE_COLUMNS
    result = features[identifiers + all_columns].replace([np.inf, -np.inf], np.nan)
    result["feature_set_complete"] = result[all_columns].notna().all(axis=1)
    return result.sort_values(["trading_date", "symbol"], ignore_index=True)


def write_development_features(
    research_root: Path = Path("data/processed/research"),
    output_root: Path = Path("data/processed/features"),
    catalog_root: Path = Path("data/catalog"),
) -> pd.DataFrame:
    """Persist only features derived from the guarded development accessor."""
    panel = load_development_panel(research_root)
    boundaries = json.loads((catalog_root / "temporal_split.json").read_text(encoding="utf-8"))
    dates = pd.to_datetime(panel["trading_date"])
    if dates.max() > pd.Timestamp(boundaries["development_end"]):
        raise ValueError("Development accessor returned observations beyond development_end")
    features = build_session_features(panel)
    minute_paths = sorted(
        (research_root / "development_minute_returns").glob("symbol=*/returns.parquet")
    )
    if not minute_paths:
        raise FileNotFoundError("No development minute-return partitions were found")
    minute_returns = pd.concat((pd.read_parquet(path) for path in minute_paths), ignore_index=True)
    realized_measures = build_daily_realized_measures(minute_returns)
    tape_pace = build_tape_pace_features(minute_returns)
    features = features.merge(
        realized_measures,
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    features = features.merge(
        tape_pace,
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    complete_columns = (
        FEATURE_COLUMNS + MARKET_STRUCTURE_COLUMNS + TAPE_PACE_COLUMNS + REALIZED_MEASURE_COLUMNS
    )
    features["feature_set_complete"] = features[complete_columns].notna().all(axis=1)
    output_root.mkdir(parents=True, exist_ok=True)
    features.to_parquet(
        output_root / "development_session_features.parquet", index=False, compression="zstd"
    )
    metadata = {
        "source": "development_session_panel.parquet",
        "development_end": boundaries["development_end"],
        "decision_timing": "Features are available only after feature_available_at_utc.",
        "contains_targets": False,
        "feature_columns": (
            FEATURE_COLUMNS
            + MARKET_STRUCTURE_COLUMNS
            + TAPE_PACE_COLUMNS
            + REALIZED_MEASURE_COLUMNS
        ),
        "feature_families": {
            **FEATURE_FAMILIES,
            **MARKET_STRUCTURE_FAMILIES,
            "minute_bar_pace_and_distribution": TAPE_PACE_COLUMNS,
            "intraday_realized_risk": REALIZED_MEASURE_COLUMNS,
        },
        "rows": len(features),
        "complete_rows": int(features["feature_set_complete"].sum()),
    }
    (catalog_root / "development_features.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    unavailable = {
        "implied_volatility": {
            "status": "unavailable",
            "reason": (
                "Current inputs contain futures OHLCV bars but no options chain or IV series."
            ),
            "required_data": [
                "point-in-time option quotes",
                "expiry and strike",
                "underlying price",
                "rates and contract metadata",
            ],
            "planned_features": [
                "implied_minus_realized_volatility",
                "implied_to_realized_volatility_ratio",
                "volatility_term_structure",
                "skew and risk_reversal",
            ],
        },
        "true_tick_pace": {
            "status": "unavailable",
            "reason": "One-minute OHLCV has no trade count, bid/ask, or message timestamps.",
            "implemented_proxy": "minute_bar_pace_and_distribution",
        },
    }
    (catalog_root / "unavailable_features.json").write_text(
        json.dumps(unavailable, indent=2), encoding="utf-8"
    )
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe development features.")
    parser.add_argument("--research", type=Path, default=Path("data/processed/research"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/features"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog"))
    arguments = parser.parse_args()
    features = write_development_features(arguments.research, arguments.output, arguments.catalog)
    print(
        f"Built {len(features):,} development feature rows; "
        f"{int(features['feature_set_complete'].sum()):,} have the full required history."
    )


if __name__ == "__main__":
    main()
