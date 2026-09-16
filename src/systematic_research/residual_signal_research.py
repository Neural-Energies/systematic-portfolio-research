"""Development-only factor-neutral momentum and reversal diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FX_SIGNS = {
    "EUR_USD": 1.0,
    "GBP_USD": 1.0,
    "AUD_USD": 1.0,
    "USD_JPY": -1.0,
    "USD_CAD": -1.0,
    "USD_CHF": -1.0,
}
LOOKBACKS = {"1h": 4, "4h": 16, "1d": 96}
HOLDS = {"1h": 4, "4h": 16, "1d": 96}


def _returns(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path)[["symbol", "timestamp_utc", "close"]] for path in paths]
    data = pd.concat(frames, ignore_index=True).sort_values(["symbol", "timestamp_utc"])
    data["return"] = data.groupby("symbol", observed=True)["close"].pct_change(fill_method=None)
    return data.pivot(index="timestamp_utc", columns="symbol", values="return").sort_index()


def _residuals(returns: pd.DataFrame, factor: pd.Series, window: int = 1920) -> pd.DataFrame:
    """Use only lagged rolling hedge ratios, avoiding future return information."""
    variance = factor.rolling(window, min_periods=window // 2).var().shift(1)
    result = pd.DataFrame(index=returns.index)
    for asset in returns:
        covariance = returns[asset].rolling(window, min_periods=window // 2).cov(factor).shift(1)
        beta = covariance / variance
        result[asset] = returns[asset] - beta * factor
    return result


def run_tests(residuals: pd.DataFrame, factor: pd.Series, family: str) -> pd.DataFrame:
    volatility = factor.rolling(1920, min_periods=480).std().shift(1)
    regime_cutoff = volatility.expanding(min_periods=1920).median().shift(1)
    high_volatility = volatility.gt(regime_cutoff)
    rows: list[dict[str, float | int | str]] = []
    for asset in residuals:
        series = residuals[asset]
        for lookback_name, lookback in LOOKBACKS.items():
            past = series.rolling(lookback, min_periods=lookback).sum().shift(1)
            for hold_name, hold in HOLDS.items():
                future = series.rolling(hold, min_periods=hold).sum().shift(-hold)
                for style, direction in {"momentum": 1.0, "mean_reversion": -1.0}.items():
                    pnl = direction * np.sign(past) * future
                    for regime, mask in {
                        "all": pnl.notna(),
                        "low_vol": ~high_volatility,
                        "high_vol": high_volatility,
                    }.items():
                        values = pnl[mask].dropna()
                        if len(values) < 250:
                            continue
                        mean = float(values.mean())
                        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
                        rows.append(
                            {
                                "family": family,
                                "asset": asset,
                                "style": style,
                                "lookback": lookback_name,
                                "hold": hold_name,
                                "regime": regime,
                                "observations": len(values),
                                "mean_residual_return": mean,
                                "t_stat_naive": mean / standard_error if standard_error else np.nan,
                                "positive_fraction": float((values > 0).mean()),
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run development-only residual signal diagnostics."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/residual_signal_research.csv"))
    arguments = parser.parse_args()
    root = Path("data/raw/free_api")
    fx_returns = _returns(
        sorted(root.glob("oanda_practice/symbol=*/granularity=M15/bars_20230822*.parquet"))
    )
    factor = fx_returns.mul(pd.Series(FX_SIGNS)).mean(axis=1)
    fx = run_tests(_residuals(fx_returns, factor), factor, "fx_usd_neutral")
    crypto_returns = _returns(
        sorted(root.glob("binance_spot/symbol=*/interval=15m/bars_20230822*.parquet"))
    )
    crypto = run_tests(
        _residuals(crypto_returns[["ETHUSDT"]], crypto_returns["BTCUSDT"]),
        crypto_returns["BTCUSDT"],
        "eth_btc_neutral",
    )
    results = pd.concat([fx, crypto], ignore_index=True).sort_values(
        "t_stat_naive", key=lambda x: x.abs(), ascending=False
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(arguments.output, index=False)
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
