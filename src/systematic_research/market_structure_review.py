"""Descriptive, no-signal market-structure review for futures, FX, and crypto."""

# ruff: noqa: E501  # Literal HTML report template.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def _daily_from_futures(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data["trading_date"] = pd.to_datetime(data["trading_date"])
    result = data.rename(columns={"trading_date": "date", "close_to_close_return": "return"})[
        ["symbol", "date", "open", "high", "low", "close", "volume", "return"]
    ].copy()
    result["asset_class"] = "futures"
    return result


def _daily_from_raw(paths: list[Path], asset_class: str) -> pd.DataFrame:
    frames = []
    for path in paths:
        bars = pd.read_parquet(path)
        liquidity_column = "volume" if "volume" in bars.columns else "tick_count"
        bars["date"] = (
            pd.to_datetime(bars["timestamp_utc"], utc=True).dt.tz_localize(None).dt.normalize()
        )
        daily = (
            bars.groupby(["symbol", "date"], observed=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=(liquidity_column, "sum"),
            )
            .reset_index()
        )
        frames.append(daily)
    result = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    result["return"] = result.groupby("symbol", observed=True)["close"].pct_change(fill_method=None)
    result["asset_class"] = asset_class
    return result


def _metrics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset_class, symbol), group in data.groupby(["asset_class", "symbol"], observed=True):
        returns = group["return"].dropna()
        prior = group["close"].shift()
        tr = pd.concat(
            [
                group["high"] - group["low"],
                (group["high"] - prior).abs(),
                (group["low"] - prior).abs(),
            ],
            axis=1,
        ).max(axis=1)
        equity = (1 + returns).cumprod()
        rows.append(
            {
                "asset_class": asset_class,
                "symbol": symbol,
                "days": len(returns),
                "ann_vol": returns.std() * np.sqrt(252),
                "skew": returns.skew(),
                "kurtosis": returns.kurt(),
                "q01": returns.quantile(0.01),
                "atr14_pct": (tr.rolling(14).mean() / group["close"]).iloc[-1],
                "max_dd": (equity / equity.cummax() - 1).min(),
                "median_volume": group["volume"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["asset_class", "ann_vol"], ascending=[True, False])


def _regime_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset_class, symbol), group in data.groupby(["asset_class", "symbol"], observed=True):
        r = group["return"].dropna()
        vol = r.rolling(20).std().shift(1)
        high = vol > vol.median()
        rows.append(
            {
                "asset_class": asset_class,
                "symbol": symbol,
                "low_vol_mean": r[~high].mean(),
                "high_vol_mean": r[high].mean(),
                "low_vol_ann": r[~high].std() * np.sqrt(252),
                "high_vol_ann": r[high].std() * np.sqrt(252),
                "high_vol_fraction": high.mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create descriptive market-structure review.")
    parser.add_argument("--output", type=Path, default=Path("outputs/market_structure_review.html"))
    args = parser.parse_args()
    futures = _daily_from_futures(
        Path("data/processed/databento_research/development_session_panel.parquet")
    )
    fx = _daily_from_raw(
        sorted(
            Path("data/raw/free_api/oanda_practice").glob(
                "symbol=*/granularity=M15/bars_20230822*.parquet"
            )
        ),
        "fx",
    )
    crypto = _daily_from_raw(
        sorted(
            Path("data/raw/free_api/binance_spot").glob(
                "symbol=*/interval=15m/bars_20230822*.parquet"
            )
        ),
        "crypto",
    )
    data = pd.concat([futures, fx, crypto], ignore_index=True)
    metrics = _metrics(data)
    regimes = _regime_table(data)
    common = data.pivot(index="date", columns="symbol", values="return").sort_index()
    corr = common.corr(min_periods=100)
    aligned = common.dropna(axis=1, thresh=200).dropna(how="any")
    pca = PCA().fit(aligned.fillna(0))
    loadings = pd.DataFrame(
        pca.components_[:3].T, index=aligned.columns, columns=["pc1", "pc2", "pc3"]
    )
    explained = pd.DataFrame(
        {
            "component": ["pc1", "pc2", "pc3"],
            "variance_explained": pca.explained_variance_ratio_[:3],
        }
    )
    horizon = []
    for symbol, group in data.groupby("symbol", observed=True):
        close = group.set_index("date")["close"]
        for name, rule in {"1d": "1D", "1w": "W-FRI", "1m": "ME", "1q": "QE"}.items():
            r = close.resample(rule).last().pct_change().dropna()
            horizon.append(
                {
                    "symbol": symbol,
                    "horizon": name,
                    "observations": len(r),
                    "mean": r.mean(),
                    "vol": r.std(),
                    "q05": r.quantile(0.05),
                    "q95": r.quantile(0.95),
                }
            )
    horizons = pd.DataFrame(horizon)
    figures = [
        px.bar(
            metrics,
            x="symbol",
            y="ann_vol",
            color="asset_class",
            title="Annualized daily volatility by market",
        ),
        px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            title="Daily return correlation — overlapping dates only",
        ),
        px.bar(
            explained,
            x="component",
            y="variance_explained",
            title="Cross-market PCA variance explained",
        ),
    ]
    charts = "".join(figure.to_html(full_html=False, include_plotlyjs="cdn") for figure in figures)
    def table(frame: pd.DataFrame) -> str:
        return frame.round(4).to_html(index=False, border=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"""<html><head><title>Market Structure Review</title><style>body{{font-family:Arial;max-width:1500px;margin:auto;padding:24px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border-bottom:1px solid #ddd;padding:6px;text-align:right}}td:first-child,th:first-child{{text-align:left}}</style></head><body><h1>Market Structure Review — Development Only</h1><p>Descriptive analysis only. No signal selection, strategy backtest, or performance claim. Futures are volume-rolled continuous series; FX uses OANDA midpoint/bid/ask candles; crypto uses Binance spot candles.</p><h2>1. Risk, range and liquidity proxy</h2>{table(metrics)}<h2>2. Correlation by market and regime</h2>{table(regimes)}<h2>3. Factor structure</h2>{table(explained)}{table(loadings.reset_index(names="symbol"))}<h2>4. Return distributions by horizon</h2>{table(horizons)}<h2>5. Visual review</h2>{charts}<h2>6. Research observations</h2><ul><li>Use cluster/factor exposures before forming hypotheses.</li><li>Volatility regimes materially change risk; do not pool them blindly.</li><li>Compare only overlapping data periods and retain source/venue boundaries.</li></ul></body></html>""",
        encoding="utf-8",
    )
    metrics.to_csv(args.output.with_name("market_structure_metrics.csv"), index=False)
    regimes.to_csv(args.output.with_name("market_structure_regimes.csv"), index=False)
    horizons.to_csv(args.output.with_name("market_structure_horizons.csv"), index=False)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
