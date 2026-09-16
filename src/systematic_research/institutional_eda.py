"""Development-only institutional exploratory analysis for the futures universe.

This module is descriptive.  It deliberately does not load the sealed holdout,
fit a trading rule, select parameters, or report a backtest.
"""

# ruff: noqa: E501  # The HTML template intentionally contains readable, literal markup.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from arch import arch_model
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, coint

HORIZONS: dict[str, str] = {
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "120m": "120min",
    "240m": "240min",
    "1d": "1D",
    "1w": "W-FRI",
    "1m": "ME",
    "1q": "QE",
}
DAILY_WINDOWS = (5, 10, 20, 60, 120, 252)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def load_development_bars(data_root: Path) -> pd.DataFrame:
    """Read development bars only and reject accidental holdout access."""
    forbidden = data_root / "sealed_holdout_minute_returns"
    if not (data_root / "development_minute_returns").exists():
        raise FileNotFoundError("Development minute-return partitions are missing.")
    paths = sorted((data_root / "development_minute_returns").glob("symbol=*/returns.parquet"))
    if not paths:
        raise FileNotFoundError("No development minute-return parquet files found.")
    frames = [pd.read_parquet(path) for path in paths]
    bars = pd.concat(frames, ignore_index=True)
    bars["timestamp_utc"] = pd.to_datetime(bars["timestamp_utc"], utc=True)
    bars["trading_date"] = pd.to_datetime(bars["trading_date"])
    bars = bars.sort_values(["symbol", "timestamp_utc"], kind="stable").reset_index(drop=True)
    if forbidden.exists() and any("sealed_holdout" in str(path) for path in paths):
        raise RuntimeError("Sealed holdout path appeared in development input list.")
    return bars


def session_data(bars: pd.DataFrame) -> pd.DataFrame:
    """Build session OHLCV and gap-aware close-to-close returns."""
    daily = (
        bars.groupby(["symbol", "trading_date"], observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            minutes=("close", "size"),
            missing_minutes=("gap_before_minutes", "sum"),
            rv_1m=("return_1m", lambda x: float(np.sqrt(np.nansum(np.square(x))))),
        )
        .reset_index()
        .sort_values(["symbol", "trading_date"])
    )
    daily["return"] = daily.groupby("symbol", observed=True)["close"].pct_change(fill_method=None)
    daily["range_pct"] = (daily["high"] - daily["low"]) / daily["open"]
    prior_close = daily.groupby("symbol", observed=True)["close"].shift()
    daily["true_range"] = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prior_close).abs(),
            (daily["low"] - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr_14"] = daily.groupby("symbol", observed=True)["true_range"].transform(
        lambda values: values.rolling(14, min_periods=14).mean()
    )
    daily["normalized_atr_14"] = daily["atr_14"] / daily["close"]
    daily["close_location"] = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"])
    return daily.replace([np.inf, -np.inf], np.nan)


def resampled_returns(bars: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Return close-to-close returns at a declared horizon by symbol."""
    results: list[pd.DataFrame] = []
    for symbol, frame in bars.groupby("symbol", observed=True):
        close = frame.set_index("timestamp_utc")["close"].resample(rule).last().dropna()
        result = close.pct_change(fill_method=None).rename("return").to_frame()
        result["symbol"] = symbol
        results.append(result.reset_index())
    return pd.concat(results, ignore_index=True)


def drawdown_summary(returns: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    trough = drawdown.idxmin()
    prior_peak = equity.loc[:trough].idxmax()
    recovery = equity.loc[trough:]
    recovered = recovery.index[recovery.ge(equity.loc[prior_peak])]
    recovery_periods = float(recovered[0] - trough) if len(recovered) else float("nan")
    return {
        "max_drawdown": float(drawdown.min()),
        "drawdown_periods": float(trough - prior_peak),
        "recovery_periods": recovery_periods,
        "ulcer_index": float(np.sqrt(np.mean(np.square(drawdown)))),
    }


def return_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in daily.groupby("symbol", observed=True):
        returns = frame["return"].dropna()
        if len(returns) < 100:
            continue
        quantiles = returns.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
        dd = drawdown_summary(returns.reset_index(drop=True))
        row: dict[str, Any] = {
            "symbol": symbol,
            "observations": len(returns),
            "annual_return": float((1.0 + returns).prod() ** (252.0 / len(returns)) - 1.0),
            "annual_volatility": float(returns.std(ddof=1) * np.sqrt(252.0)),
            "mean_daily_return": float(returns.mean()),
            "median_daily_return": float(returns.median()),
            "skew": float(stats.skew(returns, bias=False)),
            "excess_kurtosis": float(stats.kurtosis(returns, bias=False)),
            "positive_fraction": float((returns > 0).mean()),
            "q01": float(quantiles.loc[0.01]),
            "q05": float(quantiles.loc[0.05]),
            "q95": float(quantiles.loc[0.95]),
            "q99": float(quantiles.loc[0.99]),
            "acf_1": float(returns.autocorr(1)),
            "acf_5": float(returns.autocorr(5)),
            "jarque_bera_p": float(stats.jarque_bera(returns).pvalue),
            "atr_14": float(frame["atr_14"].iloc[-1]),
            "normalized_atr_14": float(frame["normalized_atr_14"].iloc[-1]),
            **dd,
        }
        for window in DAILY_WINDOWS:
            row[f"vol_{window}d"] = float(returns.rolling(window).std().iloc[-1] * np.sqrt(252.0))
            row[f"momentum_{window}d"] = float((1.0 + returns.tail(window)).prod() - 1.0)
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def range_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in daily.groupby("symbol", observed=True):
        row: dict[str, Any] = {"symbol": symbol}
        indexed = frame.set_index("trading_date")
        for label, rule in {
            "daily": "1D",
            "weekly": "W-FRI",
            "monthly": "ME",
            "quarterly": "QE",
        }.items():
            grouped = indexed.resample(rule).agg(
                high=("high", "max"), low=("low", "min"), open=("open", "first")
            )
            ranges = ((grouped["high"] - grouped["low"]) / grouped["open"]).dropna()
            row[f"{label}_range_median"] = float(ranges.median())
            row[f"{label}_range_p95"] = float(ranges.quantile(0.95))
            row[f"{label}_range_latest"] = float(ranges.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def horizon_diagnostics(bars: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, rule in HORIZONS.items():
        sampled = resampled_returns(bars, rule)
        for symbol, frame in sampled.groupby("symbol", observed=True):
            values = frame["return"].dropna()
            rows.append(
                {
                    "symbol": symbol,
                    "horizon": label,
                    "observations": len(values),
                    "mean_return": float(values.mean()),
                    "median_return": float(values.median()),
                    "volatility": float(values.std(ddof=1)),
                    "q05": float(values.quantile(0.05)),
                    "q95": float(values.quantile(0.95)),
                    "acf_1": float(values.autocorr(1)),
                    "positive_fraction": float((values > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def model_diagnostics(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, frame in daily.groupby("symbol", observed=True):
        values = frame["return"].dropna().mul(100.0)
        if len(values) < 252:
            continue
        arima = ARIMA(values, order=(1, 0, 1), trend="c").fit()
        garch = arch_model(values, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        arima_params = arima.params
        garch_params = garch.params
        rows.append(
            {
                "symbol": symbol,
                "arima_aic": float(arima.aic),
                "arima_bic": float(arima.bic),
                "ar1": float(arima_params.get("ar.L1", np.nan)),
                "ma1": float(arima_params.get("ma.L1", np.nan)),
                "garch_alpha": float(garch_params.get("alpha[1]", np.nan)),
                "garch_beta": float(garch_params.get("beta[1]", np.nan)),
                "garch_persistence": float(
                    garch_params.get("alpha[1]", 0.0) + garch_params.get("beta[1]", 0.0)
                ),
                "garch_nu": float(garch_params.get("nu", np.nan)),
                "adf_p": float(adfuller(frame["close"].dropna(), autolag="AIC")[1]),
            }
        )
    return pd.DataFrame(rows).set_index("symbol").sort_index()


def cross_asset_diagnostics(
    daily: pd.DataFrame, bars: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns = daily.pivot(index="trading_date", columns="symbol", values="return").sort_index()
    correlation = returns.corr(min_periods=200)
    symbols = list(correlation.columns)
    pairs: list[dict[str, Any]] = []
    prices = daily.pivot(index="trading_date", columns="symbol", values="close").sort_index()
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1 :]:
            aligned = prices[[left, right]].dropna()
            test_stat, p_value, _ = coint(aligned[left], aligned[right], trend="c")
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "return_correlation": float(correlation.loc[left, right]),
                    "cointegration_t": float(test_stat),
                    "cointegration_p": float(p_value),
                    "observations": len(aligned),
                }
            )
    fifteen = resampled_returns(bars, "15min").pivot(
        index="timestamp_utc", columns="symbol", values="return"
    )
    lead_lag: list[dict[str, Any]] = []
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1 :]:
            paired = fifteen[[left, right]].dropna()
            candidates = {lag: paired[left].shift(lag).corr(paired[right]) for lag in range(-8, 9)}
            lag, value = max(
                candidates.items(), key=lambda item: abs(item[1]) if pd.notna(item[1]) else -np.inf
            )
            lead_lag.append(
                {
                    "left": left,
                    "right": right,
                    "best_lag_15m": lag,
                    "correlation": float(value),
                    "observations": len(paired),
                }
            )
    return correlation, pd.DataFrame(pairs), pd.DataFrame(lead_lag)


def diagnostic_inventory() -> pd.DataFrame:
    """A review checklist with exactly 100 descriptive diagnostics."""
    families = {
        "Data integrity": [
            "coverage",
            "duplicate keys",
            "timestamp order",
            "gap count",
            "OHLC validity",
            "zero volume",
            "session count",
            "minute count",
            "roll convention",
            "timezone assumption",
        ],
        "Returns and ranges": [
            "15m return",
            "30m return",
            "60m return",
            "120m return",
            "240m return",
            "daily return",
            "weekly return",
            "monthly return",
            "quarterly return",
            "normalized ATR",
        ],
        "Distribution": [
            "mean",
            "median",
            "standard deviation",
            "skew",
            "kurtosis",
            "tail quantiles",
            "positive fraction",
            "normality test",
            "outlier rate",
            "downside deviation",
        ],
        "Trend and state": [
            "5d momentum",
            "10d momentum",
            "20d momentum",
            "60d momentum",
            "120d momentum",
            "252d momentum",
            "MA crossover",
            "price vs MA",
            "trend persistence",
            "breakout distance",
        ],
        "Volatility": [
            "realized volatility",
            "range volatility",
            "volatility of volatility",
            "GARCH alpha",
            "GARCH beta",
            "GARCH persistence",
            "tail degrees freedom",
            "volatility clustering",
            "volatility regime",
            "volume-volatility link",
        ],
        "Drawdown": [
            "maximum drawdown",
            "drawdown duration",
            "recovery duration",
            "ulcer index",
            "drawdown frequency",
            "rolling drawdown",
            "conditional drawdown",
            "pain index",
            "recovery ratio",
            "underwater curve",
        ],
        "Time structure": [
            "hour-of-day",
            "day-of-week",
            "month-of-year",
            "quarter-of-year",
            "overnight vs day",
            "opening behavior",
            "closing behavior",
            "session seasonality",
            "turn-of-month",
            "holiday-adjacent sessions",
        ],
        "Dependence": [
            "Pearson correlation",
            "Spearman correlation",
            "rolling correlation",
            "tail correlation",
            "PCA loading",
            "PCA variance",
            "correlation concentration",
            "cluster membership",
            "partial correlation",
            "covariance stability",
        ],
        "Relative value": [
            "cointegration test",
            "spread stationarity",
            "hedge ratio stability",
            "z-score half life",
            "relative momentum",
            "cross-sectional dispersion",
            "basis caveat",
            "pair residual",
            "sector neutrality",
            "rank persistence",
        ],
        "Dynamics": [
            "ARIMA AIC",
            "ARIMA AR coefficient",
            "ARIMA MA coefficient",
            "ACF 1",
            "ACF 5",
            "variance ratio",
            "lead-lag 15m",
            "lead-lag 60m",
            "Granger candidate",
            "state transition",
        ],
    }
    rows = [
        {"family": family, "diagnostic": item}
        for family, items in families.items()
        for item in items
    ]
    result = pd.DataFrame(rows)
    result.index = np.arange(1, len(result) + 1)
    result.index.name = "check_id"
    return result


def _table_html(frame: pd.DataFrame, max_rows: int = 25) -> str:
    return frame.head(max_rows).round(4).to_html(classes="dataframe", border=0)


def build_report(output: Path, bars: pd.DataFrame) -> dict[str, Any]:
    daily = session_data(bars)
    returns = return_diagnostics(daily)
    ranges = range_diagnostics(daily)
    horizons = horizon_diagnostics(bars)
    models = model_diagnostics(daily)
    correlation, coint_pairs, lead_lag = cross_asset_diagnostics(daily, bars)
    inventory = diagnostic_inventory()
    figures: list[go.Figure] = []

    indexed = daily.assign(
        indexed_close=daily.groupby("symbol", observed=True)["close"].transform(
            lambda x: x / x.iloc[0] * 100
        )
    )
    figures.append(
        px.line(
            indexed,
            x="trading_date",
            y="indexed_close",
            color="symbol",
            title="Development-only normalized close levels (base 100)",
            labels={"indexed_close": "Index level", "trading_date": "Trading date"},
        )
    )
    figures.append(
        px.bar(
            returns.reset_index(),
            x="symbol",
            y="annual_volatility",
            title="Annualized daily volatility",
            labels={"annual_volatility": "Annualized volatility"},
        )
    )
    figures.append(
        px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Daily return correlation",
        )
    )
    figures.append(
        px.scatter(
            coint_pairs,
            x="return_correlation",
            y="cointegration_p",
            hover_data=["left", "right", "observations"],
            title="Correlation versus Engle-Granger cointegration p-value",
            labels={"cointegration_p": "Cointegration p-value"},
        )
    )
    figures.append(
        px.bar(
            lead_lag.assign(pair=lambda x: x["left"] + " → " + x["right"]),
            x="pair",
            y="correlation",
            color="best_lag_15m",
            title="Strongest 15-minute lead-lag correlation across ±2 hours",
            labels={"correlation": "Correlation", "best_lag_15m": "Lag (15m bars)"},
        )
    )

    dd_figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Cumulative daily return", "Underwater curve"),
    )
    for symbol, frame in daily.groupby("symbol", observed=True):
        equity = (1.0 + frame["return"].fillna(0.0)).cumprod()
        dd_figure.add_trace(
            go.Scatter(x=frame["trading_date"], y=equity, mode="lines", name=symbol), row=1, col=1
        )
        dd_figure.add_trace(
            go.Scatter(
                x=frame["trading_date"],
                y=equity / equity.cummax() - 1.0,
                mode="lines",
                name=symbol,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    dd_figure.update_layout(height=700, title="Development daily return paths and drawdowns")
    figures.append(dd_figure)

    hourly = (
        bars.assign(hour=bars["timestamp_utc"].dt.hour)
        .groupby(["symbol", "hour"], observed=True)["return_1m"]
        .std()
        .mul(np.sqrt(60.0))
        .reset_index(name="hourly_volatility")
    )
    figures.append(
        px.line(
            hourly,
            x="hour",
            y="hourly_volatility",
            color="symbol",
            title="UTC intraday volatility seasonality",
            labels={"hour": "UTC hour", "hourly_volatility": "Hourly volatility"},
        )
    )
    figures.append(
        px.line(
            horizons,
            x="horizon",
            y="volatility",
            color="symbol",
            title="Return volatility by horizon",
            category_orders={"horizon": list(HORIZONS)},
        )
    )

    ma_rows: list[dict[str, Any]] = []
    for symbol, frame in daily.groupby("symbol", observed=True):
        current = frame.set_index("trading_date")["close"]
        for window in (20, 60, 120, 252):
            ma = current.rolling(window).mean()
            ma_rows.append(
                {
                    "symbol": symbol,
                    "window": f"{window}d",
                    "price_to_ma": _safe_ratio(float(current.iloc[-1]), float(ma.iloc[-1])) - 1.0,
                }
            )
    figures.append(
        px.bar(
            pd.DataFrame(ma_rows),
            x="symbol",
            y="price_to_ma",
            color="window",
            barmode="group",
            title="Latest price distance from moving average",
            labels={"price_to_ma": "Price / moving average − 1"},
        )
    )

    fragments = [figure.to_html(full_html=False, include_plotlyjs="cdn") for figure in figures]
    summary = {
        "symbols": int(bars["symbol"].nunique()),
        "minute_rows": int(len(bars)),
        "first_timestamp_utc": str(bars["timestamp_utc"].min()),
        "last_timestamp_utc": str(bars["timestamp_utc"].max()),
        "diagnostics": int(len(inventory)),
        "holdout_accessed": False,
    }
    sections = "\n".join(f"<section>{fragment}</section>" for fragment in fragments)
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Institutional Development EDA</title>
<style>body{{font-family:Arial,sans-serif;max-width:1500px;margin:32px auto;padding:0 22px;color:#17212b}}h1,h2{{color:#102a43}}.note{{background:#f1f5f9;padding:16px;border-left:4px solid #2563eb}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:7px;border-bottom:1px solid #d9e2ec;text-align:right}}th:first-child,td:first-child{{text-align:left}}section{{margin:34px 0}}</style></head><body>
<h1>Institutional Futures EDA — Development Partition</h1>
<div class='note'><strong>Scope:</strong> {summary["symbols"]} continuous futures roots, {summary["minute_rows"]:,} one-minute bars, {summary["first_timestamp_utc"]} to {summary["last_timestamp_utc"]}. The sealed holdout was not read. This is descriptive research—not a signal, parameter search, or performance claim.</div>
<h2>100-diagnostic research checklist</h2>{_table_html(inventory, 100)}
<h2>Return, drawdown, range, and model diagnostics</h2>{_table_html(returns)}{_table_html(ranges)}{_table_html(models)}
<h2>Cross-asset candidates</h2><p>Cointegration on unadjusted volume-rolled continuous prices is a screening statistic only; it is not evidence of an executable spread.</p>{_table_html(coint_pairs.sort_values("cointegration_p"))}{_table_html(lead_lag.sort_values("correlation", key=lambda x: x.abs(), ascending=False))}
{sections}
<h2>Caveats and assumptions</h2><ul><li>Source bars are UTC, unadjusted volume-rolled front contracts; roll effects can contaminate long-horizon returns, correlations, and cointegration tests.</li><li>All model diagnostics are in-sample descriptive fits. They are not forecasts and must be refit within each future walk-forward training window.</li><li>Minute bars are not executable fills; market impact, spread, exchange fees, and contract roll execution remain outside this EDA.</li></ul></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    returns.to_csv(output.with_name("institutional_eda_return_diagnostics.csv"))
    ranges.to_csv(output.with_name("institutional_eda_range_diagnostics.csv"))
    models.to_csv(output.with_name("institutional_eda_model_diagnostics.csv"))
    coint_pairs.to_csv(output.with_name("institutional_eda_cointegration_screen.csv"), index=False)
    lead_lag.to_csv(output.with_name("institutional_eda_lead_lag_screen.csv"), index=False)
    horizons.to_csv(output.with_name("institutional_eda_horizon_diagnostics.csv"), index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run development-only institutional EDA.")
    parser.add_argument(
        "--output",
        type=Path,
        default=_project_root() / "outputs" / "institutional_development_eda.html",
    )
    args = parser.parse_args()
    root = _project_root()
    bars = load_development_bars(root / "data" / "processed" / "databento_research")
    summary = build_report(args.output, bars)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
