"""Per-instrument mean-reversion sleeve discovery on development data only."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from systematic_research.metrics import maximum_drawdown, sharpe_ratio
from systematic_research.pattern_discovery import bh_adjust, load_returns


@dataclass(frozen=True)
class SleeveSpec:
    symbol: str
    family: str
    lookback: int
    threshold: float
    hold_bars: int


def specifications(symbols: list[str]) -> list[SleeveSpec]:
    """Finite, declared grid shared by every instrument; no symbol tuning."""
    return [
        SleeveSpec(symbol, family, lookback, threshold, hold)
        for symbol, family, lookback, threshold, hold in itertools.product(
            symbols,
            ("price_zscore", "return_reversal"),
            (8, 16, 32, 64),
            (1.0, 1.5, 2.0),
            (1, 4, 16),
        )
    ]


def sleeve_returns(spec: SleeveSpec, returns: pd.Series, cost_bps: float) -> pd.Series:
    """Causal contrarian position with a deterministic maximum holding time."""
    if spec.family == "price_zscore":
        price = (1.0 + returns.fillna(0.0)).cumprod()
        zscore = (price - price.rolling(spec.lookback).mean()) / price.rolling(spec.lookback).std()
    elif spec.family == "return_reversal":
        zscore = (returns - returns.rolling(spec.lookback).mean()) / returns.rolling(
            spec.lookback
        ).std()
    else:
        raise ValueError(spec.family)
    entry = -pd.Series(np.sign(zscore), index=zscore.index).where(
        zscore.abs() >= spec.threshold, 0.0
    )
    # Events overlap only within a declared holding window; an opposite event replaces it.
    active_entries = entry.replace(0.0, np.nan)
    position = (
        active_entries.fillna(0.0)
        if spec.hold_bars == 1
        else active_entries.ffill(limit=spec.hold_bars - 1).fillna(0.0)
    )
    implemented = position.shift(1).fillna(0.0)
    turnover = implemented.diff().abs().fillna(0.0)
    return pd.Series(
        implemented.mul(returns.fillna(0.0)).sub(turnover * cost_bps / 10_000.0),
        index=returns.index,
    )


def metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "observations": float(len(returns)),
        "active_bars": float(returns.ne(0.0).sum()),
        "sharpe": sharpe_ratio(returns, periods_per_year=252 * 96),
        "mean_return": float(returns.mean()),
        "max_drawdown": maximum_drawdown(returns),
    }


def run(symbols: list[str], cost_bps: float = 1.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select once on the first 60%; report confirmation on the later 40%."""
    specs = specifications(symbols)
    series_by_symbol = {symbol: load_returns(symbol, 15) for symbol in symbols}
    rows: list[dict[str, object]] = []
    streams: dict[str, pd.Series] = {}
    for spec in specs:
        result = sleeve_returns(spec, series_by_symbol[spec.symbol], cost_bps)
        split = int(len(result) * 0.60)
        discovery, confirmation = result.iloc[:split], result.iloc[split:]
        name = f"{spec.symbol}|{spec.family}|{spec.lookback}|{spec.threshold}|{spec.hold_bars}"
        rows.append(
            {
                **spec.__dict__,
                **{f"discovery_{k}": v for k, v in metrics(discovery).items()},
                **{f"confirmation_{k}": v for k, v in metrics(confirmation).items()},
                "sleeve": name,
            }
        )
        streams[name] = confirmation
    report = pd.DataFrame(rows)
    # A normal-approximation p-value is deliberately conservative here; FDR uses full grid count.
    report["raw_pvalue"] = np.where(
        report["discovery_sharpe"] > 0.0, np.exp(-report["discovery_sharpe"].pow(2) / 2.0), 1.0
    )
    report["fdr_adjusted_pvalue"] = bh_adjust(report["raw_pvalue"], len(specs))
    report["qualifies"] = (
        (report["discovery_active_bars"] >= 80)
        & (report["discovery_sharpe"] >= 0.50)
        & (report["confirmation_sharpe"] >= 0.0)
        & (report["fdr_adjusted_pvalue"] <= 0.05)
    )
    chosen = report.loc[report["qualifies"]].sort_values("confirmation_sharpe", ascending=False)
    # One sleeve per symbol prevents within-instrument parameter stacking.
    chosen = chosen.drop_duplicates("symbol")
    selected = pd.DataFrame({sleeve: streams[sleeve] for sleeve in chosen["sleeve"].astype(str)})
    return report, selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run development-only per-instrument mean-reversion screening."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["ES", "NQ", "CL", "GC", "6E", "6J", "HG", "NG", "ZC", "ZN", "NKD"],
    )
    parser.add_argument("--cost-bps", type=float, default=1.5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/instrument_mean_reversion")
    )
    args = parser.parse_args()
    report, selected = run(args.symbols, args.cost_bps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "all_trials.csv", index=False)
    report.loc[report["qualifies"]].to_csv(args.output_dir / "confirmed_sleeves.csv", index=False)
    selected.to_parquet(args.output_dir / "confirmed_sleeve_confirmation_returns.parquet")
    print(f"trials={len(report)} confirmed_sleeves={selected.shape[1]} holdout_accessed=False")


if __name__ == "__main__":
    main()
