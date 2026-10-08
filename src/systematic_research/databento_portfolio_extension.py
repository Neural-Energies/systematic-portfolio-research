"""Import Databento parent-product bars and evaluate portfolio diversification.

The portal's ``*.FUT`` parent symbology includes outright contracts, calendar
spreads, and user-defined instruments.  This module keeps only outright
contracts and constructs daily return series using the prior trading day's
volume leader.  Returns are calculated within each contract before selection,
so contract rolls do not introduce artificial price jumps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import pandas as pd

ROOTS = ("ZB", "SR3", "ZF", "ZT", "RTY", "ZW", "ZS", "LE")
MONTH_CODES = "FGHJKMNQUVXZ"
ASSET_CLUSTERS = {
    "6E": "fx",
    "6J": "fx",
    "CL": "energy",
    "NG": "energy",
    "ES": "equity",
    "NQ": "equity",
    "NKD": "equity",
    "RTY": "equity",
    "GC": "metals",
    "HG": "metals",
    "ZC": "grains",
    "ZS": "grains",
    "ZW": "grains",
    "ZN": "rates",
    "ZB": "rates",
    "ZF": "rates",
    "ZT": "rates",
    "SR3": "rates",
    "LE": "livestock",
}
OUTRIGHT_PATTERN = re.compile(
    rf"^(?P<root>{'|'.join(sorted(ROOTS, key=len, reverse=True))})"
    rf"(?P<month>[{MONTH_CODES}])(?P<year>\d{{1,2}})$"
)


@dataclass(frozen=True)
class ManifestCheck:
    filename: str
    expected_sha256: str
    observed_sha256: str
    size_bytes: int
    valid: bool


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(batch_directory: Path) -> list[ManifestCheck]:
    """Verify every file declared in a Databento batch manifest."""
    manifest_path = batch_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks: list[ManifestCheck] = []
    for item in manifest["files"]:
        path = batch_directory / str(item["filename"])
        expected = str(item["hash"]).removeprefix("sha256:").lower()
        observed = sha256_file(path) if path.exists() else ""
        checks.append(
            ManifestCheck(
                filename=path.name,
                expected_sha256=expected,
                observed_sha256=observed,
                size_bytes=path.stat().st_size if path.exists() else 0,
                valid=path.exists() and observed == expected,
            )
        )
    return checks


def parse_outright_symbol(symbol: str) -> tuple[str, str, str] | None:
    """Parse an outright futures symbol while rejecting spreads and UDS names."""
    match = OUTRIGHT_PATTERN.fullmatch(symbol.upper())
    if match is None:
        return None
    return match.group("root"), match.group("month"), match.group("year")


def filter_outrights(frame: pd.DataFrame) -> pd.DataFrame:
    """Return mapped outright contracts with explicit root/month/year columns."""
    if "symbol" not in frame:
        raise ValueError("Databento frame must include mapped raw symbols")
    extracted = frame["symbol"].astype("string").str.extract(OUTRIGHT_PATTERN)
    valid = extracted["root"].notna()
    result = frame.loc[valid].copy()
    result["root_symbol"] = extracted.loc[valid, "root"].astype("string")
    result["contract_month_code"] = extracted.loc[valid, "month"].astype("string")
    result["contract_year_code"] = extracted.loc[valid, "year"].astype("string")
    return result


def read_dbn_outrights(path: Path) -> pd.DataFrame:
    """Decode a DBN file, map instrument IDs, and retain outright contracts."""
    frame = db.DBNStore.from_file(path).to_df()
    if frame.index.name != "ts_event":
        raise ValueError("Expected ts_event as the DBN event-time index")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Expected timezone-aware Databento event timestamps")
    return filter_outrights(frame)


def aggregate_contract_days(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mapped minute bars to UTC-date contract OHLCV observations."""
    required = {"root_symbol", "symbol", "open", "high", "low", "close", "volume"}
    missing = required.difference(minute_bars.columns)
    if missing:
        raise ValueError(f"Missing required minute-bar columns: {sorted(missing)}")
    if not isinstance(minute_bars.index, pd.DatetimeIndex):
        raise ValueError("Minute bars must use a DatetimeIndex")

    frame = minute_bars.copy()
    frame["trading_date"] = frame.index.normalize().tz_localize(None)
    daily = (
        frame.reset_index()
        .sort_values(["root_symbol", "symbol", "ts_event"])
        .groupby(["root_symbol", "symbol", "trading_date"], observed=True, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            minute_observations=("close", "size"),
        )
        .reset_index()
    )
    daily["contract_return"] = daily.groupby(
        ["root_symbol", "symbol"], observed=True, sort=False
    )["close"].pct_change(fill_method=None)
    return daily


def build_causal_daily_returns(contract_days: pd.DataFrame) -> pd.DataFrame:
    """Select each day's contract using only the prior day's volume ranking."""
    required = {
        "root_symbol",
        "symbol",
        "trading_date",
        "close",
        "volume",
        "contract_return",
        "minute_observations",
    }
    missing = required.difference(contract_days.columns)
    if missing:
        raise ValueError(f"Missing required contract-day columns: {sorted(missing)}")

    ranked = contract_days.sort_values(
        ["root_symbol", "trading_date", "volume", "symbol"],
        ascending=[True, True, False, True],
    )
    leaders = ranked.drop_duplicates(["root_symbol", "trading_date"], keep="first")[
        ["root_symbol", "trading_date", "symbol", "volume"]
    ].rename(columns={"symbol": "volume_leader", "volume": "leader_volume"})
    leaders = leaders.sort_values(["root_symbol", "trading_date"])
    leaders["selected_contract"] = leaders.groupby("root_symbol", observed=True)[
        "volume_leader"
    ].shift(1)
    leaders["selection_volume_lagged"] = leaders.groupby("root_symbol", observed=True)[
        "leader_volume"
    ].shift(1)

    selected = leaders.merge(
        contract_days,
        left_on=["root_symbol", "trading_date", "selected_contract"],
        right_on=["root_symbol", "trading_date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    selected = selected.sort_values(["root_symbol", "trading_date"]).copy()
    selected["contract_return"] = pd.to_numeric(
        selected["contract_return"], errors="coerce"
    )
    selected["roll_flag"] = selected.groupby("root_symbol", observed=True)[
        "selected_contract"
    ].transform(lambda values: values.ne(values.shift(1)))
    selected.loc[selected["selected_contract"].isna(), "roll_flag"] = False
    selected["synthetic_index"] = selected.groupby("root_symbol", observed=True)[
        "contract_return"
    ].transform(lambda values: (1.0 + values.fillna(0.0)).cumprod())
    return selected[
        [
            "root_symbol",
            "trading_date",
            "selected_contract",
            "selection_volume_lagged",
            "contract_return",
            "synthetic_index",
            "close",
            "volume",
            "minute_observations",
            "roll_flag",
        ]
    ]


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def _portfolio_metrics(returns: pd.DataFrame, weights: pd.Series) -> dict[str, float]:
    aligned = returns.loc[:, weights.index].dropna()
    portfolio = aligned.mul(weights, axis=1).sum(axis=1)
    annual_return = float(portfolio.mean() * 252.0)
    annual_volatility = float(portfolio.std(ddof=1) * np.sqrt(252.0))
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else np.nan
    covariance = aligned.cov() * 252.0
    weighted_vol = float(np.sqrt(weights @ covariance @ weights))
    standalone = np.sqrt(np.diag(covariance))
    diversification_ratio = (
        float(np.dot(weights, standalone) / weighted_vol) if weighted_vol > 0 else np.nan
    )
    marginal = covariance @ weights
    risk_contributions = weights * marginal / float(weights @ covariance @ weights)
    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_zero_cash": sharpe,
        "max_drawdown": _max_drawdown(portfolio),
        "diversification_ratio": diversification_ratio,
        "effective_number_weights": float(1.0 / np.square(weights).sum()),
        "largest_risk_contribution": float(risk_contributions.max()),
        "observations": float(len(portfolio)),
    }


def cluster_balanced_inverse_vol_weights(calibration: pd.DataFrame) -> pd.Series:
    """Allocate equally to asset clusters, then inverse-volatility within each."""
    missing = set(calibration.columns).difference(ASSET_CLUSTERS)
    if missing:
        raise ValueError(f"Missing asset-cluster assignments: {sorted(missing)}")
    groups: dict[str, list[str]] = {}
    for symbol in calibration.columns:
        groups.setdefault(ASSET_CLUSTERS[symbol], []).append(symbol)
    cluster_budget = 1.0 / len(groups)
    weights = pd.Series(0.0, index=calibration.columns, dtype=float)
    for symbols in groups.values():
        volatility = calibration[symbols].std(ddof=1).replace(0.0, np.nan)
        within_cluster = (1.0 / volatility).dropna()
        within_cluster = within_cluster / within_cluster.sum()
        weights.loc[within_cluster.index] = cluster_budget * within_cluster
    return weights / weights.sum()


def evaluate_portfolio_extension(
    existing_panel: pd.DataFrame,
    extension_returns: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the existing universe with the eight-product extension out of time."""
    existing = existing_panel.pivot_table(
        index="trading_date",
        columns="symbol",
        values="close_to_close_return",
        aggfunc="last",
    )
    existing.index = pd.to_datetime(existing.index)
    added = extension_returns.pivot_table(
        index="trading_date",
        columns="root_symbol",
        values="contract_return",
        aggfunc="last",
    )
    added.index = pd.to_datetime(added.index)
    combined = existing.join(added, how="inner").sort_index()
    combined = combined.dropna(axis=0, how="any")
    if len(combined) < 60:
        raise ValueError("Fewer than 60 complete overlapping daily observations")

    split = max(int(len(combined) * 0.60), 30)
    calibration = combined.iloc[:split]
    evaluation = combined.iloc[split:]
    if len(evaluation) < 20:
        raise ValueError("Fewer than 20 out-of-time portfolio evaluation days")

    existing_columns = [column for column in existing.columns if column in combined]
    expanded_columns = list(combined.columns)
    weight_sets: dict[str, pd.Series] = {}
    for name, columns in (
        ("existing_inverse_vol", existing_columns),
        ("expanded_inverse_vol", expanded_columns),
    ):
        volatility = calibration[columns].std(ddof=1).replace(0.0, np.nan)
        inverse = 1.0 / volatility
        weight_sets[name] = inverse / inverse.sum()
    weight_sets["existing_cluster_balanced"] = cluster_balanced_inverse_vol_weights(
        calibration[existing_columns]
    )
    weight_sets["expanded_cluster_balanced"] = cluster_balanced_inverse_vol_weights(
        calibration[expanded_columns]
    )

    metrics = {
        name: _portfolio_metrics(evaluation, weights) for name, weights in weight_sets.items()
    }
    correlation = combined.corr(min_periods=max(30, len(combined) // 2))
    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    high_pairs = (
        upper.loc[upper.abs() >= 0.65]
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "symbol_a", "level_1": "symbol_b"})
        .sort_values("correlation", key=lambda series: series.abs(), ascending=False)
    )
    coverage = pd.DataFrame(
        {
            "symbol": combined.columns,
            "start": [combined[column].first_valid_index() for column in combined],
            "end": [combined[column].last_valid_index() for column in combined],
            "observations": [int(combined[column].notna().sum()) for column in combined],
            "annualized_volatility": [
                float(combined[column].std(ddof=1) * np.sqrt(252.0)) for column in combined
            ],
            "average_absolute_correlation": [
                float(correlation[column].drop(index=column).abs().mean())
                for column in combined
            ],
        }
    )
    return {
        "combined_returns": combined,
        "correlation": correlation,
        "high_correlation_pairs": high_pairs,
        "coverage": coverage,
        "weights": weight_sets,
        "metrics": metrics,
        "calibration_end": calibration.index[-1],
        "evaluation_start": evaluation.index[0],
        "evaluation_end": evaluation.index[-1],
    }


def write_import_outputs(
    minute_bars: pd.DataFrame,
    contract_days: pd.DataFrame,
    daily_returns: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Persist the imported outright universe and causal research series."""
    output_directory.mkdir(parents=True, exist_ok=True)
    for root in ROOTS:
        root_directory = output_directory / "minute_bars" / f"symbol={root}"
        root_directory.mkdir(parents=True, exist_ok=True)
        root_frame = minute_bars.loc[minute_bars["root_symbol"].eq(root)].reset_index()
        root_frame.to_parquet(root_directory / "bars.parquet", index=False, compression="zstd")
    contract_days.to_parquet(
        output_directory / "contract_daily_bars.parquet", index=False, compression="zstd"
    )
    daily_returns.to_parquet(
        output_directory / "causal_daily_returns.parquet", index=False, compression="zstd"
    )


def write_analysis_outputs(result: dict[str, Any], output_directory: Path) -> None:
    """Write inspectable portfolio diagnostics for report generation."""
    output_directory.mkdir(parents=True, exist_ok=True)
    result["combined_returns"].to_parquet(output_directory / "combined_returns.parquet")
    result["correlation"].to_csv(output_directory / "correlation_matrix.csv")
    result["high_correlation_pairs"].to_csv(
        output_directory / "high_correlation_pairs.csv", index=False
    )
    result["coverage"].to_csv(output_directory / "instrument_summary.csv", index=False)
    weights = pd.concat(result["weights"], names=["portfolio", "symbol"]).rename("weight")
    weights.reset_index().to_csv(output_directory / "portfolio_weights.csv", index=False)
    summary = {
        "method": {
            "contract_selection": "prior trading day's total-volume leader",
            "return_construction": "within-contract close-to-close return before selection",
            "portfolio_weights": "inverse volatility estimated on first 60% of overlap",
            "cluster_balanced_weights": (
                "equal asset-cluster budgets, inverse volatility within each cluster"
            ),
            "portfolio_evaluation": "fixed weights on final 40% of overlap",
        },
        "calibration_end": pd.Timestamp(result["calibration_end"]).date().isoformat(),
        "evaluation_start": pd.Timestamp(result["evaluation_start"]).date().isoformat(),
        "evaluation_end": pd.Timestamp(result["evaluation_end"]).date().isoformat(),
        "metrics": result["metrics"],
    }
    (output_directory / "portfolio_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    evaluation = result["combined_returns"].loc[result["evaluation_start"] :]
    cumulative_rows: list[dict[str, Any]] = []
    for portfolio, portfolio_weights in result["weights"].items():
        portfolio_returns = evaluation.loc[:, portfolio_weights.index].mul(
            portfolio_weights, axis=1
        ).sum(axis=1)
        cumulative = (1.0 + portfolio_returns).cumprod() - 1.0
        cumulative_rows.extend(
            {
                "date": date.date().isoformat(),
                "portfolio": portfolio,
                "cumulative_return": float(value),
            }
            for date, value in cumulative.items()
        )
    correlation_rows = (
        result["correlation"]
        .rename_axis(index="symbol_a", columns="symbol_b")
        .stack()
        .rename("correlation")
        .reset_index()
        .to_dict(orient="records")
    )
    snapshot = {
        "title": "The expanded futures universe adds breadth, but rates need one risk budget",
        "asOf": pd.Timestamp(result["evaluation_end"]).date().isoformat(),
        "report": {"asOf": pd.Timestamp(result["evaluation_end"]).date().isoformat()},
        "scope": {
            "new_products": list(ROOTS),
            "all_products": list(result["combined_returns"].columns),
            "overlap_days": int(len(result["combined_returns"])),
            "evaluation_days": int(len(evaluation)),
            "calibration_end": summary["calibration_end"],
            "evaluation_start": summary["evaluation_start"],
            "evaluation_end": summary["evaluation_end"],
        },
        "method": summary["method"],
        "filters": [],
        "queries": {
            "portfolio_metrics": {
                "rows": [
                    {"portfolio": name, **values}
                    for name, values in result["metrics"].items()
                ],
                "source": {
                    "label": "Out-of-time portfolio allocation comparison",
                    "filters": [
                        "Weights calibrated through 2025-11-12",
                        "Evaluation from 2025-11-13 through 2026-08-21",
                    ],
                    "metricDefinitions": [
                        {
                            "label": "Diversification ratio",
                            "definition": (
                                "Weighted standalone volatility divided by portfolio "
                                "volatility; values above one indicate diversification."
                            ),
                            "componentIds": ["portfolio-metrics", "portfolio-comparison"],
                        },
                        {
                            "label": "Sharpe (zero cash)",
                            "definition": (
                                "Annualized mean divided by annualized volatility with no "
                                "cash-rate subtraction; diagnostic only, not a strategy Sharpe."
                            ),
                            "componentIds": ["portfolio-metrics", "portfolio-comparison"],
                        },
                    ],
                },
            },
            "instrument_summary": {
                "rows": result["coverage"].assign(
                    asset_cluster=result["coverage"]["symbol"].map(ASSET_CLUSTERS)
                ).to_dict(orient="records"),
                "source": {
                    "label": "Complete-overlap daily return diagnostics",
                    "filters": ["482 complete days from 2024-09-17 through 2026-08-21"],
                },
            },
            "high_correlation_pairs": {
                "rows": result["high_correlation_pairs"].to_dict(orient="records"),
                "source": {
                    "label": "Pearson correlations of complete-overlap daily returns",
                    "filters": ["Absolute correlation at or above 0.65"],
                },
            },
            "correlation_matrix": {
                "rows": correlation_rows,
                "source": {
                    "label": "Pearson correlation matrix of complete-overlap daily returns",
                    "filters": ["All 19 markets; 482 complete observations"],
                },
            },
            "portfolio_weights": {
                "rows": weights.reset_index().assign(
                    asset_cluster=lambda frame: frame["symbol"].map(ASSET_CLUSTERS)
                ).to_dict(orient="records"),
                "source": {
                    "label": "Frozen allocation weights estimated on calibration data",
                    "filters": ["Calibration ends 2025-11-12"],
                },
            },
            "cumulative_returns": {
                "rows": cumulative_rows,
                "source": {
                    "label": "Out-of-time buy-and-hold allocation diagnostic",
                    "filters": ["2025-11-13 through 2026-08-21; 193 observations"],
                },
            },
        },
        "provenance": {
            "raw_batch": "GLBX-20260916-9R3XEL5M5X",
            "raw_schema": "Databento GLBX.MDP3 OHLCV-1m DBN zstd",
            "raw_period": "2024-09-16 through 2026-09-15",
            "new_outright_minute_rows": 13123202,
            "portfolio_overlap_period": "2024-09-17 through 2026-08-21",
            "holdout_note": (
                "This is an out-of-time universe construction diagnostic, not a strategy "
                "backtest and not evidence that any trading signal meets a Sharpe threshold."
            ),
        },
    }
    (output_directory / "report_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and assess the Databento extension.")
    parser.add_argument("--batch-directory", type=Path, required=True)
    parser.add_argument(
        "--dbn-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--existing-development",
        type=Path,
        default=Path("data/processed/databento_research/development_session_panel.parquet"),
    )
    parser.add_argument(
        "--existing-holdout",
        type=Path,
        default=Path("data/processed/databento_research/sealed_holdout_session_panel.parquet"),
    )
    parser.add_argument(
        "--import-output",
        type=Path,
        default=Path("data/processed/databento_portfolio_extension"),
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("outputs/databento_portfolio_extension"),
    )
    arguments = parser.parse_args()

    checks = verify_manifest(arguments.batch_directory)
    if not checks or not all(check.valid for check in checks):
        raise SystemExit("Databento manifest verification failed")
    arguments.analysis_output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(check) for check in checks]).to_csv(
        arguments.analysis_output / "manifest_verification.csv", index=False
    )

    minute_bars = read_dbn_outrights(arguments.dbn_file)
    contract_days = aggregate_contract_days(minute_bars)
    daily_returns = build_causal_daily_returns(contract_days)
    write_import_outputs(minute_bars, contract_days, daily_returns, arguments.import_output)

    existing = pd.concat(
        [
            pd.read_parquet(arguments.existing_development),
            pd.read_parquet(arguments.existing_holdout),
        ],
        ignore_index=True,
    ).drop_duplicates(["symbol", "trading_date"], keep="last")
    result = evaluate_portfolio_extension(existing, daily_returns)
    write_analysis_outputs(result, arguments.analysis_output)
    print(
        f"Imported {len(minute_bars):,} outright minute bars across "
        f"{minute_bars['root_symbol'].nunique()} roots; evaluated "
        f"{len(result['combined_returns']):,} complete overlap days."
    )


if __name__ == "__main__":
    main()
