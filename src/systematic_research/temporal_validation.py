"""Leakage-safe temporal holdout and guarded research-data access."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitPolicy:
    holdout_months: int = 12
    minimum_holdout_months: int = 6
    embargo_days: int = 5

    def __post_init__(self) -> None:
        if self.holdout_months < self.minimum_holdout_months:
            raise ValueError("holdout_months cannot be below the required minimum")
        if self.minimum_holdout_months < 6:
            raise ValueError("minimum holdout must be at least six months")
        if self.embargo_days < 0:
            raise ValueError("embargo_days cannot be negative")


@dataclass(frozen=True)
class SplitBoundaries:
    latest_session: str
    development_end: str
    holdout_start: str
    holdout_months: int
    embargo_days: int


def create_temporal_split(
    sessions: pd.DataFrame,
    policy: SplitPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBoundaries]:
    """Create an anchored holdout with an embargo between development and evaluation."""
    frame = sessions.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    latest = pd.Timestamp(frame["trading_date"].max()).normalize()
    holdout_start = (latest - pd.DateOffset(months=policy.holdout_months)).normalize()
    development_end = holdout_start - pd.Timedelta(days=policy.embargo_days + 1)
    development = frame.loc[frame["trading_date"] <= development_end].copy()
    holdout = frame.loc[frame["trading_date"] >= holdout_start].copy()

    rows: list[dict[str, object]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        symbol_development = development.loc[development["symbol"].eq(symbol)]
        symbol_holdout = holdout.loc[holdout["symbol"].eq(symbol)]
        rows.append(
            {
                "symbol": symbol,
                "first_session": group["trading_date"].min().date().isoformat(),
                "last_session": group["trading_date"].max().date().isoformat(),
                "development_sessions": len(symbol_development),
                "holdout_sessions": len(symbol_holdout),
                "eligibility": "development_and_holdout"
                if len(symbol_development) > 0
                else "holdout_only",
            }
        )
    coverage = pd.DataFrame(rows)
    boundaries = SplitBoundaries(
        latest.date().isoformat(),
        development_end.date().isoformat(),
        holdout_start.date().isoformat(),
        policy.holdout_months,
        policy.embargo_days,
    )
    return development, holdout, coverage, boundaries


def write_temporal_split(
    sessions: pd.DataFrame,
    output_root: Path,
    catalog_root: Path,
    policy: SplitPolicy,
) -> SplitBoundaries:
    """Persist physically separated development and sealed holdout datasets."""
    development, holdout, coverage, boundaries = create_temporal_split(sessions, policy)
    output_root.mkdir(parents=True, exist_ok=True)
    development.to_parquet(
        output_root / "development_session_panel.parquet", index=False, compression="zstd"
    )
    holdout.to_parquet(
        output_root / "sealed_holdout_session_panel.parquet", index=False, compression="zstd"
    )
    catalog_root.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(catalog_root / "temporal_split_coverage.csv", index=False)
    (catalog_root / "temporal_split.json").write_text(
        json.dumps(asdict(boundaries), indent=2), encoding="utf-8"
    )

    development_matrix = development.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    development_matrix.corr(min_periods=60).to_csv(
        catalog_root / "development_return_correlations.csv"
    )
    (development_matrix.std(ddof=1) * np.sqrt(252.0)).rename("annualized_volatility").to_csv(
        catalog_root / "development_annualized_volatility.csv"
    )
    for unsafe_name in ("return_correlations.csv", "annualized_volatility.csv"):
        unsafe_path = catalog_root / unsafe_name
        if unsafe_path.exists():
            unsafe_path.unlink()
    return boundaries


def seal_minute_partitions(output_root: Path, boundaries: SplitBoundaries) -> None:
    """Physically separate minute returns and remove combined research copies."""
    combined_root = output_root / "minute_returns"
    development_end = pd.Timestamp(boundaries.development_end)
    holdout_start = pd.Timestamp(boundaries.holdout_start)
    for source in combined_root.glob("symbol=*/returns.parquet"):
        frame = pd.read_parquet(source)
        dates = pd.to_datetime(frame["trading_date"])
        development = frame.loc[dates <= development_end]
        holdout = frame.loc[dates >= holdout_start]
        symbol_directory = source.parent.name
        for name, partition in (
            ("development_minute_returns", development),
            ("sealed_holdout_minute_returns", holdout),
        ):
            destination = output_root / name / symbol_directory
            destination.mkdir(parents=True, exist_ok=True)
            partition.to_parquet(destination / "returns.parquet", index=False, compression="zstd")
        source.unlink()
    combined_sessions = output_root / "session_panel.parquet"
    if combined_sessions.exists():
        combined_sessions.unlink()


def load_development_panel(path: Path = Path("data/processed/research")) -> pd.DataFrame:
    """Default research accessor; it cannot return sealed evaluation observations."""
    return pd.read_parquet(path / "development_session_panel.parquet")


def load_sealed_holdout(
    path: Path = Path("data/processed/research"), *, sequential_evaluator: bool = False
) -> pd.DataFrame:
    """Allow holdout access only to the future sequential evaluation engine."""
    if not sequential_evaluator:
        raise PermissionError("Sealed holdout is unavailable to strategy-development code")
    return pd.read_parquet(path / "sealed_holdout_session_panel.parquet")
