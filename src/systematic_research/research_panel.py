"""Build non-backtesting minute and session research panels from canonical bars."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

from systematic_research.instruments import load_instrument_master
from systematic_research.temporal_validation import (
    SplitPolicy,
    seal_minute_partitions,
    write_temporal_split,
)


@dataclass(frozen=True)
class PanelQuality:
    symbol: str
    minute_rows: int
    sessions: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    duplicate_keys: int
    gap_events: int
    missing_minutes_inside_sessions: int
    one_minute_return_coverage_pct: float


def build_minute_returns(
    frame: pd.DataFrame, timezone: str, session_open_hour: int
) -> pd.DataFrame:
    """Create gap-aware one-minute returns without forward filling or crossing sessions."""
    required = {"symbol", "timestamp_utc", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Canonical bars missing columns: {sorted(missing)}")
    result = frame.sort_values(["symbol", "timestamp_utc"], ignore_index=True).copy()
    if result.duplicated(["symbol", "timestamp_utc"]).any():
        raise ValueError("Canonical symbol/timestamp keys must be unique")

    timestamps = pd.to_datetime(result["timestamp_utc"], utc=True)
    local = timestamps.dt.tz_convert(ZoneInfo(timezone))
    session_midnight = local.dt.normalize().dt.tz_localize(None)
    after_open = local.dt.hour >= session_open_hour
    result["trading_date"] = (
        session_midnight + pd.to_timedelta(after_open.astype(int), unit="D")
    ).dt.date

    same_symbol = result["symbol"].eq(result["symbol"].shift())
    same_session = result["trading_date"].eq(result["trading_date"].shift())
    minute_difference = timestamps.diff().dt.total_seconds().div(60.0)
    consecutive = same_symbol & same_session & minute_difference.eq(1.0)
    raw_return = result.groupby("symbol", sort=False)["close"].pct_change(fill_method=None)
    result["return_1m"] = raw_return.where(consecutive)
    gaps = (minute_difference - 1.0).where(
        same_symbol & same_session & minute_difference.gt(1.0), 0.0
    )
    result["gap_before_minutes"] = gaps.fillna(0.0).astype("int64")
    result["timestamp_utc"] = timestamps
    return result[
        [
            "symbol",
            "trading_date",
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_1m",
            "gap_before_minutes",
        ]
    ]


def build_session_panel(minute_returns: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-minute observations to descriptive session bars and risk measures."""
    frame = minute_returns.copy()
    frame["squared_return_1m"] = frame["return_1m"].pow(2)
    grouped = frame.groupby(["symbol", "trading_date"], sort=True, observed=True)
    sessions = grouped.agg(
        session_open_utc=("timestamp_utc", "min"),
        session_close_utc=("timestamp_utc", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_observations=("timestamp_utc", "size"),
        missing_minutes_inside_session=("gap_before_minutes", "sum"),
        realized_variance_1m=("squared_return_1m", "sum"),
    ).reset_index()
    sessions["realized_volatility_1m"] = np.sqrt(sessions.pop("realized_variance_1m"))
    sessions["close_to_close_return"] = sessions.groupby("symbol", sort=False)["close"].pct_change(
        fill_method=None
    )
    return sessions


def summarize_quality(minute_returns: pd.DataFrame) -> PanelQuality:
    symbol = str(minute_returns["symbol"].iloc[0])
    valid_returns = int(minute_returns["return_1m"].notna().sum())
    return PanelQuality(
        symbol=symbol,
        minute_rows=len(minute_returns),
        sessions=int(minute_returns["trading_date"].nunique()),
        first_timestamp_utc=pd.Timestamp(minute_returns["timestamp_utc"].min()).isoformat(),
        last_timestamp_utc=pd.Timestamp(minute_returns["timestamp_utc"].max()).isoformat(),
        duplicate_keys=int(minute_returns.duplicated(["symbol", "timestamp_utc"]).sum()),
        gap_events=int(minute_returns["gap_before_minutes"].gt(0).sum()),
        missing_minutes_inside_sessions=int(minute_returns["gap_before_minutes"].sum()),
        one_minute_return_coverage_pct=round(valid_returns / len(minute_returns) * 100.0, 4),
    )


def build_research_panel(
    canonical_root: Path,
    output_root: Path,
    catalog_root: Path,
    timezone: str,
    session_open_hour: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build physical minute partitions and cross-market descriptive tables."""
    quality_records: list[PanelQuality] = []
    session_frames: list[pd.DataFrame] = []
    active_symbols: set[str] = set()
    for path in sorted(canonical_root.glob("symbol=*/bars.parquet")):
        canonical = pd.read_parquet(path)
        minute_returns = build_minute_returns(canonical, timezone, session_open_hour)
        symbol = str(minute_returns["symbol"].iloc[0])
        active_symbols.add(symbol)
        destination = output_root / "minute_returns" / f"symbol={symbol}"
        destination.mkdir(parents=True, exist_ok=True)
        temporary = destination / "returns.parquet.tmp"
        output_file = destination / "returns.parquet"
        minute_returns.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(output_file)
        quality_records.append(summarize_quality(minute_returns))
        session_frames.append(build_session_panel(minute_returns))

    minute_root = output_root / "minute_returns"
    if minute_root.exists():
        for stale_file in minute_root.glob("symbol=*/returns.parquet"):
            stale_symbol = stale_file.parent.name.removeprefix("symbol=")
            if stale_symbol not in active_symbols:
                stale_file.unlink()

    sessions = pd.concat(session_frames, ignore_index=True).sort_values(
        ["trading_date", "symbol"], ignore_index=True
    )
    output_root.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(output_root / "session_panel.parquet", index=False, compression="zstd")

    catalog_root.mkdir(parents=True, exist_ok=True)
    quality = pd.DataFrame([asdict(record) for record in quality_records])
    quality.to_csv(catalog_root / "panel_quality.csv", index=False)
    return quality, sessions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build descriptive minute/session research panels."
    )
    parser.add_argument("--canonical", type=Path, default=Path("data/processed/canonical"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/research"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog"))
    parser.add_argument("--instruments", type=Path, default=Path("config/instruments.yaml"))
    parser.add_argument("--validation", type=Path, default=Path("config/validation.yaml"))
    arguments = parser.parse_args()
    master = load_instrument_master(arguments.instruments)
    quality, sessions = build_research_panel(
        arguments.canonical,
        arguments.output,
        arguments.catalog,
        master.session.timezone,
        master.session.open_time.hour,
    )
    with arguments.validation.open(encoding="utf-8") as stream:
        validation = yaml.safe_load(stream)["temporal_validation"]
    policy = SplitPolicy(
        holdout_months=int(validation["holdout_months"]),
        minimum_holdout_months=int(validation["minimum_holdout_months"]),
        embargo_days=int(validation["embargo_days"]),
    )
    boundaries = write_temporal_split(sessions, arguments.output, arguments.catalog, policy)
    seal_minute_partitions(arguments.output, boundaries)
    print(
        f"Built research panels for {len(quality)} symbols; "
        f"sealed holdout begins {boundaries.holdout_start}."
    )


if __name__ == "__main__":
    main()
