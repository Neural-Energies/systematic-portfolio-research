"""Canonical, provenance-preserving normalization of validated minute bars."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from systematic_research.data_catalog import FileProfile, scan_directory

CONTRACT_PATTERN = re.compile(r"^(.+)([FGHJKMNQUVXZ])(\d{1,2})$")
CANONICAL_COLUMNS = [
    "symbol",
    "root_symbol",
    "contract_month_code",
    "contract_year",
    "timestamp_local",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "source_file",
    "source_sha256",
]


class NormalizationError(ValueError):
    """Raised when raw data cannot safely enter the canonical layer."""


@dataclass(frozen=True)
class NormalizationRecord:
    symbol: str
    source_file: str
    source_sha256: str
    input_rows: int
    output_rows: int
    timezone: str
    output_file: str
    status: str
    note: str


def parse_contract(symbol: str, as_of_year: int) -> tuple[str, str, int]:
    """Split a compact futures symbol into root, month code, and four-digit year."""
    match = CONTRACT_PATTERN.fullmatch(symbol)
    if match is None:
        raise NormalizationError(f"Unsupported futures contract symbol: {symbol}")
    root, month_code, year_text = match.groups()
    year_value = int(year_text)
    if len(year_text) == 1:
        decade = as_of_year - as_of_year % 10
        year_value += decade
        if year_value < as_of_year - 5:
            year_value += 10
    return root, month_code, year_value


def normalize_bar_file(
    profile: FileProfile,
    output_root: Path,
    timezone: str,
    as_of_year: int,
) -> NormalizationRecord:
    """Normalize one quality-approved minute-bar export to canonical Parquet."""
    if profile.data_kind != "minute_bars":
        raise NormalizationError(f"Only minute bars are supported in Step 1: {profile.file}")
    header = pd.read_csv(profile.path, nrows=0)
    is_databento = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}.issubset(
        header.columns
    )
    if is_databento:
        frame = pd.read_csv(profile.path, low_memory=False)
        frame["symbol"] = profile.symbol
        frame["timestamp_local"] = (
            frame["Date"].astype(str).str.cat(frame["Time"].astype(str), sep=" ")
        )
        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        frame["open_interest"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    else:
        frame = pd.read_csv(profile.path, sep="\t", low_memory=False)
        frame = frame.rename(
            columns={
                "-Ticker": "symbol",
                "Date Time": "timestamp_local",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Last": "close",
                "Volume": "volume",
                "Open Interest": "open_interest",
            }
        )
    if frame.duplicated(["symbol", "timestamp_local"]).any():
        raise NormalizationError(f"Duplicate primary keys in {profile.file}")
    symbols = frame["symbol"].astype(str).str.strip().str.upper()
    if symbols.nunique() != 1 or symbols.iloc[0] != profile.symbol:
        raise NormalizationError(f"Symbol identity changed while reading {profile.file}")
    frame["symbol"] = symbols

    local_times = pd.to_datetime(frame["timestamp_local"], errors="raise")
    if local_times.dt.tz is not None:
        raise NormalizationError(
            f"Expected timezone-naive Investor/RT timestamps in {profile.file}"
        )
    try:
        utc_times = local_times.dt.tz_localize(
            ZoneInfo(timezone), ambiguous="raise", nonexistent="raise"
        ).dt.tz_convert("UTC")
    except (ValueError, TypeError) as error:
        raise NormalizationError(
            f"Timezone localization failed for {profile.file}; confirm source timezone"
        ) from error
    frame["timestamp_local"] = local_times
    frame["timestamp_utc"] = utc_times

    for column in ("open", "high", "low", "close", "volume", "open_interest"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise NormalizationError(f"Nonpositive prices in {profile.file}")
    if (frame["volume"] < 0).any():
        raise NormalizationError(f"Negative volume in {profile.file}")
    if (frame["open_interest"] == 0).all():
        frame["open_interest"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")

    match = CONTRACT_PATTERN.fullmatch(profile.symbol)
    if match is None:
        root = profile.symbol
        frame["contract_month_code"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["contract_year"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    else:
        root, month_code, contract_year = parse_contract(profile.symbol, as_of_year)
        frame["contract_month_code"] = month_code
        frame["contract_year"] = contract_year
    frame["root_symbol"] = root
    frame["source_file"] = profile.file
    frame["source_sha256"] = profile.sha256
    frame = frame[CANONICAL_COLUMNS].sort_values("timestamp_utc", ignore_index=True)
    if frame.duplicated(["symbol", "timestamp_utc"]).any():
        raise NormalizationError(f"UTC primary key collision in {profile.file}")

    symbol_directory = output_root / f"symbol={profile.symbol}"
    symbol_directory.mkdir(parents=True, exist_ok=True)
    output_file = symbol_directory / "bars.parquet"
    temporary_file = symbol_directory / "bars.parquet.tmp"
    frame.to_parquet(temporary_file, index=False, compression="zstd")
    temporary_file.replace(output_file)
    return NormalizationRecord(
        profile.symbol,
        profile.file,
        profile.sha256,
        len(frame),
        len(frame),
        timezone,
        str(output_file),
        "normalized",
        "Open interest converted to null when source contained only zeroes.",
    )


def normalize_directory(
    source: Path,
    output_root: Path,
    timezone: str,
    as_of_year: int,
    file_glob: str | None = None,
) -> list[NormalizationRecord]:
    """Normalize approved bars and explicitly quarantine blocked or tick files."""
    catalog = scan_directory(source, file_glob)
    blocked_files = {
        issue.file for issue in catalog.issues if issue.severity in {"critical", "high"}
    }
    approved_bar_symbols = {
        profile.symbol
        for profile in catalog.profiles
        if profile.data_kind == "minute_bars" and profile.file not in blocked_files
    }
    records: list[NormalizationRecord] = []
    for profile in catalog.profiles:
        if profile.file in blocked_files:
            if profile.data_kind == "minute_bars" and profile.symbol not in approved_bar_symbols:
                stale_output = output_root / f"symbol={profile.symbol}" / "bars.parquet"
                if stale_output.exists():
                    stale_output.unlink()
            records.append(
                NormalizationRecord(
                    profile.symbol,
                    profile.file,
                    profile.sha256,
                    profile.rows,
                    0,
                    timezone,
                    "",
                    "quarantined",
                    "Critical or high-severity catalog issue.",
                )
            )
        elif profile.data_kind != "minute_bars":
            records.append(
                NormalizationRecord(
                    profile.symbol,
                    profile.file,
                    profile.sha256,
                    profile.rows,
                    0,
                    timezone,
                    "",
                    "deferred",
                    "Tick normalization is outside Step 1 bar normalization.",
                )
            )
        else:
            records.append(normalize_bar_file(profile, output_root, timezone, as_of_year))
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(
        output_root / "normalization_manifest.csv", index=False
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical minute-bar Parquet data.")
    parser.add_argument("--config", type=Path, default=Path("config/data_sources.yaml"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)["market_data"]
    timezone = settings.get("assumed_timezone")
    if not timezone:
        raise SystemExit("Set market_data.assumed_timezone before normalization.")
    output = arguments.output or Path(
        str(settings.get("canonical_output_directory", "data/processed/canonical"))
    )
    records = normalize_directory(
        Path(str(settings["source_directory"])),
        output,
        str(timezone),
        int(settings["as_of_year"]),
        settings.get("file_glob"),
    )
    normalized = sum(record.status == "normalized" for record in records)
    quarantined = sum(record.status == "quarantined" for record in records)
    print(f"Normalized {normalized} files; quarantined {quarantined}; raw files unchanged.")


if __name__ == "__main__":
    main()
