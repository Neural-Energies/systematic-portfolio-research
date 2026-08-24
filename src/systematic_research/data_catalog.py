"""Read-only catalog and quality gate for supported market-data exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

Severity = Literal["critical", "high", "medium", "low"]
BAR_COLUMNS = {"-Ticker", "Date Time", "Open", "High", "Low", "Last", "Volume"}
DATABASE_BAR_COLUMNS = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
TICK_COLUMNS = {"-SYMBOL", "DATE", "PRICE", "TICKVOL", "BID", "ASK"}


@dataclass(frozen=True)
class Issue:
    severity: Severity
    code: str
    file: str
    symbol: str
    count: int
    message: str


@dataclass(frozen=True)
class FileProfile:
    file: str
    path: str
    sha256: str
    size_bytes: int
    data_kind: str
    symbol: str
    rows: int
    start: str
    end: str
    status: str


@dataclass(frozen=True)
class CatalogResult:
    profiles: list[FileProfile]
    issues: list[Issue]

    @property
    def blocking_issue_count(self) -> int:
        return sum(issue.severity in {"critical", "high"} for issue in self.issues)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _issue(
    issues: list[Issue],
    severity: Severity,
    code: str,
    path: Path,
    symbol: str,
    count: int,
    message: str,
) -> None:
    if count:
        issues.append(Issue(severity, code, path.name, symbol, count, message))


def _profile_file(path: Path) -> tuple[FileProfile, list[Issue]]:
    header = pd.read_csv(path, nrows=0)
    if DATABASE_BAR_COLUMNS.issubset(header.columns):
        frame = pd.read_csv(path, dtype=str, low_memory=False)
        source_format = "databento_continuous"
    else:
        frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        source_format = "investor_rt"
    issues: list[Issue] = []
    expected_symbol = path.stem.split("_")[0].upper()
    if BAR_COLUMNS.issubset(frame.columns):
        kind, symbol_column, date_column = "minute_bars", "-Ticker", "Date Time"
        timestamp_values = frame[date_column]
        close_column = "Last"
        key_columns = [symbol_column, date_column]
    elif DATABASE_BAR_COLUMNS.issubset(frame.columns):
        kind, symbol_column, date_column = "minute_bars", None, "Date"
        timestamp_values = frame["Date"].str.cat(frame["Time"], sep=" ")
        close_column = "Close"
        key_columns = ["Date", "Time"]
    elif TICK_COLUMNS.issubset(frame.columns):
        kind, symbol_column, date_column = "ticks", "-SYMBOL", "DATE"
        timestamp_values = frame[date_column]
        close_column = "PRICE"
        key_columns = [symbol_column, date_column]
    else:
        issue = Issue(
            "critical",
            "unknown_schema",
            path.name,
            expected_symbol,
            1,
            f"Columns: {list(frame.columns)}",
        )
        return (
            FileProfile(
                path.name,
                str(path),
                _hash_file(path),
                path.stat().st_size,
                "unknown",
                expected_symbol,
                len(frame),
                "",
                "",
                "blocked",
            ),
            [issue],
        )

    embedded_headers = int(
        frame[symbol_column].eq(symbol_column).sum()
        if symbol_column is not None
        else frame["Date"].eq("Date").sum()
    )
    _issue(
        issues,
        "high",
        "embedded_headers",
        path,
        expected_symbol,
        embedded_headers,
        "Header rows are embedded inside the data payload.",
    )
    if symbol_column is not None:
        frame = frame.loc[~frame[symbol_column].eq(symbol_column)].copy()
        symbols = sorted(
            str(value).strip().upper() for value in frame[symbol_column].dropna().unique()
        )
    else:
        frame = frame.loc[~frame["Date"].eq("Date")].copy()
        symbols = [expected_symbol]
    symbol = symbols[0] if len(symbols) == 1 else ",".join(symbols)
    _issue(
        issues,
        "critical",
        "multiple_symbols_in_file",
        path,
        symbol,
        max(len(symbols) - 1, 0),
        f"Expected one symbol; found {symbols}.",
    )
    _issue(
        issues,
        "critical",
        "filename_symbol_mismatch",
        path,
        symbol,
        int(bool(symbols) and expected_symbol not in symbols),
        f"Filename implies {expected_symbol}; payload contains {symbols}.",
    )

    if source_format == "databento_continuous":
        timestamp_values = frame["Date"].str.cat(frame["Time"], sep=" ")
    else:
        timestamp_values = frame[date_column]
    timestamps = pd.to_datetime(timestamp_values, errors="coerce")
    _issue(
        issues,
        "critical",
        "invalid_timestamps",
        path,
        symbol,
        int(timestamps.isna().sum()),
        "One or more timestamps could not be parsed.",
    )
    _issue(
        issues,
        "high",
        "out_of_order",
        path,
        symbol,
        int((timestamps.diff().dropna() < pd.Timedelta(0)).sum()),
        "Rows move backward in time.",
    )
    _issue(
        issues,
        "high",
        "exact_duplicate_rows",
        path,
        symbol,
        int(frame.duplicated().sum()),
        "Exact duplicate records would double-count observations.",
    )

    if kind == "minute_bars":
        if source_format == "databento_continuous":
            numeric_names = ["Open", "High", "Low", "Close", "Volume"]
        else:
            numeric_names = ["Open", "High", "Low", "Last", "Volume"]
            if "Open Interest" in frame:
                numeric_names.append("Open Interest")
        numeric = frame[numeric_names].apply(pd.to_numeric, errors="coerce")
        _issue(
            issues,
            "critical",
            "duplicate_symbol_timestamp",
            path,
            symbol,
            int(frame.duplicated(key_columns).sum()),
            "The minute-bar primary key is duplicated.",
        )
        _issue(
            issues,
            "critical",
            "invalid_numeric_values",
            path,
            symbol,
            int(numeric.isna().sum().sum()),
            "Required numeric values could not be parsed.",
        )
        invalid_ohlc = (numeric["High"] < numeric[["Open", "Low", close_column]].max(axis=1)) | (
            numeric["Low"] > numeric[["Open", "High", close_column]].min(axis=1)
        )
        _issue(
            issues,
            "critical",
            "invalid_ohlc",
            path,
            symbol,
            int(invalid_ohlc.sum()),
            "High/low does not contain open and last.",
        )
        _issue(
            issues,
            "high",
            "constant_price_series",
            path,
            symbol,
            int(numeric[close_column].nunique(dropna=True) <= 1),
            "The entire close-price series is constant and cannot produce valid returns.",
        )
        if "Open Interest" in numeric and bool((numeric["Open Interest"] == 0).all()):
            _issue(
                issues,
                "medium",
                "open_interest_unavailable",
                path,
                symbol,
                len(frame),
                "Open interest is always zero; treat it as unavailable.",
            )
    else:
        numeric = frame[["PRICE", "TICKVOL", "BID", "ASK"]].apply(pd.to_numeric, errors="coerce")
        _issue(
            issues,
            "critical",
            "invalid_numeric_values",
            path,
            symbol,
            int(numeric.isna().sum().sum()),
            "Required numeric values could not be parsed.",
        )
        _issue(
            issues,
            "high",
            "nonpositive_tick_volume",
            path,
            symbol,
            int((numeric["TICKVOL"] <= 0).sum()),
            "Trade tick volume must be positive.",
        )
        outside_quote = (numeric["PRICE"] < numeric["BID"]) | (numeric["PRICE"] > numeric["ASK"])
        _issue(
            issues,
            "medium",
            "price_outside_quote",
            path,
            symbol,
            int(outside_quote.sum()),
            "Trade price is outside the recorded bid/ask; confirm feed timing semantics.",
        )

    valid_times = timestamps.dropna()
    start = valid_times.min().isoformat() if not valid_times.empty else ""
    end = valid_times.max().isoformat() if not valid_times.empty else ""
    span = valid_times.max() - valid_times.min() if len(valid_times) > 1 else pd.Timedelta(0)
    _issue(
        issues,
        "medium",
        "short_history",
        path,
        symbol,
        int(span < pd.Timedelta(days=30)),
        "Less than 30 calendar days of history.",
    )
    status = "blocked" if any(i.severity in {"critical", "high"} for i in issues) else "review"
    if not issues:
        status = "ready"
    return (
        FileProfile(
            path.name,
            str(path),
            _hash_file(path),
            path.stat().st_size,
            kind,
            symbol,
            len(frame),
            start,
            end,
            status,
        ),
        issues,
    )


def scan_directory(source: Path, file_glob: str | None = None) -> CatalogResult:
    """Profile files and enforce one unique raw file per symbol and data type."""
    profiles: list[FileProfile] = []
    issues: list[Issue] = []
    paths = source.glob(file_glob) if file_glob else source.iterdir()
    for path in sorted(paths):
        if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".tsv"}:
            profile, file_issues = _profile_file(path)
            profiles.append(profile)
            issues.extend(file_issues)

    for key, code in (("sha256", "duplicate_file_payload"),):
        groups: dict[str, list[FileProfile]] = {}
        for profile in profiles:
            groups.setdefault(str(getattr(profile, key)), []).append(profile)
        for value, group in groups.items():
            if value and len(group) > 1:
                names = [profile.file for profile in group]
                for profile in group:
                    issues.append(
                        Issue(
                            "critical",
                            code,
                            profile.file,
                            profile.symbol,
                            len(group) - 1,
                            f"Uniqueness violation across files: {names}.",
                        )
                    )
    symbol_kind_groups: dict[tuple[str, str], list[FileProfile]] = {}
    for profile in profiles:
        symbol_kind_groups.setdefault((profile.symbol, profile.data_kind), []).append(profile)
    for (symbol, data_kind), group in symbol_kind_groups.items():
        if symbol and len(group) > 1:
            names = [profile.file for profile in group]
            for profile in group:
                issues.append(
                    Issue(
                        "critical",
                        "duplicate_symbol_files",
                        profile.file,
                        symbol,
                        len(group) - 1,
                        f"Only one {data_kind} file is allowed per symbol; found {names}.",
                    )
                )
    return CatalogResult(profiles, issues)


def write_catalog(result: CatalogResult, output: Path) -> None:
    """Write local evidence without modifying raw exports."""
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(profile) for profile in result.profiles]).to_csv(
        output / "manifest.csv", index=False
    )
    pd.DataFrame([asdict(issue) for issue in result.issues]).to_csv(
        output / "issues.csv", index=False
    )
    summary = {
        "files": len(result.profiles),
        "symbols": len({profile.symbol for profile in result.profiles if profile.symbol}),
        "issues": len(result.issues),
        "blocking_issues": result.blocking_issue_count,
        "severity_counts": {
            severity: sum(issue.severity == severity for issue in result.issues)
            for severity in ("critical", "high", "medium", "low")
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the read-only market-data quality catalog.")
    parser.add_argument("--config", type=Path, default=Path("config/data_sources.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    arguments = parser.parse_args()
    with arguments.config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    market_data = settings["market_data"]
    source = Path(str(market_data["source_directory"]))
    output = arguments.output or Path(
        str(market_data.get("catalog_output_directory", "data/catalog"))
    )
    result = scan_directory(source, market_data.get("file_glob"))
    write_catalog(result, output)
    print(
        f"Cataloged {len(result.profiles)} files; {len(result.issues)} issues, "
        f"{result.blocking_issue_count} blocking."
    )
    if arguments.strict and result.blocking_issue_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
