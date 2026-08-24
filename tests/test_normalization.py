from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from systematic_research.data_catalog import scan_directory
from systematic_research.normalization import (
    CANONICAL_COLUMNS,
    NormalizationError,
    normalize_bar_file,
    normalize_directory,
    parse_contract,
)


def _write_bar(path: Path, symbol: str, timestamps: list[str], open_interest: int = 0) -> None:
    header = "-Ticker\tDate Time\tOpen\tHigh\tLow\tLast\tVolume\tOpen Interest\n"
    rows = [
        f"{symbol}\t{timestamp}\t{100 + index}\t{101 + index}\t{99 + index}\t"
        f"{100.5 + index}\t10\t{open_interest}"
        for index, timestamp in enumerate(timestamps)
    ]
    path.write_text(header + "\n".join(rows), encoding="utf-8")


def _write_databento_bar(path: Path) -> None:
    path.write_text(
        "Date,Time,Open,High,Low,Close,Volume\n"
        "2023-08-22,00:00:00,4405.75,4406.50,4405.50,4406.00,344\n"
        "2023-08-22,00:01:00,4406.00,4406.25,4405.50,4405.50,198\n",
        encoding="utf-8",
    )


def test_contract_symbol_is_parsed() -> None:
    assert parse_contract("6JU6", 2026) == ("6J", "U", 2026)
    assert parse_contract("GCZ6", 2026) == ("GC", "Z", 2026)


def test_bar_normalization_preserves_provenance_and_adds_utc(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _write_bar(source / "ESU6_Export.csv", "ESU6", ["2026-01-02T09:30"])
    profile = scan_directory(source).profiles[0]
    record = normalize_bar_file(profile, output, "America/New_York", 2026)
    normalized = pd.read_parquet(record.output_file)
    assert normalized.columns.tolist() == CANONICAL_COLUMNS
    assert str(normalized["timestamp_utc"].dt.tz) == "UTC"
    assert normalized["source_sha256"].iloc[0] == profile.sha256
    assert normalized["open_interest"].isna().all()


def test_duplicate_bar_keys_are_rejected(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _write_bar(
        source / "ESU6_Export.csv",
        "ESU6",
        ["2026-01-02T09:30", "2026-01-02T09:30"],
        open_interest=1,
    )
    profile = scan_directory(source).profiles[0]
    with pytest.raises(NormalizationError, match="Duplicate primary keys"):
        normalize_bar_file(profile, output, "America/New_York", 2026)


def test_blocked_file_is_quarantined_without_output(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _write_bar(
        source / "ESU6_Export.csv",
        "ESU6",
        ["2026-01-02T09:30", "2026-01-02T09:30"],
        open_interest=1,
    )
    records = normalize_directory(source, output, "America/New_York", 2026)
    assert records[0].status == "quarantined"
    assert not (output / "symbol=ESU6" / "bars.parquet").exists()


def test_blocked_ticks_do_not_remove_approved_bars(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _write_bar(
        source / "NQU6_Export.csv",
        "NQU6",
        ["2026-01-01T00:00", "2026-02-01T00:00"],
    )
    (source / "NQU6_Ticks.txt").write_text(
        "-SYMBOL\tDATE\tPRICE\tTICKVOL\tBID\tASK\nNQU6\t2026-01-01T00:00:00\t100\t0\t99\t101\n",
        encoding="utf-8",
    )
    records = normalize_directory(source, output, "America/New_York", 2026)
    assert {record.status for record in records} == {"normalized", "quarantined"}
    assert (output / "symbol=NQU6" / "bars.parquet").exists()


def test_databento_continuous_bar_normalization_preserves_utc_and_root(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    _write_databento_bar(source / "ES_1Min_2023-08-22_to_2026-08-22.csv")
    profile = scan_directory(source, "*_1Min_*.csv").profiles[0]
    record = normalize_bar_file(profile, output, "UTC", 2026)
    normalized = pd.read_parquet(record.output_file)
    assert normalized["symbol"].unique().tolist() == ["ES"]
    assert normalized["root_symbol"].unique().tolist() == ["ES"]
    assert normalized["contract_month_code"].isna().all()
    assert normalized["contract_year"].isna().all()
    assert normalized["open_interest"].isna().all()
    assert str(normalized["timestamp_utc"].dt.tz) == "UTC"
