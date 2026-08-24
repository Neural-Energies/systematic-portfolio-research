from __future__ import annotations

from pathlib import Path

from systematic_research.data_catalog import scan_directory, write_catalog


def _write_bar(path: Path, symbol: str, rows: list[str]) -> None:
    header = "-Ticker\tDate Time\tOpen\tHigh\tLow\tLast\tVolume\tOpen Interest\n"
    path.write_text(header + "\n".join(f"{symbol}\t{row}" for row in rows), encoding="utf-8")


def _write_databento_bar(path: Path, rows: list[str]) -> None:
    header = "Date,Time,Open,High,Low,Close,Volume\n"
    path.write_text(header + "\n".join(rows), encoding="utf-8")


def test_valid_bar_file_is_cataloged(tmp_path: Path) -> None:
    _write_bar(
        tmp_path / "ESU6_Export.csv",
        "ESU6",
        [
            "2026-01-01T00:00\t100\t101\t99\t100.5\t10\t1",
            "2026-02-01T00:00\t101\t102\t100\t101.5\t12\t1",
        ],
    )
    result = scan_directory(tmp_path)
    assert len(result.profiles) == 1
    assert result.blocking_issue_count == 0


def test_duplicate_symbol_files_are_blocked(tmp_path: Path) -> None:
    row = ["2026-01-01T00:00\t100\t101\t99\t100.5\t10\t1"]
    _write_bar(tmp_path / "ESU6_Export.csv", "ESU6", row)
    _write_bar(tmp_path / "ESU6_Copy.csv", "ESU6", row)
    codes = {issue.code for issue in scan_directory(tmp_path).issues}
    assert {"duplicate_file_payload", "duplicate_symbol_files"}.issubset(codes)


def test_filename_symbol_mismatch_is_blocked(tmp_path: Path) -> None:
    _write_bar(
        tmp_path / "GCU6_Export.csv",
        "ESU6",
        ["2026-01-01T00:00\t100\t101\t99\t100.5\t10\t1"],
    )
    codes = {issue.code for issue in scan_directory(tmp_path).issues}
    assert "filename_symbol_mismatch" in codes


def test_catalog_artifacts_are_written(tmp_path: Path) -> None:
    source, output = tmp_path / "source", tmp_path / "catalog"
    source.mkdir()
    _write_bar(
        source / "ESU6_Export.csv",
        "ESU6",
        ["2026-01-01T00:00\t100\t101\t99\t100.5\t10\t1"],
    )
    write_catalog(scan_directory(source), output)
    assert {path.name for path in output.iterdir()} == {
        "manifest.csv",
        "issues.csv",
        "summary.json",
    }


def test_constant_price_series_is_blocked(tmp_path: Path) -> None:
    _write_bar(
        tmp_path / "6JU6_Export.csv",
        "6JU6",
        [
            "2026-01-01T00:00\t0.01\t0.01\t0.01\t0.01\t10\t0",
            "2026-02-01T00:00\t0.01\t0.01\t0.01\t0.01\t12\t0",
        ],
    )
    issues = scan_directory(tmp_path).issues
    assert "constant_price_series" in {issue.code for issue in issues}


def test_databento_continuous_bar_file_is_cataloged_as_utc_minutes(tmp_path: Path) -> None:
    _write_databento_bar(
        tmp_path / "ES_1Min_2023-08-22_to_2026-08-22.csv",
        [
            "2023-08-22,00:00:00,4405.75,4406.50,4405.50,4406.00,344",
            "2023-08-22,00:01:00,4406.00,4406.25,4405.50,4405.50,198",
        ],
    )
    result = scan_directory(tmp_path, "*_1Min_*.csv")
    assert result.blocking_issue_count == 0
    assert result.profiles[0].symbol == "ES"
    assert result.profiles[0].data_kind == "minute_bars"


def test_scan_glob_excludes_source_sidecar_files(tmp_path: Path) -> None:
    _write_databento_bar(
        tmp_path / "ZN_1Min_sample.csv",
        ["2023-08-22,00:00:00,108.96,108.97,108.95,108.96,395"],
    )
    (tmp_path / "validation_summary.csv").write_text("symbol,rows\nZN,1\n", encoding="utf-8")
    result = scan_directory(tmp_path, "*_1Min_*.csv")
    assert [profile.file for profile in result.profiles] == ["ZN_1Min_sample.csv"]
