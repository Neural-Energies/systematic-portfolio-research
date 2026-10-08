from __future__ import annotations

import csv
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

from systematic_research.crypto_volume_profile_bootstrap import (
    DataQualityRecord,
    audit_csv,
    build_sqx_project,
    select_universe,
)


def _record(symbol: str, notional: float, eligible: bool = True) -> DataQualityRecord:
    return DataQualityRecord(
        symbol=symbol,
        path=f"{symbol}.csv",
        bytes=1,
        modified_utc="2026-09-30T00:00:00+00:00",
        sha256="0" * 64,
        rows=1,
        first_timestamp_utc="2023-08-01T00:00:00Z",
        last_timestamp_utc="2026-08-31T23:59:00Z",
        duplicate_or_reversed_timestamps=0,
        missing_minutes=0,
        invalid_ohlc_rows=0,
        nonpositive_price_rows=0,
        negative_volume_rows=0,
        zero_volume_rows=0,
        selection_quote_notional=notional,
        full_period_eligible=eligible,
    )


def test_select_universe_uses_only_eligible_preperiod_liquidity() -> None:
    records = [_record(f"S{i}", float(i)) for i in range(12)]
    records.append(_record("INCOMPLETE", 1_000_000.0, eligible=False))
    assert select_universe(records) == [f"S{i}" for i in range(11, 1, -1)]


def test_audit_csv_detects_a_missing_minute(tmp_path: Path) -> None:
    path = tmp_path / "TESTUSDT_M1.csv"
    start = datetime(2023, 8, 1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for offset in (0, 1, 3):
            timestamp = start + timedelta(minutes=offset)
            writer.writerow(
                [timestamp.strftime("%Y.%m.%d"), timestamp.strftime("%H:%M"), 1, 2, 1, 2, 3]
            )
    record = audit_csv(path)
    assert record.rows == 3
    assert record.missing_minutes == 1
    assert not record.full_period_eligible


def test_build_sqx_project_replaces_tasks_and_periods(tmp_path: Path) -> None:
    template = tmp_path / "template.cfx"
    task = (
        b"<Root><Chart name='rules' symbol='OLD'/>"
        b"<Setup dateFrom='1' dateTo='2' slippage='0' engine='x'>"
        b"<Chart name='data' symbol='OLD' timeframe='M1' spread='2'/></Setup>"
        b"<Range dateFrom='1' dateTo='2'/></Root>"
    )
    config = (
        b"<Project name='old' version='144.2953'><Tasks>"
        b"<Task taskXMLFile='Build-Task1.xml'/></Tasks><Resources/><Databanks/></Project>"
    )
    with zipfile.ZipFile(template, "w") as archive:
        archive.writestr("config.xml", config)
        archive.writestr("Build-Task1.xml", task)
    output = tmp_path / "project.cfx"
    build_sqx_project(template, output, ["BTCUSDT", "ETHUSDT"])
    with zipfile.ZipFile(output) as archive:
        project = ElementTree.fromstring(archive.read("config.xml"))
        first = ElementTree.fromstring(archive.read("Build-Task1.xml"))
        assert project.attrib["name"] == "Crypto VP Top10 H1 Research"
        tasks = project.find("Tasks")
        assert tasks is not None
        assert len(tasks) == 2
        assert {chart.attrib["symbol"] for chart in first.findall(".//Chart")} == {"BTCUSDT"}
        assert first.find("Range").attrib["dateFrom"] == "2025.03.01"  # type: ignore[union-attr]
