"""Bootstrap a leakage-safe StrategyQuant crypto volume-profile research project."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree

import pandas as pd

CSV_COLUMNS = ("date", "time", "open", "high", "low", "close", "volume")
SELECTION_START = datetime(2023, 8, 1)
DEVELOPMENT_START = datetime(2023, 9, 1)
DEVELOPMENT_END = datetime(2025, 8, 31, 23, 59)
OOS_START = datetime(2025, 3, 1)
LOCKOUT_START = datetime(2025, 9, 1)
LOCKOUT_END = datetime(2026, 8, 31, 23, 59)
PROJECT_NAME = "Crypto VP Top10 H1 Research"
CRYPTO_TICK_SIZES = {
    "BTCUSDT": "0.01",
    "ETHUSDT": "0.01",
    "XRPUSDT": "0.0001",
    "BNBUSDT": "0.01",
    "SOLUSDT": "0.01",
    "LTCUSDT": "0.01",
    "DOGEUSDT": "0.00001",
    "OPUSDT": "0.0001",
    "BCHUSDT": "0.1",
    "ARBUSDT": "0.0001",
}


@dataclass(frozen=True)
class DataQualityRecord:
    symbol: str
    path: str
    bytes: int
    modified_utc: str
    sha256: str
    rows: int
    first_timestamp_utc: str
    last_timestamp_utc: str
    duplicate_or_reversed_timestamps: int
    missing_minutes: int
    invalid_ohlc_rows: int
    nonpositive_price_rows: int
    negative_volume_rows: int
    zero_volume_rows: int
    selection_quote_notional: float
    full_period_eligible: bool


def _parse_timestamp(date_value: str, time_value: str) -> datetime:
    return datetime.strptime(f"{date_value} {time_value}", "%Y.%m.%d %H:%M")


def audit_csv(path: Path) -> DataQualityRecord:
    digest = hashlib.sha256()
    with path.open("rb") as binary:
        for block in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(block)

    rows = duplicates = missing = invalid_ohlc = nonpositive = negative_volume = zero_volume = 0
    selection_quote_notional = 0.0
    first: datetime | None = None
    previous: datetime | None = None
    last: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for values in reader:
            if len(values) != len(CSV_COLUMNS):
                raise ValueError(f"{path}: expected 7 columns, found {len(values)}")
            timestamp = _parse_timestamp(values[0], values[1])
            open_, high, low, close, volume = map(float, values[2:])
            if first is None:
                first = timestamp
            if previous is not None:
                delta_minutes = int((timestamp - previous).total_seconds() // 60)
                if delta_minutes <= 0:
                    duplicates += 1
                elif delta_minutes > 1:
                    missing += delta_minutes - 1
            previous = last = timestamp
            rows += 1
            if min(open_, high, low, close) <= 0:
                nonpositive += 1
            if high < max(open_, low, close) or low > min(open_, high, close):
                invalid_ohlc += 1
            if volume < 0:
                negative_volume += 1
            if volume == 0:
                zero_volume += 1
            if SELECTION_START <= timestamp < DEVELOPMENT_START:
                selection_quote_notional += close * volume
    if first is None or last is None:
        raise ValueError(f"{path}: empty file")
    eligible = (
        first <= SELECTION_START
        and last >= LOCKOUT_END
        and duplicates == 0
        and invalid_ohlc == 0
        and nonpositive == 0
        and negative_volume == 0
        and missing / max(rows + missing, 1) <= 0.001
    )
    stat = path.stat()
    return DataQualityRecord(
        symbol=path.stem.removesuffix("_M1"),
        path=str(path.resolve()),
        bytes=stat.st_size,
        modified_utc=datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(),
        sha256=digest.hexdigest(),
        rows=rows,
        first_timestamp_utc=first.isoformat() + "Z",
        last_timestamp_utc=last.isoformat() + "Z",
        duplicate_or_reversed_timestamps=duplicates,
        missing_minutes=missing,
        invalid_ohlc_rows=invalid_ohlc,
        nonpositive_price_rows=nonpositive,
        negative_volume_rows=negative_volume,
        zero_volume_rows=zero_volume,
        selection_quote_notional=selection_quote_notional,
        full_period_eligible=eligible,
    )


def select_universe(records: list[DataQualityRecord], count: int = 10) -> list[str]:
    eligible = [record for record in records if record.full_period_eligible]
    ranked = sorted(eligible, key=lambda record: record.selection_quote_notional, reverse=True)
    if len(ranked) < count:
        raise ValueError(f"Only {len(ranked)} full-period symbols are eligible; need {count}.")
    return [record.symbol for record in ranked[:count]]


def build_sqx_project(template: Path, destination: Path, symbols: list[str]) -> None:
    """Create a ten-task SQX project from the installed volume-profile template."""
    with zipfile.ZipFile(template) as source:
        template_task = source.read("Build-Task1.xml")
        config_root = ElementTree.fromstring(source.read("config.xml"))
    config_root.attrib["name"] = PROJECT_NAME
    tasks = config_root.find("Tasks")
    if tasks is None:
        raise ValueError("Template has no Tasks node")
    tasks.clear()
    generated: dict[str, bytes] = {}
    for index, symbol in enumerate(symbols, start=1):
        task_filename = f"Build-Task{index}.xml"
        ElementTree.SubElement(
            tasks,
            "Task",
            {
                "type": "Build",
                "name": f"VP {symbol}",
                "showSettingsOverview": "false",
                "sampleName": "Custom",
                "active": "true",
                "version": config_root.attrib.get("version", "144.2953"),
                "taskXMLFile": task_filename,
            },
        )
        root = ElementTree.fromstring(template_task)
        primary_setup = root.find(".//Setup")
        charts = root.findall(".//Chart")
        primary_range = root.find(".//Range")
        if primary_setup is None or not charts or primary_range is None:
            raise ValueError("Template is missing primary Setup/Chart/Range settings")
        primary_setup.attrib.update(
            {
                "dateFrom": DEVELOPMENT_START.strftime("%Y.%m.%d"),
                "dateTo": DEVELOPMENT_END.strftime("%Y.%m.%d"),
                "slippage": "1",
                "engine": "MetaTrader5 (hedged)",
            }
        )
        for chart in charts:
            chart.attrib.update({"symbol": symbol, "timeframe": "H1", "spread": "0"})
        primary_range.attrib.update(
            {
                "dateFrom": OOS_START.strftime("%Y.%m.%d"),
                "dateTo": DEVELOPMENT_END.strftime("%Y.%m.%d"),
            }
        )
        generated[task_filename] = ElementTree.tostring(
            root, encoding="utf-8", xml_declaration=True
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as output:
        output.writestr(
            "config.xml",
            ElementTree.tostring(config_root, encoding="utf-8", xml_declaration=True),
        )
        for filename, payload in generated.items():
            output.writestr(filename, payload)


def write_cli_commands(output: Path, records: list[DataQualityRecord], symbols: list[str]) -> None:
    by_symbol = {record.symbol: record for record in records}
    lines: list[str] = []
    for symbol in symbols:
        tick_size = CRYPTO_TICK_SIZES[symbol]
        lines.append(
            f"-instrument action=add instrument={symbol} description=Binance_spot_{symbol} "
            f"pointvalue=1 ticksize={tick_size} tickstep={tick_size} defaultspread=0 "
            "datatype=crypto orderSizeMultiplier=1 orderSizeStep=0.000001"
        )
        lines.append(
            f"-symbol action=add symbols={symbol} instrument={symbol} bartype=startofbar "
            "datatype=M1 datasource=file timeframe=M1"
        )
    for symbol in symbols:
        csv_path = Path(by_symbol[symbol].path).as_posix()
        lines.append(
            f"-data action=import symbol={symbol} instrument={symbol} filepath={csv_path} "
            "bartype=startofbar errorhandling=stop timezone=Etc/UCT timeframe=M1 "
            'format="MetaTrader4 bar format" removeWeekends=false'
        )
    lines.extend(
        [
            f'-project action=start name="{PROJECT_NAME}"',
            f'-project action=status name="{PROJECT_NAME}"',
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    data_root: Path, output_root: Path, template: Path, sqx_project_root: Path
) -> dict[str, object]:
    paths = sorted(data_root.glob("*USDT_M1.csv"))
    if not paths:
        raise FileNotFoundError(f"No USDT minute CSV files found in {data_root}")
    records = [audit_csv(path) for path in paths]
    symbols = select_universe(records)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(asdict(record) for record in records).sort_values(
        "selection_quote_notional", ascending=False
    ).to_csv(output_root / "data_quality.csv", index=False)
    universe = {
        "selection_rule": (
            "Top 10 full-period eligible symbols by 2023-08 Binance spot quote notional"
        ),
        "selection_window_utc": ["2023-08-01T00:00:00Z", "2023-09-01T00:00:00Z"],
        "symbols": symbols,
    }
    (output_root / "frozen_universe.json").write_text(
        json.dumps(universe, indent=2), encoding="utf-8"
    )
    project_dir = sqx_project_root / PROJECT_NAME
    project_dir.mkdir(parents=True, exist_ok=True)
    for databank in ("Results", "Initial population", "Last generation", "Strategies to improve"):
        (project_dir / "databanks" / databank).mkdir(parents=True, exist_ok=True)
    build_sqx_project(template, project_dir / "project.cfx", symbols)
    write_cli_commands(output_root / "sqx_import_and_start.txt", records, symbols)
    manifest: dict[str, object] = {
        "program": "crypto_volume_profile_2026-09-30",
        "created_at_local": datetime.now().astimezone().isoformat(),
        "raw_data_root": str(data_root.resolve()),
        "raw_files_audited": len(records),
        "eligible_files": sum(record.full_period_eligible for record in records),
        "universe": symbols,
        "periods": {
            "selection": [
                SELECTION_START.isoformat(),
                (DEVELOPMENT_START - timedelta(minutes=1)).isoformat(),
            ],
            "in_sample": [
                DEVELOPMENT_START.isoformat(),
                (OOS_START - timedelta(minutes=1)).isoformat(),
            ],
            "development_oos": [OOS_START.isoformat(), DEVELOPMENT_END.isoformat()],
            "sealed_lockout": [LOCKOUT_START.isoformat(), LOCKOUT_END.isoformat()],
        },
        "strategyquant_project": str(project_dir.resolve()),
        "discovery_cost_note": (
            "SQX uses one-tick slippage; promoted candidates require independent dynamic "
            "fee/fill/impact validation."
        ),
        "lockout_status": "sealed_not_evaluated",
    }
    (output_root / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--sqx-project-root", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = run(
        arguments.data_root,
        arguments.output_root,
        arguments.template,
        arguments.sqx_project_root,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
