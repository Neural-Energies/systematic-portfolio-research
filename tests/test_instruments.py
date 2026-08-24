from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from systematic_research.instruments import (
    InstrumentMaster,
    load_instrument_master,
    trading_date,
    validate_contract_coverage,
)


@pytest.fixture
def master() -> InstrumentMaster:
    return load_instrument_master("config/instruments.yaml")


def test_current_instrument_roots_are_unique(master: InstrumentMaster) -> None:
    assert len(master.by_root()) == 12


def test_contract_economics_are_correct(master: InstrumentMaster) -> None:
    instruments = master.by_root()
    assert instruments["ES"].dollar_change(0.25) == 12.50
    assert instruments["HG"].dollar_change(0.0005) == 12.50
    assert instruments["6J"].dollar_change(0.0000005) == 6.25
    assert instruments["ZN"].dollar_change(0.015625) == 15.625
    assert instruments["ZT"].dollar_change(0.00390625) == 7.8125
    assert instruments["6E"].dollar_change(0.00005) == 6.25
    assert instruments["NG"].dollar_change(0.001) == 10.0
    assert instruments["NKD"].dollar_change(5.0) == 25.0
    assert instruments["ZC"].dollar_change(0.25) == 12.5


def test_evening_timestamp_belongs_to_next_trading_date(master: InstrumentMaster) -> None:
    timestamp = datetime(2026, 8, 20, 22, 30, tzinfo=UTC)
    assert trading_date(timestamp, master.session).isoformat() == "2026-08-21"


def test_manifest_has_full_metadata_coverage(tmp_path: Path, master: InstrumentMaster) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"symbol": ["ESU6", "CLU6", "NQU6"], "status": ["normalized", "normalized", "quarantined"]}
    ).to_csv(manifest, index=False)
    coverage = validate_contract_coverage(manifest, master)
    assert coverage["metadata_status"].eq("configured").all()


def test_continuous_root_manifest_has_full_metadata_coverage(
    tmp_path: Path, master: InstrumentMaster
) -> None:
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"symbol": ["ES", "6E", "NG", "NKD", "ZC"], "status": ["normalized"] * 5}).to_csv(
        manifest, index=False
    )
    coverage = validate_contract_coverage(manifest, master)
    assert coverage["metadata_status"].eq("configured").all()
