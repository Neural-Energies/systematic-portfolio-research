"""Validated futures instrument master, session labels, and roll-policy metadata."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class RollPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["volume_crossover"]
    confirmation_sessions: int = Field(ge=1)
    fallback_business_days: int = Field(ge=1)


class InstrumentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    root: str
    name: str
    asset_class: str
    exchange: str
    currency: str
    price_multiplier: float = Field(gt=0)
    tick_size: float = Field(gt=0)
    tick_value: float = Field(gt=0)
    contract_months: list[str] = Field(min_length=1)
    settlement: Literal["cash", "physical"]
    roll: RollPolicy

    @model_validator(mode="after")
    def validate_tick_economics(self) -> InstrumentSpec:
        calculated = self.price_multiplier * self.tick_size
        if abs(calculated - self.tick_value) > 1e-9:
            raise ValueError(
                f"{self.root}: tick_size * price_multiplier={calculated}, "
                f"not tick_value={self.tick_value}"
            )
        return self

    def dollar_change(self, price_change: float, contracts: float = 1.0) -> float:
        """Convert a quoted price change into contract P&L before costs."""
        return price_change * self.price_multiplier * contracts


class SessionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timezone: str
    open_time: time
    close_time: time
    maintenance_break_minutes: int = Field(ge=0)
    note: str


class InstrumentMaster(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instruments: list[InstrumentSpec]
    session: SessionSpec

    @model_validator(mode="after")
    def validate_unique_roots(self) -> InstrumentMaster:
        roots = [instrument.root for instrument in self.instruments]
        if len(roots) != len(set(roots)):
            raise ValueError("Instrument roots must be unique")
        return self

    def by_root(self) -> dict[str, InstrumentSpec]:
        return {instrument.root: instrument for instrument in self.instruments}


def load_instrument_master(path: str | Path) -> InstrumentMaster:
    with Path(path).open(encoding="utf-8") as stream:
        return InstrumentMaster.model_validate(yaml.safe_load(stream))


def trading_date(timestamp_utc: datetime, session: SessionSpec) -> date:
    """Assign the preliminary Globex trading date using the configured session open.

    Holiday and early-close overrides are deliberately deferred to an exchange calendar.
    """
    if timestamp_utc.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware")
    local = timestamp_utc.astimezone(ZoneInfo(session.timezone))
    label = local.date()
    if local.time() >= session.open_time:
        label += timedelta(days=1)
    return label


def validate_contract_coverage(
    normalization_manifest: Path, master: InstrumentMaster
) -> pd.DataFrame:
    """Reconcile every observed contract root to exactly one metadata record."""
    manifest = pd.read_csv(normalization_manifest)
    known = master.by_root()
    symbols = manifest["symbol"].astype(str)
    contract_roots = symbols.str.extract(r"^(.+)[FGHJKMNQUVXZ]\d{1,2}$")[0]
    roots = contract_roots.where(contract_roots.notna(), symbols.where(symbols.isin(known)))
    result = manifest[["symbol", "status"]].copy()
    result["root"] = roots
    result["metadata_status"] = result["root"].map(
        lambda root: "configured" if root in known else "missing"
    )
    result["tick_value"] = result["root"].map(
        lambda root: known[root].tick_value if root in known else pd.NA
    )
    result["price_multiplier"] = result["root"].map(
        lambda root: known[root].price_multiplier if root in known else pd.NA
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate instrument metadata coverage.")
    parser.add_argument("--config", type=Path, default=Path("config/instruments.yaml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/canonical/normalization_manifest.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/catalog/instrument_coverage.csv"))
    arguments = parser.parse_args()
    master = load_instrument_master(arguments.config)
    coverage = validate_contract_coverage(arguments.manifest, master)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(arguments.output, index=False)
    missing = int(coverage["metadata_status"].eq("missing").sum())
    print(f"Validated {len(coverage)} contracts against {len(master.instruments)} instruments.")
    if missing:
        raise SystemExit(f"Missing instrument metadata for {missing} contracts.")


if __name__ == "__main__":
    main()
