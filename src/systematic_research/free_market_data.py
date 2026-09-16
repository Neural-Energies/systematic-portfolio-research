"""Credential-free historical crypto ingestion with provenance and TLS validation.

The Binance public endpoint is used only for research data intake.  It is not an
execution integration, and every output retains its venue and download metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi
import pandas as pd

BINANCE_PUBLIC_KLINES = "https://data-api.binance.vision/api/v3/klines"
KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_ms",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]


@dataclass(frozen=True)
class DownloadRecord:
    symbol: str
    interval: str
    start_utc: str
    end_utc: str
    rows: int
    output_file: str
    sha256: str
    endpoint: str
    downloaded_at_utc: str


def parse_utc(value: str) -> datetime:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC").to_pydatetime()


def _request_json(parameters: dict[str, str | int]) -> list[list[Any]]:
    context = ssl.create_default_context(cafile=certifi.where())
    url = f"{BINANCE_PUBLIC_KLINES}?{urlencode(parameters)}"
    with urlopen(url, context=context, timeout=30) as response:  # noqa: S310 - fixed HTTPS endpoint
        payload: list[list[Any]] = json.loads(response.read().decode("utf-8"))
    return payload


def fetch_binance_klines(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Fetch public spot candles with pagination and strict timestamp checks."""
    if end <= start:
        raise ValueError("End timestamp must be later than start timestamp.")
    cursor = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    pages: list[list[Any]] = []
    while cursor < end_ms:
        payload = _request_json(
            {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms - 1,
                "limit": 1000,
            }
        )
        if not payload:
            break
        pages.extend(payload)
        next_cursor = int(payload[-1][0]) + 1
        if next_cursor <= cursor:
            raise RuntimeError("Public API pagination did not advance.")
        cursor = next_cursor
        if len(payload) == 1000:
            sleep(0.05)
    frame = pd.DataFrame(pages, columns=KLINE_COLUMNS)
    if frame.empty:
        raise ValueError(f"No {interval} bars returned for {symbol} in the requested interval.")
    frame["timestamp_utc"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame = frame.loc[frame["timestamp_utc"].lt(pd.Timestamp(end))].copy()
    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["trade_count"] = pd.to_numeric(frame["trade_count"], errors="raise")
    if (
        frame.duplicated("timestamp_utc").any()
        or not frame["timestamp_utc"].is_monotonic_increasing
    ):
        raise ValueError("Public API returned non-unique or unordered timestamps.")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Public API returned nonpositive prices.")
    if (frame["high"] < frame[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("Public API returned invalid high prices.")
    if (frame["low"] > frame[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Public API returned invalid low prices.")
    frame["symbol"] = symbol
    frame["venue"] = "binance_spot"
    frame["interval"] = interval
    return frame[
        [
            "symbol",
            "venue",
            "interval",
            "timestamp_utc",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trade_count",
        ]
    ]


def write_download(
    frame: pd.DataFrame, output_root: Path, start: datetime, end: datetime
) -> DownloadRecord:
    symbol = str(frame["symbol"].iloc[0])
    interval = str(frame["interval"].iloc[0])
    destination = output_root / "binance_spot" / f"symbol={symbol}" / f"interval={interval}"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / f"bars_{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}.parquet"
    frame.to_parquet(output, index=False, compression="zstd")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return DownloadRecord(
        symbol=symbol,
        interval=interval,
        start_utc=start.isoformat(),
        end_utc=end.isoformat(),
        rows=len(frame),
        output_file=str(output),
        sha256=digest,
        endpoint=BINANCE_PUBLIC_KLINES,
        downloaded_at_utc=datetime.now(UTC).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import free public Binance historical spot candles."
    )
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", required=True, help="Inclusive UTC ISO-8601 timestamp.")
    parser.add_argument("--end", required=True, help="Exclusive UTC ISO-8601 timestamp.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/free_api"))
    arguments = parser.parse_args()
    start, end = parse_utc(arguments.start), parse_utc(arguments.end)
    records = [
        asdict(
            write_download(
                fetch_binance_klines(symbol, arguments.interval, start, end),
                arguments.output,
                start,
                end,
            )
        )
        for symbol in arguments.symbols
    ]
    manifest = arguments.output / "binance_spot" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
