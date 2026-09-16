"""Read-only OANDA practice historical-candle importer.

Credentials are loaded exclusively from an ignored local .env file or the
process environment. This module has no order-management capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi
import pandas as pd

OANDA_PRACTICE = "https://api-fxpractice.oanda.com/v3/accounts"
DEFAULT_SYMBOLS = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF"]
STEP = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
}


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key and key not in os.environ:
            os.environ[key] = value


def _utc(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def fetch_candles(
    symbol: str, granularity: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if granularity not in STEP:
        raise ValueError(f"Supported granularities are {sorted(STEP)}.")
    token, account = os.environ.get("OANDA_API_TOKEN"), os.environ.get("OANDA_ACCOUNT_ID")
    if not token or not account:
        raise RuntimeError("Set OANDA_API_TOKEN and OANDA_ACCOUNT_ID in local .env.")
    cursor, rows = start, []
    context = ssl.create_default_context(cafile=certifi.where())
    while cursor < end:
        parameters = {
            "from": cursor.isoformat(),
            "count": 5000,
            "granularity": granularity,
            "price": "MBA",
        }
        endpoint = (
            f"{OANDA_PRACTICE}/{account}/instruments/{symbol}/candles?{urlencode(parameters)}"
        )
        request = Request(endpoint, headers={"Authorization": f"Bearer {token}"})
        with urlopen(request, context=context, timeout=30) as response:  # noqa: S310 - fixed HTTPS host
            candles = json.loads(response.read().decode("utf-8"))["candles"]
        complete = [candle for candle in candles if candle["complete"]]
        if not complete:
            break
        rows.extend(complete)
        next_cursor = _utc(complete[-1]["time"]) + STEP[granularity]
        if next_cursor <= cursor:
            raise RuntimeError("OANDA pagination did not advance.")
        cursor = next_cursor
        sleep(0.04)
    result = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "venue": "oanda_practice",
                "granularity": granularity,
                "timestamp_utc": _utc(candle["time"]),
                "open": float(candle["mid"]["o"]),
                "high": float(candle["mid"]["h"]),
                "low": float(candle["mid"]["l"]),
                "close": float(candle["mid"]["c"]),
                "bid_close": float(candle["bid"]["c"]),
                "ask_close": float(candle["ask"]["c"]),
                "tick_count": int(candle["volume"]),
            }
            for candle in rows
        ]
    )
    result = (
        result.loc[result["timestamp_utc"].lt(end)]
        .drop_duplicates("timestamp_utc")
        .sort_values("timestamp_utc")
    )
    if result.empty or result.duplicated("timestamp_utc").any():
        raise ValueError(f"Invalid OANDA candle payload for {symbol}.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import read-only OANDA practice historical candles."
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--granularity", default="M1")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/free_api/oanda_practice"))
    arguments = parser.parse_args()
    _load_dotenv(Path(".env"))
    start, end = _utc(arguments.start), _utc(arguments.end)
    records = []
    for symbol in arguments.symbols:
        data = fetch_candles(symbol, arguments.granularity, start, end)
        path = arguments.output / f"symbol={symbol}" / f"granularity={arguments.granularity}"
        path.mkdir(parents=True, exist_ok=True)
        output = path / f"bars_{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}.parquet"
        data.to_parquet(output, index=False, compression="zstd")
        records.append(
            {
                "symbol": symbol,
                "rows": len(data),
                "output_file": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    manifest = arguments.output / "download_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
