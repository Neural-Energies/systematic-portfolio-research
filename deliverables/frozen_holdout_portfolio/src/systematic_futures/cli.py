from __future__ import annotations

import argparse
from pathlib import Path

from .data import read_minute_csv, to_hourly
from .signals import generate_candidate_orders


def main() -> None:
    parser = argparse.ArgumentParser(description="Systematic futures signal generator")
    sub = parser.add_subparsers(dest="command", required=True)
    signals = sub.add_parser("signals", help="generate candidate stop orders")
    signals.add_argument("--symbol", required=True)
    signals.add_argument("--input", required=True, type=Path)
    signals.add_argument("--output", required=True, type=Path)
    signals.add_argument("--quantity", type=float, default=1.0)
    args = parser.parse_args()

    minute = read_minute_csv(args.input)
    hourly = to_hourly(minute)
    orders = generate_candidate_orders(args.symbol, hourly, args.quantity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(args.output, index=False, date_format="%Y-%m-%dT%H:%M:%SZ")
    print(f"wrote {len(orders):,} candidate orders to {args.output}")


if __name__ == "__main__":
    main()
