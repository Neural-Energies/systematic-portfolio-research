from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_REFERENCE = Path(
    os.environ.get("FUTURES_HOLDOUT_REFERENCE", ROOT / "data" / "private_reference")
).expanduser()
OUTPUT = ROOT / "reports" / "holdout_visual_report"
START = pd.Timestamp("2025-08-22", tz="UTC")
END = pd.Timestamp("2026-08-21", tz="UTC")
END_EXCLUSIVE = pd.Timestamp("2026-08-22", tz="UTC")
INITIAL_CAPITAL = 700_000.0
SYMBOLS = ["6E", "6J", "CL", "ES", "GC", "HG", "NG"]


def compact_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.replace({np.nan: None, np.inf: None, -np.inf: None})
    return cast(list[dict[str, Any]], clean.to_dict(orient="records"))


def source_spec() -> dict[str, Any]:
    return {
        "id": "holdout-ledgers",
        "label": "Frozen holdout trade ledgers (seven futures strategies)",
        "query": {
            "engine": "DuckDB + Python/pandas",
            "language": "SQL",
            "sql": "SELECT * FROM read_csv_auto('holdout_reference/*/strategy_*.csv', delim=';', header=true, union_by_name=true, filename=true);",
            "description": "Loads all seven semicolon-delimited closed-trade ledgers; the report transformation standardizes UTC timestamps and aggregates P&L by close date and strategy.",
            "tables_used": [
                f"holdout_reference/{symbol}/strategy_{symbol}.csv" for symbol in SYMBOLS
            ],
            "filters": [
                "Frozen test window: August 22, 2025 through August 21, 2026",
                "All seven strategy ledgers included",
                "One contract per strategy",
                "Closed trades aggregated on UTC close date",
            ],
            "metric_definitions": [
                "Net profit = sum of ledger Profit/Loss across all 1,471 trades.",
                "Daily Sharpe = mean daily return divided by sample standard deviation, multiplied by sqrt(252); calendar-day zero-return observations are retained.",
                "Profit factor = gross positive trade P&L divided by absolute gross negative trade P&L.",
                "Closed-trade drawdown = current cumulative closed-trade equity minus its prior peak.",
                "Contribution share = strategy net profit divided by portfolio net profit.",
            ],
        },
    }


def load_trades() -> pd.DataFrame:
    if not PRIVATE_REFERENCE.is_dir():
        raise FileNotFoundError(
            "Holdout ledgers are not available. Set FUTURES_HOLDOUT_REFERENCE to the "
            "private directory containing one <symbol>/strategy_<symbol>.csv file for "
            "each of 6E, 6J, CL, ES, GC, HG, and NG."
        )
    frames = []
    required = {
        "Ticket",
        "Symbol",
        "Type",
        "Open time",
        "Open price",
        "Size",
        "Close time",
        "Close price",
        "Profit/Loss",
        "Close type",
    }
    for symbol in SYMBOLS:
        path = PRIVATE_REFERENCE / symbol / f"strategy_{symbol}.csv"
        frame = pd.read_csv(path, sep=";")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        frame["strategy"] = symbol
        frames.append(frame)
    trades = pd.concat(frames, ignore_index=True)
    trades["open_time"] = pd.to_datetime(trades["Open time"], format="%Y.%m.%d %H:%M:%S", utc=True)
    trades["close_time"] = pd.to_datetime(
        trades["Close time"], format="%Y.%m.%d %H:%M:%S", utc=True
    )
    trades["pnl"] = pd.to_numeric(trades["Profit/Loss"])
    if trades.duplicated(["strategy", "Ticket"]).any():
        raise ValueError("duplicate strategy/ticket pairs found")
    if trades["open_time"].min() < START or trades["close_time"].max() >= END_EXCLUSIVE:
        raise ValueError("trade timestamps extend outside the frozen holdout")
    if len(trades) != 1471:
        raise ValueError(f"expected 1,471 trades, found {len(trades):,}")
    if not math.isclose(float(trades["pnl"].sum()), 194_682.0, abs_tol=0.01):
        raise ValueError("net profit does not reconcile to the frozen result")
    return trades.sort_values(["close_time", "strategy", "Ticket"]).reset_index(drop=True)


def build() -> tuple[dict[str, Any], dict[str, Any]]:
    trades = load_trades()
    calendar = pd.date_range(START, END, freq="D")
    daily_pnl = (
        trades.set_index("close_time")["pnl"].resample("D").sum().reindex(calendar, fill_value=0.0)
    )
    daily_return = daily_pnl / INITIAL_CAPITAL
    equity = INITIAL_CAPITAL + daily_pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    drawdown_pct = equity / peak - 1.0

    sharpe = math.sqrt(252) * daily_return.mean() / daily_return.std(ddof=1)
    weekday_pnl = daily_pnl[pd.DatetimeIndex(daily_pnl.index).dayofweek < 5]
    weekday_return = weekday_pnl / INITIAL_CAPITAL
    weekday_sharpe = math.sqrt(252) * weekday_return.mean() / weekday_return.std(ddof=1)

    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    profit_factor = gross_profit / gross_loss
    win_rate = (trades["pnl"] > 0).mean()

    trade_equity = INITIAL_CAPITAL + trades["pnl"].cumsum()
    trade_drawdown = trade_equity - trade_equity.cummax()
    trade_max_drawdown = float(trade_drawdown.min())

    daily = pd.DataFrame(
        {
            "date": calendar.strftime("%Y-%m-%d"),
            "daily_pnl": daily_pnl.round(2).to_numpy(),
            "equity": equity.round(2).to_numpy(),
            "drawdown_dollars": drawdown.round(2).to_numpy(),
            "drawdown_pct": drawdown_pct.to_numpy(),
            "high_water_mark": peak.round(2).to_numpy(),
        }
    )

    monthly_pnl = daily_pnl.resample("MS").sum()
    monthly_index = pd.DatetimeIndex(monthly_pnl.index)
    monthly = pd.DataFrame(
        {
            "month": monthly_index.strftime("%b %Y"),
            "month_sort": monthly_index.strftime("%Y-%m"),
            "monthly_pnl": monthly_pnl.round(2).to_numpy(),
            "positive": np.where(monthly_pnl.to_numpy() >= 0, "Positive", "Negative"),
        }
    )

    strategy_rows = []
    for symbol, group in trades.groupby("strategy", sort=False):
        gp = group.loc[group["pnl"] > 0, "pnl"].sum()
        gl = -group.loc[group["pnl"] < 0, "pnl"].sum()
        strategy_rows.append(
            {
                "strategy": symbol,
                "trades": int(len(group)),
                "net_profit": round(float(group["pnl"].sum()), 2),
                "profit_factor": round(float(gp / gl), 4) if gl else None,
                "win_rate": float((group["pnl"] > 0).mean()),
                "average_trade": round(float(group["pnl"].mean()), 2),
                "contribution_share": float(group["pnl"].sum() / trades["pnl"].sum()),
            }
        )
    strategies = (
        pd.DataFrame(strategy_rows)
        .sort_values("net_profit", ascending=False)
        .reset_index(drop=True)
    )

    bins = np.linspace(float(trades["pnl"].quantile(0.01)), float(trades["pnl"].quantile(0.99)), 25)
    clipped = trades["pnl"].clip(bins[0], bins[-1])
    counts, edges = np.histogram(clipped, bins=bins)
    distribution = pd.DataFrame(
        {
            "bin_midpoint": ((edges[:-1] + edges[1:]) / 2).round(2),
            "trade_count": counts.astype(int),
            "bin_start": edges[:-1].round(2),
            "bin_end": edges[1:].round(2),
        }
    )

    close_types = (
        trades.groupby(["Close type", "strategy"], as_index=False)
        .agg(trades=("pnl", "size"), net_profit=("pnl", "sum"))
        .sort_values(["trades", "Close type"], ascending=[False, True])
    )

    gc_share = float(strategies.loc[strategies["strategy"] == "GC", "contribution_share"].iloc[0])
    positive_months = int((monthly_pnl > 0).sum())
    metric_values = [
        {
            "net_profit": float(trades["pnl"].sum()),
            "sharpe": float(sharpe),
            "candidate_threshold": 1.5,
            "max_drawdown": trade_max_drawdown,
            "max_drawdown_pct": float(trade_max_drawdown / INITIAL_CAPITAL),
            "profit_factor": float(profit_factor),
            "trades": int(len(trades)),
            "win_rate": float(win_rate),
        }
    ]

    source = source_spec()
    generated_at = pd.Timestamp.now(tz="UTC").isoformat()
    datasets = {
        "metric_values": metric_values,
        "daily_equity": compact_rows(daily),
        "monthly_pnl": compact_rows(monthly),
        "strategy_results": compact_rows(strategies),
        "trade_distribution": compact_rows(distribution),
        "close_type_results": compact_rows(close_types),
    }

    cards = [
        {
            "id": "net-profit",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Closed-trade net profit across seven one-contract strategies.",
            "metrics": [{"label": "Net profit", "field": "net_profit", "format": "currency"}],
        },
        {
            "id": "sharpe",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Daily Sharpe with calendar-day zero returns retained and 252-day annualization.",
            "metrics": [
                {"label": "Holdout Sharpe", "field": "sharpe", "format": "number"},
                {"label": "Revised threshold", "field": "candidate_threshold", "format": "number"},
            ],
        },
        {
            "id": "drawdown",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Worst peak-to-trough decline on the closed-trade equity sequence.",
            "metrics": [
                {"label": "Max drawdown", "field": "max_drawdown", "format": "currency"},
                {"label": "Of initial capital", "field": "max_drawdown_pct", "format": "percent"},
            ],
        },
        {
            "id": "profit-factor",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Gross winning trade P&L divided by absolute gross losing trade P&L.",
            "metrics": [{"label": "Profit factor", "field": "profit_factor", "format": "number"}],
        },
        {
            "id": "trades",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Closed trades across all seven strategies during the frozen year.",
            "metrics": [{"label": "Trades", "field": "trades", "format": "number"}],
        },
        {
            "id": "win-rate",
            "dataset": "metric_values",
            "sourceId": source["id"],
            "description": "Share of closed trades with positive ledger P&L.",
            "metrics": [{"label": "Win rate", "field": "win_rate", "format": "percent"}],
        },
    ]

    charts = [
        {
            "id": "equity-curve",
            "title": "Closed-trade equity curve",
            "subtitle": "August 22, 2025 through August 21, 2026; starts at $700k",
            "type": "line",
            "intent": "trend",
            "question": "How did cumulative portfolio equity evolve through the holdout year?",
            "rationale": "A line chart shows path, stagnation, acceleration, and recovery across 365 daily observations.",
            "dataset": "daily_equity",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "date", "type": "temporal", "label": "Date"},
                "y": {
                    "field": "equity",
                    "type": "quantitative",
                    "format": "currency",
                    "label": "Equity",
                },
                "tooltip": [
                    {"field": "daily_pnl", "label": "Daily P&L", "format": "currency"},
                    {"field": "drawdown_dollars", "label": "Drawdown", "format": "currency"},
                ],
            },
            "xAxisTitle": "Date",
            "yAxisTitle": "Portfolio equity",
            "valueFormat": "currency",
            "layout": "full",
            "surface": {"legend": {"show": False}, "labels": {"values": "endpoints"}},
        },
        {
            "id": "drawdown-curve",
            "title": "Closed-trade drawdown",
            "subtitle": "Dollar distance below the prior equity high",
            "type": "area",
            "intent": "trend",
            "question": "When and how deeply did the portfolio fall below its prior peak?",
            "rationale": "A filled area emphasizes the duration and depth of underwater periods while retaining chronological context.",
            "dataset": "daily_equity",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "date", "type": "temporal", "label": "Date"},
                "y": {
                    "field": "drawdown_dollars",
                    "type": "quantitative",
                    "format": "currency",
                    "label": "Drawdown",
                },
                "tooltip": [
                    {"field": "drawdown_pct", "label": "Drawdown %", "format": "percent"},
                    {"field": "equity", "label": "Equity", "format": "currency"},
                ],
            },
            "xAxisTitle": "Date",
            "yAxisTitle": "Drawdown",
            "valueFormat": "currency",
            "layout": "full",
            "surface": {"legend": {"show": False}, "labels": {"values": "none"}},
        },
        {
            "id": "monthly-pnl",
            "title": "Monthly net P&L",
            "subtitle": f"{positive_months} of {len(monthly_pnl)} holdout months were profitable",
            "type": "bar",
            "intent": "trend",
            "question": "Was profitability persistent across months or concentrated in a few bursts?",
            "rationale": "Monthly bars expose sign, magnitude, and temporal consistency without implying a smooth path.",
            "dataset": "monthly_pnl",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "month", "type": "ordinal", "label": "Month"},
                "y": {
                    "field": "monthly_pnl",
                    "type": "quantitative",
                    "format": "currency",
                    "label": "Net P&L",
                },
                "tooltip": [{"field": "positive", "label": "Outcome"}],
            },
            "xAxisTitle": "Month",
            "yAxisTitle": "Net P&L",
            "valueFormat": "currency",
            "layout": "full",
            "surface": {"legend": {"show": False}, "labels": {"values": "auto"}},
        },
        {
            "id": "strategy-contribution",
            "title": "Net profit by strategy",
            "subtitle": f"GC generated {gc_share:.1%} of portfolio net profit",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "Which markets generated or detracted from holdout profit?",
            "rationale": "Sorted horizontal bars make concentration and negative contributors immediately comparable across seven markets.",
            "dataset": "strategy_results",
            "sourceId": source["id"],
            "encodings": {
                "x": {"field": "strategy", "type": "nominal", "label": "Strategy"},
                "y": {
                    "field": "net_profit",
                    "type": "quantitative",
                    "format": "currency",
                    "label": "Net profit",
                },
                "tooltip": [
                    {"field": "trades", "label": "Trades", "format": "number"},
                    {"field": "profit_factor", "label": "Profit factor", "format": "number"},
                    {"field": "contribution_share", "label": "Contribution", "format": "percent"},
                ],
            },
            "xAxisTitle": "Net profit",
            "yAxisTitle": "Strategy",
            "valueFormat": "currency",
            "layout": "full",
            "surface": {"legend": {"show": False}, "labels": {"values": "all"}},
        },
        {
            "id": "trade-distribution",
            "title": "Trade P&L distribution",
            "subtitle": "1st and 99th percentiles are winsorized for readability",
            "type": "bar",
            "intent": "custom",
            "question": "What is the shape of individual trade outcomes?",
            "rationale": "A pre-binned frequency chart shows skew and tail dependence without exposing all 1,471 individual rows.",
            "dataset": "trade_distribution",
            "sourceId": source["id"],
            "encodings": {
                "x": {
                    "field": "bin_midpoint",
                    "type": "quantitative",
                    "format": "currency",
                    "label": "Trade P&L bin",
                },
                "y": {
                    "field": "trade_count",
                    "type": "quantitative",
                    "format": "number",
                    "label": "Trade count",
                },
                "tooltip": [
                    {"field": "bin_start", "label": "Bin start", "format": "currency"},
                    {"field": "bin_end", "label": "Bin end", "format": "currency"},
                ],
            },
            "xAxisTitle": "Trade P&L",
            "yAxisTitle": "Number of trades",
            "valueFormat": "number",
            "layout": "full",
            "surface": {"legend": {"show": False}, "labels": {"values": "none"}},
        },
    ]

    tables = [
        {
            "id": "strategy-table",
            "title": "Strategy-level holdout results",
            "subtitle": "Exact closed-trade metrics for all seven frozen strategies",
            "dataset": "strategy_results",
            "sourceId": source["id"],
            "layout": "full",
            "density": "spacious",
            "defaultSort": {"field": "net_profit", "direction": "desc"},
            "columns": [
                {"field": "strategy", "label": "Strategy", "type": "text"},
                {"field": "trades", "label": "Trades", "format": "number"},
                {
                    "field": "net_profit",
                    "label": "Net profit",
                    "format": "currency",
                    "movement": True,
                },
                {"field": "profit_factor", "label": "Profit factor", "format": "number"},
                {"field": "win_rate", "label": "Win rate", "format": "percent"},
            ],
        }
    ]

    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Frozen Holdout Portfolio Review",
            "layout": "full",
        },
        {
            "id": "executive-summary",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": (
                "## Executive Summary\n\n"
                f"- **The unseen-data result is economically strong.** The frozen seven-strategy portfolio earned **${trades['pnl'].sum():,.0f}** on **$700,000** of starting capital across **{len(trades):,} trades**.\n\n"
                f"- **It clears the revised candidate threshold.** Daily Sharpe was **{sharpe:.2f}**, above the subsequently adopted **1.50** threshold. The original 2.0 aspiration was not met, and the 1.50 cutoff was selected after this result was visible.\n\n"
                f"- **This is a legitimate research candidate, not yet a production-approved strategy.** Profit factor was **{profit_factor:.2f}**, closed-trade drawdown reached **${abs(trade_max_drawdown):,.0f}**, and **{gc_share:.1%}** of net profit came from GC. Python order/fill parity and the effective transaction-cost configuration remain unresolved."
            ),
        },
        {
            "id": "metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
            "layout": "full",
        },
        {
            "id": "path-heading",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": "## The equity path was profitable, but not smooth\n\nThe portfolio finished materially above its starting value. Read the curve together with drawdown: the endpoint is strong, while the underwater periods show the capital and patience required to realize it.",
        },
        {"id": "equity-block", "type": "chart", "chartId": "equity-curve", "layout": "full"},
        {
            "id": "drawdown-note",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": f"## Drawdown remained meaningful\n\nThe worst closed-trade decline was **${abs(trade_max_drawdown):,.0f}** ({abs(trade_max_drawdown) / INITIAL_CAPITAL:.1%} of initial capital). This is not an intraday mark-to-market drawdown, so live risk could be worse during open positions.",
        },
        {"id": "drawdown-block", "type": "chart", "chartId": "drawdown-curve", "layout": "full"},
        {
            "id": "monthly-note",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": f"## Profitability appeared across the year\n\n**{positive_months} of {len(monthly_pnl)} months** were profitable. Monthly variation remains large, so a short paper-trading window could easily look materially better or worse than the full-year result.",
        },
        {"id": "monthly-block", "type": "chart", "chartId": "monthly-pnl", "layout": "full"},
        {
            "id": "concentration-note",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": f"## GC drove most of the portfolio result\n\nGC contributed **{gc_share:.1%}** of total net profit. CL, ES, HG, and 6J were also profitable, while 6E and NG lost money. The portfolio therefore has cross-market breadth in activity, but much less breadth in profit generation.",
        },
        {
            "id": "contribution-block",
            "type": "chart",
            "chartId": "strategy-contribution",
            "layout": "full",
        },
        {
            "id": "distribution-note",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": "## Trade outcomes contain meaningful tails\n\nThe distribution view shows that performance is not built from identical small wins. Tail wins and losses matter, making exact stop, target, slippage, and gap-handling parity essential before the Python implementation can be trusted.",
        },
        {
            "id": "distribution-block",
            "type": "chart",
            "chartId": "trade-distribution",
            "layout": "full",
        },
        {
            "id": "legitimacy",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Is this a legitimate strategy?\n\n"
                "**Legitimate as a research and paper-trading candidate: yes. Production-approved: no, not yet.** The holdout is genuinely later than the frozen research period, the result is profitable after recorded costs, trade frequency is substantial, and the conservative Sharpe convention remains above 1.5.\n\n"
                "The current blockers are implementation evidence rather than headline performance: the native Python engine does not yet reproduce every entry, protective-order update, and fill; the test requested a $5 size-based commission but the ledger reflects $7 per ordinary round turn; the revised 1.5 threshold was adopted after the holdout was observed; and profit is highly concentrated in GC."
            ),
        },
        {
            "id": "strategy-table-note",
            "type": "markdown",
            "sourceId": source["id"],
            "layout": "full",
            "body": "## Exact strategy-level results\n\nThe table provides the audit values behind the contribution chart. Profit factor near 1.0 for several markets means small execution differences can materially change their standalone result.",
        },
        {
            "id": "strategy-table-block",
            "type": "table",
            "tableId": "strategy-table",
            "layout": "full",
        },
        {
            "id": "next-steps",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Required next steps\n\n"
                "1. Complete Python trade-level parity on development data: timestamp, side, order price, protective updates, exit reason, and P&L.\n"
                "2. Resolve the $5 requested versus $7 observed transaction-cost difference and rerun the frozen test if the reference configuration was not applied as intended.\n"
                "3. Run a 30- to 60-trading-day paper/shadow period with daily signal and fill reconciliation.\n"
                "4. Add contract-roll, exchange-calendar, stale-data, position-limit, and kill-switch controls before any live connection.\n"
                "5. Require independent code and risk approval; do not transmit live orders from the current package."
            ),
        },
        {
            "id": "questions",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Further questions\n\n"
                "- Does GC remain profitable under alternative continuous-contract rolls and doubled slippage?\n"
                "- How much of the portfolio Sharpe remains after capping each strategy's profit contribution or volatility?\n"
                "- Do open-position mark-to-market and intraday drawdowns stay within the firm's limits?\n"
                "- Can the frozen rules survive a forward paper period without parameter changes?"
            ),
        },
        {
            "id": "caveats",
            "type": "markdown",
            "layout": "full",
            "body": (
                "## Caveats and assumptions\n\n"
                f"- Sharpe is **{sharpe:.2f}** when calendar-day zero returns are retained and **{weekday_sharpe:.2f}** on weekdays only. The report uses the lower, more conservative value.\n"
                "- Equity and drawdown are reconstructed from closed-trade P&L; open-position mark-to-market excursions are unavailable in the exported ledger.\n"
                "- This is one one-year holdout on seven continuous futures series, not evidence across multiple independent future regimes.\n"
                "- The report evaluates historical evidence and implementation status; it is not a forecast or investment recommendation."
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Frozen Holdout Portfolio Review",
            "description": "Visual review and validation status for the one-year frozen futures holdout.",
            "generatedAt": generated_at,
            "blocks": blocks,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": [source],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
            "accessIssues": [],
        },
        "sources": [source],
    }

    validation = {
        "assessment": "Share with caveats",
        "question": "Is the frozen unseen-data portfolio economically credible and ready for production implementation?",
        "verified": {
            "trade_count": int(len(trades)),
            "net_profit": float(trades["pnl"].sum()),
            "daily_sharpe_calendar_zeros": float(sharpe),
            "daily_sharpe_weekdays_only": float(weekday_sharpe),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "max_closed_trade_drawdown": trade_max_drawdown,
            "positive_months": positive_months,
            "gc_profit_contribution": gc_share,
            "composite_trade_ids_unique": True,
        },
        "blockers": [
            "Python trade/fill parity is not complete.",
            "Requested commission value and observed ledger cost do not reconcile.",
            "Open-position mark-to-market drawdown is unavailable.",
            "No forward paper/shadow evidence exists yet.",
        ],
        "required_caveats": [
            "The 1.5 threshold was adopted after the holdout result was observed.",
            "GC produced most of portfolio profit.",
            "One year is a single future regime, not repeated independent validation.",
        ],
    }
    return artifact, validation


def main() -> None:
    artifact, validation = build()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (OUTPUT / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(OUTPUT / "artifact.json")


if __name__ == "__main__":
    main()
