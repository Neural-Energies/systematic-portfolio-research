"""Prespecified recurring session entry timing, independent of future prices."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from adaptive_entries import SOURCE, period_scope, period_trades, stable_score
from entry_screen import fdr, incremental_test, trade_metrics
from experiment import END, SEED, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST

from systematic_research.fixed_hold import fixed_hold_outcomes


def clock_signal(frame, offset):
    elapsed = (frame.index - pd.DatetimeIndex(frame.anchor)).total_seconds() / 60
    eligible = pd.Series((elapsed >= offset) & (elapsed < offset + 15), index=frame.index)
    # First known quote in a narrow scheduled window. No future-price lookup.
    count = eligible.astype(int).groupby(frame.anchor).cumsum()
    return eligible & count.eq(1)


def run():
    out = BASE / "clock_entry_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    minute = load_minutes("NQ")
    rows = []
    selections = []
    alltrades = []
    evaluations = []
    dates = list(pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))) + [END]
    offsets = (5, 15, 30, 60, 90, 120, 180, 240, 300)
    for session in ("RTH", "Globex"):
        base = pd.read_parquet(SOURCE / f"outcomes_{session}_60.parquet")
        for hold in (60, 120, 240, 360, 1440):
            outcome = base.copy()
            if hold != 60:
                planned = pd.DatetimeIndex(
                    np.minimum(
                        (outcome.index + pd.Timedelta(minutes=hold)).asi8,
                        pd.DatetimeIndex(outcome.session_end).asi8,
                    ),
                    tz="UTC",
                )
                computed = fixed_hold_outcomes(minute, outcome.index, planned)
                for column in computed:
                    outcome[column] = computed[column]
                outcome["capped"] = outcome.effective_minutes.lt(hold)
            signals = {offset: clock_signal(outcome, offset) for offset in offsets}
            held = []
            for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True), 1):
                valid_start = start - pd.DateOffset(months=3)
                train_start = max(outcome.index.min(), start - pd.DateOffset(months=12))
                train_days = pd.DatetimeIndex(
                    period_scope(outcome, train_start, valid_start).anchor.unique()
                )
                valid_days = pd.DatetimeIndex(
                    period_scope(outcome, valid_start, start).anchor.unique()
                )
                rank = []
                for offset, signal in signals.items():
                    a = period_trades(outcome, signal, train_start, valid_start)
                    b = period_trades(outcome, signal, valid_start, start)
                    score = min(stable_score(a, train_days), stable_score(b, valid_days))
                    rows.append(
                        {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "offset": offset,
                            "score": score,
                            "train_net": trade_metrics(a)["net_dollars"],
                            "validation_net": trade_metrics(b)["net_dollars"],
                        }
                    )
                    if np.isfinite(score):
                        rank.append((score, offset))
                if rank:
                    score, offset = max(rank)
                    trades = period_trades(outcome, signals[offset], start, end)
                    trades["session"], trades["hold"], trades["fold"], trades["offset"] = (
                        session,
                        hold,
                        fold,
                        offset,
                    )
                    held.append(trades)
                    alltrades.append(trades)
                    selections.append(
                        {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "offset": offset,
                            **trade_metrics(trades),
                        }
                    )
                else:
                    selections.append(
                        {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "offset": -1,
                            "trades": 0,
                            "net_dollars": 0,
                        }
                    )
            days = pd.DatetimeIndex(
                pd.concat(
                    [
                        period_scope(outcome, a, b)
                        for a, b in zip(dates[:-1], dates[1:], strict=True)
                    ]
                ).anchor.unique()
            ).sort_values()
            if held:
                trades = pd.concat(held).sort_index()
                q = (
                    trades.loc[trades.scorable]
                    .groupby("fold")
                    .net_dollars.sum()
                    .reindex(range(1, 5), fill_value=0)
                )
                evaluations.append(
                    {
                        "session": session,
                        "hold": hold,
                        **trade_metrics(trades),
                        **incremental_test(trades, days, SEED),
                        "sessions_without_entry": len(days) - trades.anchor.nunique(),
                        "candidate_days": len(days),
                        "positive_quarters": int(q.gt(0).sum()),
                        "net_without_bestquarter": float(q.sum() - q.max()),
                    }
                )
            else:
                evaluations.append(
                    {
                        "session": session,
                        "hold": hold,
                        "trades": 0,
                        "net_dollars": 0,
                        "p_hac": 1,
                        "sessions_without_entry": len(days),
                        "candidate_days": len(days),
                    }
                )
            print(
                session,
                hold,
                evaluations[-1]["net_dollars"],
                evaluations[-1]["sessions_without_entry"],
                flush=True,
            )
            pd.DataFrame(evaluations).to_csv(out / "summary.csv", index=False)
    result = pd.DataFrame(evaluations)
    result["fdr_q"] = fdr(result.p_hac.fillna(1).to_numpy())
    result.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "candidate_history.csv", index=False)
    pd.DataFrame(selections).to_csv(out / "selections.csv", index=False)
    if alltrades:
        pd.concat(alltrades).to_parquet(out / "trades.parquet")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "offsets": offsets,
                "holds": [60, 120, 240, 360, 1440],
                "selection": (
                    "past 9 month train / 3 month validation, require every "
                    "session and positive stress/delay in both; freeze next "
                    "quarter"
                ),
                "purpose": (
                    "Recurrence benchmark: whether scheduled daily timing "
                    "survives costs and chronology; this is not a price/volume "
                    "edge"
                ),
                "frequency": "Every available RTH/Globex session separately",
                "data": (
                    "Reused development data; unverified contract volume "
                    "rollover; not a fresh holdout"
                ),
            },
            indent=2,
        )
    )
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    run()
