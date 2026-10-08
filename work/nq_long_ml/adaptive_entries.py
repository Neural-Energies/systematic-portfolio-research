"""Chronological entry selection with past-only regime filters; research only."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from entry_screen import fdr, incremental_test, matched_control, trade_metrics
from experiment import END, SEED, aggregate, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST

from systematic_research.fixed_hold import fixed_hold_outcomes, select_nonoverlapping

SOURCE = BASE / "entry_runs" / "20261006T164818Z"


def period_scope(outcome, start, end):
    mask = (outcome.index >= start) & (outcome.index < end)
    if "session_end" in outcome:
        # Calendar boundaries can slice overnight sessions. Evaluate complete sessions only.
        mask &= outcome.anchor.ge(start) & outcome.session_end.le(end)
    return outcome.loc[mask]


WINDOW_CACHE = {}


def period_trades(outcome, signal, start, end):
    # Purge all trades whose scheduled label is unavailable before the cutoff.
    key = (id(outcome), start.value, end.value)
    if key not in WINDOW_CACHE:
        scope = period_scope(outcome, start, end).copy()
        scope.loc[scope.planned_exit >= end, "scorable"] = False
        scope["control"] = matched_control(scope)
        WINDOW_CACHE[key] = (outcome, scope)
    scope = WINDOW_CACHE[key][1]
    picks = select_nonoverlapping(
        signal.reindex(scope.index).fillna(False).to_numpy(dtype=bool),
        scope.index.asi8,
        pd.DatetimeIndex(scope.planned_exit).asi8,
    )
    chosen = scope.iloc[picks].copy()
    chosen.loc[chosen.planned_exit >= end, "scorable"] = False
    return chosen


def stable_score(trades, days):
    v = trades.loc[trades.scorable]
    if trades.anchor.nunique() != len(days) or len(v) < 25 or v.anchor.nunique() < 15:
        return -np.inf
    if v.stress_dollars.sum() <= 0 or v.delayed_net_dollars.sum() <= 0:
        return -np.inf
    daily = v.groupby("anchor").net_dollars.sum().reindex(days, fill_value=0)
    if daily.std() <= 0:
        return -np.inf
    # Penalize unstable realized daily returns, not reward the largest total profit.
    return float(daily.mean() / daily.std())


def run():
    out = BASE / "adaptive_entry_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    signals = pd.read_parquet(SOURCE / "signal_events.parquet")
    registry = pd.read_csv(SOURCE / "signals.csv")
    names = registry.signal.tolist()
    for family, group in registry.groupby("family", sort=False):
        name = "family_union_" + family
        signals[name] = signals[group.signal.tolist()].any(axis=1)
        names.append(name)
    nq = aggregate(load_minutes("NQ"), 5)
    es = aggregate(load_minutes("ES"), 5)
    clock = pd.DatetimeIndex(nq.available_at)
    price = pd.Series(nq.close.to_numpy(), index=clock)
    ema = pd.Series(
        nq.close.ewm(span=55, adjust=False, min_periods=55).mean().to_numpy(), index=clock
    )
    es_move = pd.Series(es.close.pct_change(3).to_numpy(), index=pd.DatetimeIndex(es.available_at))
    trend = price.gt(ema).reindex(signals.index).fillna(False)
    confirmation = es_move.gt(0).reindex(signals.index).fillna(False)
    candidates, selections, all_trades, summaries = [], [], [], []
    dates = list(pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))) + [END]
    for session in ("RTH", "Globex"):
        for hold in (60, 120, 240, 360, 1440):
            WINDOW_CACHE.clear()
            outcome = pd.read_parquet(SOURCE / f"outcomes_{session}_60.parquet")
            if hold != 60:
                planned = pd.DatetimeIndex(
                    np.minimum(
                        (outcome.index + pd.Timedelta(minutes=hold)).asi8,
                        pd.DatetimeIndex(outcome.session_end).asi8,
                    ),
                    tz="UTC",
                )
                computed = fixed_hold_outcomes(load_minutes("NQ"), outcome.index, planned)
                for column in computed:
                    outcome[column] = computed[column]
                outcome["capped"] = outcome.effective_minutes.lt(hold)
                outcome["control"] = matched_control(outcome)
            outcome.to_parquet(out / f"outcomes_{session}_{hold}.parquet")
            gates = {
                "unfiltered": pd.Series(True, index=outcome.index),
                "trend_and_es": trend.reindex(outcome.index) & confirmation.reindex(outcome.index),
            }
            selected_trades = []
            for fold, (start, end) in enumerate(zip(dates[:-1], dates[1:], strict=True), 1):
                valid_start = start - pd.DateOffset(months=3)
                train_start = max(outcome.index.min(), start - pd.DateOffset(months=12))
                training_days = pd.DatetimeIndex(
                    period_scope(outcome, train_start, valid_start).anchor.unique()
                )
                validation_days = pd.DatetimeIndex(
                    period_scope(outcome, valid_start, start).anchor.unique()
                )
                ranked = []
                for name in names:
                    for gate, gmask in gates.items():
                        signal = signals[name].reindex(outcome.index).fillna(False) & gmask
                        train = period_trades(outcome, signal, train_start, valid_start)
                        validation = period_trades(outcome, signal, valid_start, start)
                        a, b = (
                            stable_score(train, training_days),
                            stable_score(validation, validation_days),
                        )

                        # Must add value over the same-clock opportunity benchmark in BOTH windows.
                        def incremental(t):
                            v = t.loc[t.scorable & t.control.notna()]
                            return float((v.net_dollars - v.control).mean()) if len(v) else -np.inf

                        ia, ib = incremental(train), incremental(validation)
                        score = min(a, b) if ia > 0 and ib > 0 else -np.inf
                        row = {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "signal": name,
                            "gate": gate,
                            "train_score": a,
                            "validation_score": b,
                            "train_incremental": ia,
                            "validation_incremental": ib,
                            "selection_score": score,
                        }
                        candidates.append(row)
                        if np.isfinite(score):
                            ranked.append((score, name, gate, signal))
                if ranked:
                    score, name, gate, signal = max(ranked, key=lambda x: x[0])
                    trades = period_trades(outcome, signal, start, end)
                    (
                        trades["signal"],
                        trades["gate"],
                        trades["fold"],
                        trades["session"],
                        trades["hold"],
                    ) = name, gate, fold, session, hold
                    selected_trades.append(trades)
                    selections.append(
                        {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "test_start": str(start),
                            "test_end": str(end),
                            "signal": name,
                            "gate": gate,
                            "past_selection_score": score,
                            **trade_metrics(trades),
                        }
                    )
                else:
                    selections.append(
                        {
                            "session": session,
                            "hold": hold,
                            "fold": fold,
                            "test_start": str(start),
                            "test_end": str(end),
                            "signal": "NO QUALIFYING RULE",
                            "gate": "none",
                            "trades": 0,
                            "net_dollars": 0,
                        }
                    )
                print(
                    f"{session} {hold} fold {fold}: {selections[-1]['signal']} "
                    f"{selections[-1]['net_dollars']:.0f}",
                    flush=True,
                )
                pd.DataFrame(selections).to_csv(out / "selections.csv", index=False)
            days = pd.DatetimeIndex(
                pd.concat(
                    [
                        period_scope(outcome, a, b)
                        for a, b in zip(dates[:-1], dates[1:], strict=True)
                    ]
                ).anchor.unique()
            ).sort_values()
            if selected_trades:
                trades = pd.concat(selected_trades).sort_index()
                summary = {
                    "session": session,
                    "hold": hold,
                    **trade_metrics(trades),
                    **incremental_test(trades, days, SEED),
                }
                summary["entry_session_coverage"] = trades.anchor.nunique() / len(days)
                summary["sessions_without_entry"] = len(days) - trades.anchor.nunique()
                q = (
                    trades.loc[trades.scorable]
                    .groupby("fold")
                    .net_dollars.sum()
                    .reindex(range(1, 5), fill_value=0)
                )
                summary["positive_quarters"] = int(q.gt(0).sum())
                summary["net_without_bestquarter"] = float(q.sum() - q.max())
                all_trades.append(trades)
            else:
                summary = {
                    "session": session,
                    "hold": hold,
                    "trades": 0,
                    "net_dollars": 0,
                    "p_hac": 1,
                }
            summaries.append(summary)
    result = pd.DataFrame(summaries)
    result["fdr_q"] = fdr(result.p_hac.fillna(1).to_numpy())
    result.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "candidate_history.csv", index=False)
    if all_trades:
        pd.concat(all_trades).to_parquet(out / "trades.parquet")
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "source": str(SOURCE),
                "candidate_count": 2400,
                "families": 20,
                "rules": "100 original variants and 20 family unions",
                "gates": 2,
                "frequency": (
                    "At least one rule-triggered entry per available session in "
                    "both preceding windows; also report every missed evaluation "
                    "session; no forced entries"
                ),
                "selection": (
                    "12 preceding months: 9 train, 3 validation; quarterly "
                    "freeze; minimum past daily Sharpe; both windows "
                    "stress/delay positive and incremental mean above same-clock "
                    "control"
                ),
                "cutoffs": "No future test outcomes used to select; labels reaching cutoff purged",
                "limitations": (
                    "Reused research data; unverified volume roll; no new "
                    "independent holdout; 2400 candidate variants plus all "
                    "earlier research trials remain part of search history"
                ),
            },
            indent=2,
        )
    )
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    run()
