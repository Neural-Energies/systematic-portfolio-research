"""Entry-only event study: forward direction and excursions, no trading policy."""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from adaptive_entries import SOURCE, period_scope
from clock_entries import clock_signal
from entry_screen import fdr, incremental_test, matched_control
from experiment import END, SEED, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST

from systematic_research.fixed_hold import fixed_hold_outcomes


def event_metrics(frame, days):
    valid = frame.loc[frame.scorable].copy()
    if not len(valid):
        return {
            "events": 0,
            "direction_success": np.nan,
            "session_coverage": 0,
            "sessions_without_signal": len(days),
            "p_hac": 1,
        }
    up = valid.exit_close.gt(valid.entry_open).astype(float)
    probe = valid.copy()
    probe["net_dollars"] = up
    stats = incremental_test(probe, days, SEED)
    daily = (
        pd.DataFrame({"up": up, "count": 1, "day": valid.anchor})
        .groupby("day")[["up", "count"]]
        .sum()
        .reindex(days, fill_value=0)
    )
    rng = np.random.default_rng(SEED)
    n = len(daily)
    starts = rng.integers(0, n, (2000, int(np.ceil(n / 10))))
    idx = ((starts[:, :, None] + np.arange(10)) % n).reshape(2000, -1)[:, :n]
    sample = daily.to_numpy()[idx].sum(axis=1)
    rates = sample[:, 0] / np.where(sample[:, 1] > 0, sample[:, 1], np.nan)
    return {
        "events": len(valid),
        "unscorable_events": len(frame) - len(valid),
        "direction_success": float(up.mean()),
        "success_ci_low": float(np.nanquantile(rates, 0.025)),
        "success_ci_high": float(np.nanquantile(rates, 0.975)),
        "matched_base_success": float(valid.control.mean()),
        "session_coverage": frame.anchor.nunique() / len(days),
        "sessions_without_signal": len(days) - frame.anchor.nunique(),
        "median_forward_points": float((valid.exit_close - valid.entry_open).median()),
        "mean_forward_points": float((valid.exit_close - valid.entry_open).mean()),
        "median_favorable_points": float(valid.mfe_points.median()),
        "median_adverse_points": float(valid.mae_points.median()),
        "favorable_1atr_rate": float(valid.mfe_points.ge(valid.atr).mean()),
        "favorable_exceeds_adverse_rate": float(valid.mfe_points.gt(valid.mae_points).mean()),
        **stats,
    }


def run(windows=(5, 15, 60, 240)):
    minute = load_minutes("NQ")
    out = BASE / "entry_quality_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    signals = pd.read_parquet(SOURCE / "signal_events.parquet")
    registry = pd.read_csv(SOURCE / "signals.csv")
    for family, g in registry.groupby("family", sort=False):
        signals["family_union_" + family] = signals[g.signal.tolist()].any(axis=1)
    rows = []
    for session in ("RTH", "Globex"):
        for horizon in windows:
            original = pd.read_parquet(
                SOURCE
                / f"outcomes_{session}_{horizon if horizon in (5, 15, 60, 240) else 60}.parquet"
            )
            if horizon not in (5, 15, 60, 240):
                planned = pd.DatetimeIndex(
                    np.minimum(
                        (original.index + pd.Timedelta(minutes=horizon)).asi8,
                        pd.DatetimeIndex(original.session_end).asi8,
                    ),
                    tz="UTC",
                )
                measured = fixed_hold_outcomes(minute, original.index, planned)
                for column in measured:
                    original[column] = measured[column]
                original["capped"] = original.effective_minutes.lt(horizon)
                original.to_parquet(out / f"outcomes_{session}_{horizon}.parquet")
            for split, start, end in (
                ("earlier", original.index.min(), FIRST_TEST),
                ("later", FIRST_TEST, END),
            ):
                scope = period_scope(original, start, end).copy()
                scope.loc[scope.planned_exit >= end, "scorable"] = False
                days = pd.DatetimeIndex(scope.anchor.unique()).sort_values()
                # Same-clock comparison uses direction labels, not dollars or hypothetical exits.
                control_frame = scope.copy()
                control_frame["gross_dollars"] = (
                    scope.exit_close.gt(scope.entry_open).astype(float) + 25
                )
                scope["control"] = matched_control(control_frame)
                checks = signals.reindex(scope.index).copy()
                for offset in (5, 15, 30, 60, 90, 120, 180, 240, 300):
                    checks[f"clock_offset_{offset}"] = clock_signal(scope, offset)
                for name in checks:
                    frame = scope.loc[checks[name].fillna(False)]
                    record = {
                        "session": session,
                        "horizon": horizon,
                        "split": split,
                        "signal": name,
                        "candidate_sessions": len(days),
                        **event_metrics(frame, days),
                    }
                    if len(frame.loc[frame.scorable]):
                        q = (
                            frame.loc[frame.scorable]
                            .groupby("quarter")
                            .apply(
                                lambda x: float(x.exit_close.gt(x.entry_open).mean()),
                                include_groups=False,
                            )
                        )
                        record["minimum_quarter_success"] = float(q.min())
                        record["quarters_at_least_60pct"] = int(q.ge(0.60).sum())
                    rows.append(record)
                print(session, horizon, split, "done", flush=True)
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    results = pd.DataFrame(rows)
    later = results.split.eq("later")
    results.loc[later, "fdr_q"] = fdr(results.loc[later, "p_hac"].fillna(1).to_numpy())
    results["entry_candidate_pass"] = False
    earlier = results.loc[results.split.eq("earlier")].set_index(["session", "horizon", "signal"])
    z = results.loc[later]
    prior = earlier.reindex(pd.MultiIndex.from_frame(z[["session", "horizon", "signal"]]))
    valid = (
        z.sessions_without_signal.eq(0)
        & z.events.ge(100)
        & z.direction_success.ge(0.60)
        & z.success_ci_low.gt(z.matched_base_success)
        & z.incremental_ci_low.gt(0)
        & z.fdr_q.le(0.05)
        & z.minimum_quarter_success.ge(0.55)
        & (z.events / (z.events + z.unscorable_events)).ge(0.95)
        & (prior.sessions_without_signal.to_numpy() == 0)
        & (prior.direction_success.to_numpy() >= 0.55)
    )
    results.loc[z.index, "entry_candidate_pass"] = valid.to_numpy()
    results.to_csv(out / "results.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "objective": (
                    "ENTRY ONLY: determine whether NQ rises after the signal and "
                    "describe favorable/adverse paths; no targets, stops, exit "
                    "optimization, portfolio or PnL selection"
                ),
                "rules": 120,
                "clock_controls": 9,
                "windows_minutes": list(windows),
                "success": (
                    "Observed last price in the window strictly above entry "
                    "open; no target-hit accuracy claim"
                ),
                "excursions": (
                    "Maximum favorable/adverse points and favorable >=1 lagged "
                    "ATR are diagnostics, not executable target/stop rules"
                ),
                "events": (
                    "All signal timestamps evaluated, including overlapping "
                    "windows; block bootstrap clusters by session to avoid "
                    "pretending repeated signals are independent trades"
                ),
                "frequency": (
                    "Every complete available session must contain at least one "
                    "signal, separately RTH and Globex"
                ),
                "qualification": (
                    "100 scored events; every session; later direction >=60%; "
                    "lower CI above matched baseline and positive incremental "
                    "CI; FDR<=.05 within this screen; worst quarter>=55%; >=95% "
                    "scored events; earlier every-session coverage and >=55% "
                    "direction"
                ),
                "limits": (
                    "Previously examined development data and earlier trial "
                    "selection; screen-level FDR does not erase entire research "
                    "history; legacy continuous NQ roll unverified; no "
                    "independent validation yet"
                ),
            },
            indent=2,
        )
    )
    print("COMPLETE", out, "passed", int(valid.sum()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", nargs="+", type=int, default=[5, 15, 60, 240])
    run(tuple(parser.parse_args().windows))
