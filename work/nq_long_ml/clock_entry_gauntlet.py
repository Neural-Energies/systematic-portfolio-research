"""1000 RTH entries scored on past-20-session, opening-up excursion references."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from entry_screen import fdr, incremental_test, matched_control
from experiment import SEED, aggregate, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST
from rth_named_setups import indicators

from systematic_research.auction_entry import entry_path_diagnostics
from systematic_research.clock_excursion import same_clock_history

NAMED = BASE / "rth_named_runs" / "20261006T200105Z"
GENERIC = BASE / "entry_runs" / "20261006T164818Z"


def make_rules(minute, frame, eligible):
    generic = pd.read_parquet(GENERIC / "signal_events.parquet").reindex(
        frame.index, fill_value=False
    )
    named = pd.read_parquet(NAMED / "signals.parquet").reindex(frame.index, fill_value=False)
    named = named.loc[:, ~named.columns.str.startswith("catalog_")]
    bases = pd.concat([generic, named], axis=1)
    bars = aggregate(minute, 5)
    atr, adx, ema = indicators(bars)
    c = bars.close
    o = bars.open
    low = bars.low
    h = bars.high
    rsi = (
        100
        * c.diff().clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        / c.diff().abs().ewm(alpha=1 / 14, adjust=False).mean().replace(0, np.nan)
    )
    bm = bars.index.tz_convert("America/New_York")
    group = bm.normalize()
    volume = bars.volume / bars.volume.shift().rolling(20).mean().replace(0, np.nan)
    vwap = (((h + low + c) / 3) * bars.volume).groupby(group).cumsum() / bars.volume.groupby(
        group
    ).cumsum().replace(0, np.nan)
    # This is NY civil-day VWAP context, explicitly different from RTH-only profile levels.
    contexts = {
        "unfiltered": pd.Series(True, index=bars.index),
        "above_ema20": c.gt(ema),
        "below_ema20": c.lt(ema),
        "above_civil_vwap": c.gt(vwap),
        "below_civil_vwap": c.lt(vwap),
        "bullish_bar": c.gt(o),
        "bearish_bar": c.lt(o),
    }
    for w in (3, 6, 12, 24, 48):
        contexts[f"momentum_{w}_up"] = c.gt(c.shift(w))
        contexts[f"momentum_{w}_down"] = c.lt(c.shift(w))
    for threshold in (30, 40, 50, 60, 70):
        contexts[f"rsi_below_{threshold}"] = rsi.lt(threshold)
        contexts[f"rsi_above_{threshold}"] = rsi.gt(threshold)
    for threshold in (20, 25, 30):
        contexts[f"adx_below_{threshold}"] = adx.lt(threshold)
        contexts[f"adx_above_{threshold}"] = adx.gt(threshold)
    for threshold in (0.75, 1, 1.25, 1.5, 2):
        contexts[f"volume_above_{threshold}"] = volume.gt(threshold)
    contexts["drawdown_1atr"] = h.rolling(20).max().sub(c).ge(atr)
    contexts["drawdown_2atr"] = h.rolling(20).max().sub(c).ge(2 * atr)
    aligned = {}
    for name, value in contexts.items():
        aligned[name] = (
            pd.Series(value.to_numpy(), index=pd.DatetimeIndex(bars.available_at))
            .reindex(frame.index, fill_value=False)
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    local = frame.index.tz_convert("America/New_York")
    slot = local.hour * 60 + local.minute
    for name, a, b in (
        ("morning", 570, 660),
        ("late_morning", 660, 720),
        ("midday", 720, 840),
        ("afternoon", 840, 960),
    ):
        aligned[name] = (slot >= a) & (slot < b)
    pool = [(base, context) for base in bases for context in aligned]
    order = np.random.default_rng(SEED).permutation(len(pool))
    seen = set()
    columns = {}
    registry = []
    discovery = (frame.index < FIRST_TEST) & eligible.to_numpy(dtype=bool)
    for offset in order:
        base, context = pool[int(offset)]
        event = (
            bases[base].fillna(False).to_numpy(dtype=bool)
            & aligned[context]
            & eligible.to_numpy(dtype=bool)
        )
        train = event[discovery]
        if train.sum() < 25:
            continue
        signature = np.packbits(train).tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        name = f"E{len(columns) + 1:04d}"
        columns[name] = event
        registry.append(
            {
                "signal": name,
                "base_rule": base,
                "context": context,
                "past_events_for_definition": int(train.sum()),
            }
        )
        if len(columns) == 1000:
            break
    if len(columns) != 1000:
        raise ValueError(
            f"Only {len(columns)} unique supported rules; expand transparent context pool"
        )
    return pd.DataFrame(columns, index=frame.index), pd.DataFrame(registry), len(pool)


def metrics(event, label, days):
    v = event.loc[event.scorable].copy()
    v["net_dollars"] = v[label]
    stats = incremental_test(v, days, SEED)
    if not len(event):
        return {"events": 0, "success_rate": np.nan, **stats}
    hit = event[label].astype(bool)
    known = event.reach_scorable
    pnl = np.where(hit, event.target_points, (event.exit_close - event.entry_open))
    draw = np.where(hit, event.adverse_before_up_points, event.mae_points)
    capture = np.where(hit, event.target_points / event.mfe_points.replace(0, np.nan), 0)
    q = event.groupby("quarter")[label].mean()
    return {
        "events": len(event),
        "known_outcomes": int(known.sum()),
        "hit_events": int(hit.sum()),
        "success_rate": float(hit.mean()),
        "miss_rate_lower_bound": float((known & ~hit).mean()),
        "unresolved_rate": float((~known).mean()),
        "success_upper_bound": float(hit.mean() + (~known).mean()),
        "session_coverage": event.anchor.nunique() / len(days),
        "sessions_with_signal": event.anchor.nunique(),
        "candidate_sessions": len(days),
        "mean_target_points": float(event.target_points.mean()),
        "median_target_points": float(event.target_points.median()),
        "mean_mae_points": float(event.mae_points.mean()),
        "median_mae_points": float(event.mae_points.median()),
        "mean_mae_percent": float((100 * event.mae_points / event.entry_open).mean()),
        "mean_drawdown_until_hypothetical_exit": float(np.nanmean(draw)),
        "mean_points_to_hypothetical_exit": float(np.nanmean(pnl)),
        "mean_net_dollars_assumed_cost": float(np.nanmean(pnl) * 20 - 25),
        "mean_target_capture_fraction": float(np.nanmean(capture)),
        "mean_missed_target_fraction": float(
            (
                event.loc[known & ~hit, "mfe_points"] / event.loc[known & ~hit, "target_points"]
            ).mean()
        ),
        "median_history_samples": float(event.retained_samples.median()),
        "minimum_quarter_hit_rate": float(q.min()),
        **stats,
    }


def run():
    out = BASE / "clock_entry_runs_1000" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    minute = load_minutes("NQ")
    frame = pd.read_parquet(GENERIC / "outcomes_RTH_60.parquet")
    frame = frame.loc[
        frame.index.tz_convert("America/New_York").strftime("%Y-%m-%d") != "2025-01-09"
    ].copy()
    phase = (frame.slot - 570) % 60
    frame["previous_hour_stream_open"] = frame.groupby(phase, sort=False).entry_open.shift()
    frame["opening_up"] = frame.entry_open.gt(frame.previous_hour_stream_open)
    frame["excursion"] = frame.mfe_points
    sessions = pd.DatetimeIndex(frame.anchor.unique()).sort_values()
    histories = {}
    for trim in (True, False):
        name = "iqr" if trim else "raw"
        stats = same_clock_history(frame, sessions, trim_iqr=trim)
        histories[name] = stats
        stats.to_parquet(out / f"historical_statistics_{name}.parquet")
        print(
            "HISTORY",
            name,
            "median eligible",
            stats.qualified_samples.median(),
            "median retained",
            stats.retained_samples.median(),
            flush=True,
        )
    mature = histories["iqr"].q70.notna() & histories["raw"].q70.notna()
    eligible = frame.opening_up & mature
    signals, registry, pool = make_rules(minute, frame, eligible)
    registry.to_csv(out / "rules.csv", index=False)
    signals.to_parquet(out / "signals.parquet")
    frame.to_parquet(out / "opportunities.parquet")
    print(
        "BUILT 1000 distinct rules from",
        pool,
        "definitions; opening-up eligible",
        int(eligible.sum()),
        flush=True,
    )
    rows = []
    baseline = []
    for trim, history in histories.items():
        for reference, column in (("p70", "q70"), ("mean70", "mean70")):
            usable = eligible & history[column].gt(0)
            scope = frame.loc[usable].copy()
            scope["target_points"] = np.ceil(history.loc[usable, column] * 4) / 4
            scope["retained_samples"] = history.loc[usable, "retained_samples"]
            paths = entry_path_diagnostics(
                minute,
                scope.index,
                pd.DatetimeIndex(scope.planned_exit),
                scope.target_points.to_numpy(dtype=float),
            )
            scope = scope.join(paths)
            scope["hit"] = scope.reached_up.fillna(0)
            # Keep unresolved outcomes in the conservative hit lower bound.
            # Never erase failed test outliers.
            scope["scorable"] = True
            comparison = scope.copy()
            comparison["gross_dollars"] = scope.hit + 25
            scope["control"] = matched_control(comparison)
            scope.to_parquet(out / f"outcomes_{trim}_{reference}.parquet")
            for split in ("development", "later_test"):
                s = scope.loc[scope.split.eq(split)]
                all_day_frame = frame.loc[frame.split.eq(split) & mature]
                days = pd.DatetimeIndex(all_day_frame.anchor.unique()).sort_values()
                baseline.append(
                    {
                        "trim": trim,
                        "reference": reference,
                        "split": split,
                        **metrics(s, "hit", days),
                    }
                )
                for name in signals:
                    event = s.loc[signals[name].reindex(s.index)]
                    rows.append(
                        {
                            "signal": name,
                            "trim": trim,
                            "reference": reference,
                            "split": split,
                            **metrics(event, "hit", days),
                        }
                    )
                print(trim, reference, split, "1000 scored", flush=True)
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
                pd.DataFrame(baseline).to_csv(out / "ordinary_opening_up_baseline.csv", index=False)
    results = pd.DataFrame(rows)
    later = results.split.eq("later_test")
    results.loc[later, "fdr_q"] = fdr(results.loc[later, "p_hac"].fillna(1).to_numpy())
    results.to_csv(out / "results.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "scope": (
                    "NQ long-only RTH; 60-minute forward window, session close cap; one-hour "
                    "rolling windows at each five-minute signal boundary"
                ),
                "open_filter": (
                    "Both current entries and historical excursion samples require current open "
                    "> prior observed hourly-stream open. Each five-minute clock offset has its "
                    "own chronological hour stream; first RTH hour compares to the last "
                    "observed preceding-session hourly bar of that offset."
                ),
                "history": (
                    "Exactly previous 20 trading-session positions at identical NY clock; only "
                    "opening-up, complete observed excursions retained; at least five "
                    "qualified/retained samples. Excludes current session. Never backextend to "
                    "20 eligible observations."
                ),
                "variable": "high minus open, not high minus low",
                "summaries": (
                    "Raw and IQR-trimmed mean, median, sample SD, quartiles, P10/P70/P90, "
                    "sample/outlier counts and fences"
                ),
                "outliers": (
                    "Past-only Q1-1.5IQR / Q3+1.5IQR. Future outcomes never removed for being "
                    "outliers."
                ),
                "references": (
                    "P70 and 0.70*mean, both rounded upward to NQ quarter-point ticks; no "
                    "trained high/stop models"
                ),
                "hypothetical_exit": (
                    "Reference price if touched; otherwise terminal observed close. No stops. "
                    "MAE full window and before hypothetical exit; losses and unobservable "
                    "outcomes disclosed; modeled $25 round-trip cost only secondary"
                ),
                "signals": 1000,
                "candidate_definitions": pool,
                "selection": (
                    "Fixed-seed order; remove exact duplicate event streams on earlier eligible "
                    "development data only; require >=25 earlier events; no performance ranking "
                    "for inclusion. Entries are variations of 100 generic and 56 named concepts "
                    "with transparent known-at-entry context filters."
                ),
                "statistics": (
                    "Session-block diagnostics; FDR across all 4000 later tests (1000 rules x 2 "
                    "references x 2 histories); same-clock/quarter/volatility matched "
                    "opening-up entries"
                ),
                "limitations": (
                    "Reused development data; unverified NQ volume roll; approximate profile "
                    "signals; overlapping research events, not independent trades; no live "
                    "execution or validated trading system"
                ),
                "iqr_source": "https://www.itl.nist.gov/div898/handbook/eda/section3/boxplot.htm",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    run()
