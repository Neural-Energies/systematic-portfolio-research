"""RTH entry-only approximations of public auction and price-action concepts."""

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from adaptive_entries import period_scope
from entry_screen import fdr, incremental_test, matched_control, session_metadata
from experiment import END, SEED, aggregate, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST

from systematic_research.auction_entry import (
    entry_path_diagnostics,
    profile_histogram,
    profile_levels,
    wilder_mean,
)

CLOSED_RTH_DATES = {"2025-01-09"}  # CME equity-index close at 09:30 NY; no RTH.


def indicators(bars):
    c, h, low = bars.close, bars.high, bars.low
    tr = pd.concat([h - low, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    up = h.diff()
    down = -low.diff()
    plus = wilder_mean(up.where((up > down) & (up > 0), 0), 14)
    minus = wilder_mean(down.where((down > up) & (down > 0), 0), 14)
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    adx = wilder_mean(dx, 14)
    return atr, adx, c.ewm(span=20, adjust=False, min_periods=20).mean()


def build_setups(minutes):
    bars = aggregate(minutes, 5)
    clock = pd.DatetimeIndex(bars.available_at)
    bm = session_metadata(bars.index)
    valid = bm.kind.eq("RTH") & session_metadata(clock).kind.eq("RTH").to_numpy()
    valid &= ~bars.index.tz_convert("America/New_York").strftime("%Y-%m-%d").isin(CLOSED_RTH_DATES)
    atr, adx, ema = indicators(bars)
    r = bars.loc[valid].copy()
    anchors = bm.loc[valid, "anchor"]
    minute_meta = session_metadata(minutes.index)
    rth_minutes = minutes.loc[
        minute_meta.kind.eq("RTH")
        & ~minutes.index.tz_convert("America/New_York").strftime("%Y-%m-%d").isin(CLOSED_RTH_DATES)
    ]
    groups = list(rth_minutes.groupby(minute_meta.loc[rth_minutes.index, "anchor"]))
    day_index = pd.DatetimeIndex([a for a, _ in groups])
    daily = pd.DataFrame(
        {
            "high": [g.high.max() for _, g in groups],
            "low": [g.low.min() for _, g in groups],
            "open": [g.open.iloc[0] for _, g in groups],
            "close": [g.close.iloc[-1] for _, g in groups],
        },
        index=day_index,
    )
    previous = daily.shift().reindex(pd.DatetimeIndex(anchors))
    previous.index = r.index
    c, o, h, low = r.close, r.open, r.high, r.low
    previous_full_close = bars.close.shift().reindex(r.index)
    a = atr.reindex(r.index)
    e = ema.reindex(r.index)
    span = (h - low).replace(0, np.nan)
    position = (c - low) / span
    matrix = {}
    registry = []
    levels_saved = []

    def add(name, source, rule, definition, profile=False):
        matrix[name] = rule.reindex(r.index).fillna(False).astype(bool)
        registry.append(
            {
                "signal": name,
                "source": source,
                "definition": definition,
                "approximate_profile": profile,
            }
        )

    for width in (1.0, 2.0, 4.0):
        for allocation in ("uniform", "close"):
            rows = [
                profile_levels(profile_histogram(g, width, allocation), width) for _, g in groups
            ]
            prior = (
                pd.DataFrame(rows, index=day_index, columns=["poc", "val", "vah"])
                .shift()
                .reindex(pd.DatetimeIndex(anchors))
            )
            prior.index = r.index
            suffix = f"{int(width)}pt_{allocation}"
            for key in ("poc", "val", "vah"):
                levels_saved.append(
                    pd.DataFrame({"level": prior[key], "name": key + "_" + suffix}, index=r.index)
                )
            add(
                "vp_val_reclaim_" + suffix,
                "FT71-inspired",
                low.le(prior.val) & c.gt(prior.val) & previous_full_close.le(prior.val) & c.gt(o),
                "Prior RTH approximate VAL reclaimed on a completed bullish bar",
                True,
            )
            add(
                "vp_poc_reclaim_" + suffix,
                "FT71-inspired",
                low.le(prior.poc) & c.gt(prior.poc) & previous_full_close.le(prior.poc) & c.gt(o),
                "Prior RTH approximate VPOC reclaimed from below",
                True,
            )
            inside = c.between(prior.val, prior.vah)
            add(
                "vp_value_acceptance_" + suffix,
                "FT71-inspired",
                inside
                & inside.groupby(anchors).shift(fill_value=False)
                & c.groupby(anchors).shift(2).lt(prior.val)
                & c.gt(o),
                "Two completed closes inside prior value after being below it",
                True,
            )
            breakout = (
                c.groupby(anchors)
                .transform(lambda x: x.shift().rolling(6, min_periods=1).max())
                .gt(prior.vah)
            )
            add(
                "vp_vah_retest_" + suffix,
                "FT71-inspired",
                breakout & low.le(prior.vah) & c.gt(prior.vah) & c.gt(o),
                "Prior value-area-high upside breakout followed by bullish retest",
                True,
            )
            session_open = daily.open.reindex(pd.DatetimeIndex(anchors))
            session_open.index = r.index
            add(
                "vp_open_rejection_" + suffix,
                "FT71-inspired",
                session_open.lt(prior.val)
                & c.gt(prior.val)
                & previous_full_close.le(prior.val)
                & c.gt(o),
                "RTH opened below prior value then reclaimed VAL",
                True,
            )
    for allocation in ("uniform", "close"):
        developing = pd.DataFrame(index=r.index, columns=["poc", "val", "vah"], dtype=float)
        for anchor, g in groups:
            hist = {}
            times = r.index[anchors.eq(anchor)]
            for stamp in times:
                chunk = g.loc[stamp : stamp + pd.Timedelta(minutes=4)]
                addition = profile_histogram(chunk, 2.0, allocation)
                for key, value in addition.items():
                    hist[key] = hist.get(key, 0) + value
                developing.loc[stamp] = profile_levels(hist, 2.0)
        previous_dev = developing.groupby(anchors).shift()
        add(
            "vp_developing_poc_" + allocation,
            "FT71-inspired",
            c.groupby(anchors).shift().le(previous_dev.poc) & c.gt(previous_dev.poc) & c.gt(o),
            "Reclaim previous completed developing 2-point-bin VPOC",
            True,
        )
        add(
            "vp_developing_val_" + allocation,
            "FT71-inspired",
            low.le(previous_dev.val) & c.gt(previous_dev.val) & c.gt(o),
            "Bullish rejection at previous developing VAL",
            True,
        )
    add(
        "auction_prior_low_reclaim",
        "Auction",
        low.lt(previous.low) & c.gt(previous.low) & c.gt(o),
        "Failed downside auction below previous RTH low",
    )
    overnight = {}
    for anchor, _ in groups:
        start = anchor - pd.Timedelta(hours=15.5)
        g = minutes.loc[start : anchor - pd.Timedelta(minutes=1)]
        overnight[anchor] = g.low.min() if len(g) else np.nan
    on_low = anchors.map(overnight)
    add(
        "auction_overnight_low_reclaim",
        "Auction",
        low.lt(on_low) & c.gt(on_low) & c.gt(o),
        "Reclaim the already completed overnight low",
    )
    for duration in (30, 60):
        initial = pd.DataFrame(index=r.index, columns=["high", "low"], dtype=float)
        for anchor, g in r.groupby(anchors):
            chunk = g.loc[g.index < anchor + pd.Timedelta(minutes=duration)]
            if len(chunk) == duration // 5:
                ready = g.index >= anchor + pd.Timedelta(minutes=duration)
                initial.loc[g.index[ready], "high"] = chunk.high.max()
                initial.loc[g.index[ready], "low"] = chunk.low.min()
        add(
            f"auction_ib_{duration}_low_reclaim",
            "Auction",
            low.lt(initial.low) & c.gt(initial.low) & c.gt(o),
            "Failed downside break of completed initial balance",
        )
        add(
            f"auction_ib_{duration}_high_retest",
            "Auction",
            c.groupby(anchors)
            .transform(lambda x: x.shift().rolling(6, min_periods=1).max())
            .gt(initial.high)
            & low.le(initial.high)
            & c.gt(initial.high)
            & c.gt(o),
            "Completed initial-balance upside breakout/retest",
        )
    for tf in (5, 15):
        b = aggregate(minutes, tf)
        at, ax, em = indicators(b)
        bc = b.close
        osc = bc.rolling(3).mean() - bc.rolling(10).mean()
        slow = osc.rolling(16).mean()
        previous_high = b.high.shift().rolling(20).max()
        previous_ema = em.shift(3)
        for threshold in (20, 25, 30):
            armed = False
            age = 0
            flags = []
            for i in range(len(b)):
                if (
                    ax.iloc[i] > threshold
                    and bc.iloc[i] > previous_high.iloc[i]
                    and em.iloc[i] > previous_ema.iloc[i]
                ):
                    armed = True
                    age = 0
                age += 1
                touch = b.low.iloc[i] <= em.iloc[i] <= b.high.iloc[i]
                hit = (
                    armed
                    and age <= 32
                    and touch
                    and bc.iloc[i] > em.iloc[i]
                    and bc.iloc[i] > b.open.iloc[i]
                )
                flags.append(hit)
                if touch or age > 32:
                    armed = False
            rule = pd.Series(flags, index=pd.DatetimeIndex(b.available_at)).reindex(
                pd.DatetimeIndex(r.available_at), fill_value=False
            )
            rule.index = r.index
            add(
                f"raschke_grail_{tf}m_adx{threshold}",
                "Raschke-inspired",
                rule,
                (
                    "First EMA20 pullback after strong-trend breakout; "
                    "completed-bar confirmation, not an intrabar stop entry"
                ),
            )
        recovery = (
            osc.gt(osc.shift())
            & osc.shift().le(osc.shift(2))
            & osc.lt(slow)
            & slow.gt(0)
            & bc.gt(em)
        )
        rule = pd.Series(recovery.to_numpy(), index=pd.DatetimeIndex(b.available_at)).reindex(
            pd.DatetimeIndex(r.available_at), fill_value=False
        )
        rule.index = r.index
        add(
            f"raschke_310_pullback_{tf}m",
            "Raschke-inspired",
            rule,
            "3-10 SMA oscillator turns upward below positive 16-SMA slow line in an uptrend",
        )
    priorlow = bars.low.shift().rolling(20).min().reindex(r.index)
    age_of_low = (
        bars.low.shift()
        .rolling(20)
        .apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)
        .reindex(r.index)
    )
    add(
        "raschke_turtle_soup_intraday",
        "Raschke-inspired",
        low.lt(priorlow) & c.gt(priorlow) & age_of_low.ge(4) & c.gt(o),
        "Intraday adaptation of failed 20-bar low; prior low at least four bars old",
    )
    add(
        "raschke_upper_close_tendency",
        "Raschke-inspired",
        position.ge(0.8) & c.gt(o),
        "Upper-20-percent close tendency adapted to completed five-minute bars",
    )
    h2 = pd.Series(False, index=r.index)
    wedges = pd.Series(False, index=r.index)
    double = pd.Series(False, index=r.index)
    previous_e = e.shift(3)
    for _, g in r.groupby(anchors):
        stage = 0
        count = 0
        age = 0
        pivots = []
        for j in range(2, len(g)):
            stamp = g.index[j]
            prev = g.iloc[j - 1]
            cur = g.iloc[j]
            before = g.iloc[j - 2]
            if prev.low < before.low and prev.low <= cur.low:
                pivots.append((j - 1, float(prev.low)))
            pivots = pivots[-4:]
            reversal = cur.close > cur.open and cur.close > prev.high
            if (
                len(pivots) >= 3
                and pivots[-3][1] > pivots[-2][1] > pivots[-1][1]
                and j - pivots[-3][0] <= 18
                and reversal
            ):
                wedges.loc[stamp] = True
            if (
                len(pivots) >= 2
                and pivots[-1][0] - pivots[-2][0] >= 3
                and abs(pivots[-1][1] - pivots[-2][1]) <= 0.5 * a.loc[stamp]
                and reversal
            ):
                double.loc[stamp] = True
            uptrend = cur.close > e.loc[stamp] and e.loc[stamp] > previous_e.loc[stamp]
            age += 1
            if age > 18 or cur.close < e.loc[stamp] - 2 * a.loc[stamp]:
                stage = 0
                count = 0
            if stage == 0 and uptrend and cur.low < prev.low:
                stage = 1
                age = 0
            if stage == 1 and cur.high > prev.high:
                count += 1
                stage = 2
            elif stage == 2 and cur.low < prev.low:
                stage = 1
            if count >= 2 and reversal and uptrend:
                h2.loc[stamp] = True
                stage = 0
                count = 0
    add(
        "brooks_high2_pullback",
        "Brooks-inspired",
        h2,
        "Causal second upside attempt after a two-legged pullback in an EMA20 uptrend",
    )
    add(
        "brooks_wedge_reversal",
        "Brooks-inspired",
        wedges,
        "Three descending one-bar-confirmed lows, followed by bullish close above previous high",
    )
    add(
        "brooks_double_bottom",
        "Brooks-inspired",
        double,
        (
            "Two confirmed lows within half ATR, separated by at least three "
            "bars; bullish reversal confirmation"
        ),
    )
    for lookback in (6, 12, 24):
        strong = position.ge(0.8) & (c - o).ge(0.5 * a) & c.gt(h.shift().rolling(lookback).max())
        add(
            f"brooks_breakout_followthrough_{lookback}",
            "Brooks-inspired",
            strong.groupby(anchors).shift(fill_value=False) & c.gt(h.shift()) & c.gt(o),
            "Strong bull breakout followed by a second completed bullish follow-through bar",
        )
    result = pd.DataFrame(matrix, index=r.index)
    result.index = pd.DatetimeIndex(r.available_at)
    for source in ("FT71-inspired", "Auction", "Raschke-inspired", "Brooks-inspired"):
        names = [v["signal"] for v in registry if v["source"] == source]
        result["catalog_" + source] = result[names].any(axis=1)
        registry.append(
            {
                "signal": "catalog_" + source,
                "source": source,
                "definition": (
                    "Predefined union of all this family's setups; no outcome-selected combination"
                ),
                "approximate_profile": source == "FT71-inspired",
            }
        )
    result["catalog_all_named"] = result.any(axis=1)
    registry.append(
        {
            "signal": "catalog_all_named",
            "source": "Combined",
            "definition": "Predefined union of all studied setups",
            "approximate_profile": True,
        }
    )
    meta = session_metadata(result.index)
    meta["atr"] = a.to_numpy()
    meta["prior_profile_available"] = previous.low.notna().to_numpy()
    return result, meta, pd.DataFrame(registry), pd.concat(levels_saved)


def run():
    out = BASE / "rth_named_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    minutes = load_minutes("NQ")
    signals, meta, registry, levels = build_setups(minutes)
    keep = (
        meta.prior_profile_available
        & meta.atr.notna()
        & minutes.open.reindex(signals.index).notna()
    )
    signals, meta = signals.loc[keep], meta.loc[keep]
    signals.to_parquet(out / "signals.parquet")
    meta.to_parquet(out / "metadata.parquet")
    registry.to_csv(out / "registry.csv", index=False)
    levels.to_parquet(out / "profile_levels.parquet")
    print("BUILT", len(registry), "setups", len(signals), "RTH timestamps", flush=True)
    rows = []
    for horizon in (5, 15, 60, 240):
        original = pd.read_parquet(
            BASE / "entry_runs" / "20261006T164818Z" / f"outcomes_RTH_{horizon}.parquet"
        ).reindex(signals.index)
        original["atr"] = meta.atr
        for distance_name, distance in (
            ("10points", np.full(len(original), 10.0)),
            ("20points", np.full(len(original), 20.0)),
            ("1atr", meta.atr.to_numpy(dtype=float)),
        ):
            paths = entry_path_diagnostics(
                minutes, original.index, pd.DatetimeIndex(original.planned_exit), distance
            )
            paths.to_parquet(out / f"paths_{horizon}_{distance_name}.parquet")
            for split, start, end in (
                ("earlier", original.index.min(), FIRST_TEST),
                ("later", FIRST_TEST, END),
            ):
                scope = period_scope(original, start, end).copy().join(paths)
                scope["scorable"] = scope.planned_exit.lt(end)
                days = pd.DatetimeIndex(scope.anchor.unique()).sort_values()
                for label in ("up_first", "reached_up"):
                    scope["label_known"] = (
                        scope.path_scorable if label == "up_first" else scope.reach_scorable
                    )
                    scope["diagnostic_label"] = scope[label].fillna(0.0)
                    comparison = scope.copy()
                    comparison["gross_dollars"] = scope.diagnostic_label + 25
                    scope["control"] = matched_control(comparison)
                    for name in signals:
                        event = scope.loc[signals[name].reindex(scope.index)]
                        valid = event.loc[event.scorable].copy()
                        probe = valid.copy()
                        probe["net_dollars"] = probe.diagnostic_label
                        stats = incremental_test(probe, days, SEED)
                        rate = float(valid.diagnostic_label.mean()) if len(valid) else np.nan
                        quarters = valid.groupby("quarter").diagnostic_label.mean()
                        rows.append(
                            {
                                "signal": name,
                                "horizon": horizon,
                                "distance": distance_name,
                                "label": label,
                                "split": split,
                                "events": len(event),
                                "scored_events": len(valid),
                                "sessions_with_signal": event.anchor.nunique(),
                                "candidate_sessions": len(days),
                                "session_coverage": event.anchor.nunique() / len(days),
                                "success_rate": rate,
                                "known_label_events": int(valid.label_known.sum()),
                                "unknown_label_rate": float((~valid.label_known).mean()),
                                "known_label_success_rate": float(
                                    valid.loc[valid.label_known, label].mean()
                                ),
                                "success_upper_bound": float(
                                    valid.diagnostic_label.mean() + (~valid.label_known).mean()
                                ),
                                "matched_rate": float(valid.control.mean()),
                                "minimum_quarter_rate": float(quarters.min()),
                                "median_favorable_points": float(valid.mfe_points.median()),
                                "median_adverse_points": float(valid.mae_points.median()),
                                "median_adverse_before_up": float(
                                    valid.adverse_before_up_points.median()
                                ),
                                "ambiguous_rate": float(valid.ambiguous.mean()),
                                "neither_rate": float(valid.neither.mean()),
                                **stats,
                            }
                        )
                print(horizon, distance_name, split, "done", flush=True)
                pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    result = pd.DataFrame(rows)
    later = result.split.eq("later") & result.label.eq("up_first")
    result.loc[later, "fdr_q_first_touch"] = fdr(result.loc[later, "p_hac"].fillna(1).to_numpy())
    result.to_csv(out / "results.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "closed_rth_dates": sorted(CLOSED_RTH_DATES),
                "scope": (
                    "RTH 09:30-16:00 NY only; entry-only; no models, order "
                    "placement, targets or stops"
                ),
                "windows": [5, 15, 60, 240],
                "profile": (
                    "Minute OHLCV volume approximation, not actual traded volume "
                    "at price. Uniform distribution across price bins vs "
                    "close-only allocation; 1/2/4 point bins; previous completed "
                    "RTH value and causally developing 2-point profile. No "
                    "final-session profile available intraday."
                ),
                "entry": (
                    "Completed five-minute setup, next boundary minute open. "
                    "Fifteen-minute Grail setup uses only its completed bars. "
                    "Discretionary methods are mechanical adaptations, not "
                    "claimed exact reproductions."
                ),
                "labels": (
                    "Reach +10/+20 points or one prior ATR; separately favorable "
                    "before equal adverse displacement. Same-minute dual touches "
                    "marked ambiguous and not counted as favorable-first. These "
                    "are path diagnostic barriers, not a target/stop strategy."
                ),
                "frequency": (
                    "Report every-session signal coverage for each individual "
                    "rule and predefined family unions; no force entry"
                ),
                "data": (
                    "Reused development only. Legacy NQ export volume roll "
                    "unverified; source has no trade-price/bid-ask records. "
                    "First day without prior RTH context excluded as declared "
                    "warmup."
                ),
                "missing_data": (
                    "Retain paths resolved before a later gap. Primary success "
                    "rate is a conservative lower bound across all entry events, "
                    "with unknown ordering/hits counted as failures; report "
                    "unknown fraction and upper bound separately."
                ),
                "statistics": (
                    "10-session block uncertainty; same-clock quarter/volatility "
                    "control, excluding own day; FDR across later-period "
                    "favorable-first tests only; exploratory reused history"
                ),
                "Frey": (
                    "Public Stony Brook risk/drawdown material informs "
                    "robustness and adverse-tail analysis, not an invented "
                    "proprietary NQ entry"
                ),
                "sources": [
                    "https://futurestrader71.wordpress.com/2009/09/29/confused-about-the-lingo/",
                    "https://lindaraschke.net/wp-content/uploads/2026/03/raschke_pt2_0304.pdf",
                    (
                        "https://www.brookstradingcourse.com/price-action/10-best-price-action-trading-patterns/"
                    ),
                    "https://www.stonybrook.edu/ams/academics/graduate/quantitative-finance.html",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    run()
