"""100 frozen long rules, four fixed holds, RTH/Globex, no learned high/exit."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from experiment import END, SEED, aggregate, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST
from scipy.stats import t as student_t

from systematic_research.fixed_hold import fixed_hold_outcomes, select_nonoverlapping

HOLDS = (5, 15, 60, 240)


def session_metadata(clock):
    local = clock.tz_convert("America/New_York")
    civil = local.tz_localize(None)
    slot = civil.hour * 60 + civil.minute
    day = civil.normalize()
    rth = (slot >= 570) & (slot < 960)
    globex = (slot >= 1080) | (slot < 570)
    overnight_day = day - pd.to_timedelta((slot < 570).astype(int), unit="D")
    anchor = day + pd.Timedelta(minutes=570)
    end = day + pd.Timedelta(hours=16)
    anchor = anchor.where(rth, overnight_day + pd.Timedelta(hours=18))
    end = end.where(rth, overnight_day + pd.Timedelta(days=1, minutes=570))
    anchor = anchor.where(rth | globex, day + pd.Timedelta(hours=16))
    end = end.where(rth | globex, day + pd.Timedelta(hours=18))
    return pd.DataFrame(
        {
            "kind": np.where(rth, "RTH", np.where(globex, "Globex", "break")),
            "anchor": anchor.tz_localize("America/New_York").tz_convert("UTC"),
            "session_end": end.tz_localize("America/New_York").tz_convert("UTC"),
            "slot": slot,
        },
        index=clock,
    )


def cross_up(a, b):
    return a.gt(b) & a.shift(1).le(b.shift(1) if isinstance(b, pd.Series) else b)


def build_signals(bars, es_bars):
    """Every rule sees only OHLCV of a completed five-minute bar or earlier history."""
    c, o, h, low, v = [
        bars[col].astype(float) for col in ("close", "open", "high", "low", "volume")
    ]
    span = (h - low).replace(0, np.nan)
    tr = pd.concat([span, (h - c.shift()).abs(), (low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean().replace(0, np.nan)
    ret = np.log(c / o)
    body = (c - o) / atr
    location = (c - low) / span
    lower_wick = (pd.concat([c, o], axis=1).min(axis=1) - low) / span
    volume_ratio = v / v.shift(1).rolling(20).mean().replace(0, np.nan)
    ema = {w: c.ewm(span=w, adjust=False, min_periods=w).mean() for w in (3, 5, 8, 13, 21, 34, 55)}
    delta = c.diff()
    rsi = (
        100
        * delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        / delta.abs().ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().replace(0, np.nan)
    )
    k = (
        100
        * (c - low.rolling(14).min())
        / (h.rolling(14).max() - low.rolling(14).min()).replace(0, np.nan)
    )
    z = (c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)
    clock = pd.DatetimeIndex(bars.available_at)
    start_meta = session_metadata(bars.index)
    end_meta = session_metadata(clock)
    anchors = start_meta.anchor
    same_session = anchors.to_numpy() == end_meta.anchor.to_numpy()
    vwap = (((h + low + c) / 3) * v).groupby(anchors).cumsum() / v.groupby(
        anchors
    ).cumsum().replace(0, np.nan)
    distance = (c - vwap) / atr
    current_slot = bars.index.tz_convert("America/New_York")
    slot = current_slot.hour * 60 + current_slot.minute
    seasonal = ret.groupby(slot).transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    sigma = ret.rolling(48).std().replace(0, np.nan)
    es_ret = pd.Series(
        np.log(es_bars.close / es_bars.open).to_numpy(),
        index=pd.DatetimeIndex(es_bars.available_at),
    ).reindex(clock)
    es_volume = pd.Series(
        (es_bars.volume / es_bars.volume.shift().rolling(20).mean().replace(0, np.nan)).to_numpy(),
        index=pd.DatetimeIndex(es_bars.available_at),
    ).reindex(clock)
    es_ret.index, es_volume.index = bars.index, bars.index
    beta = ret.rolling(128, min_periods=64).cov(es_ret) / es_ret.rolling(
        128, min_periods=64
    ).var().replace(0, np.nan)
    residual = (ret - beta * es_ret) / sigma
    signals, registry = {}, []

    def add(family, variant, rule, description):
        name = f"{len(registry) + 1:03d}_{family}_{variant}"
        signals[name] = rule.fillna(False).astype(bool)
        registry.append(
            {"signal": name, "family": family, "variant": str(variant), "description": description}
        )

    for fast, slow in ((3, 8), (5, 13), (8, 21), (13, 34), (21, 55)):
        add(
            "ema_cross",
            f"{fast}_{slow}",
            cross_up(ema[fast], ema[slow]),
            f"EMA {fast} crosses above EMA {slow}",
        )
    for w in (5, 8, 13, 21, 34):
        add("ema_reclaim", w, cross_up(c, ema[w]), f"Completed close reclaims EMA {w}")
    for w in (3, 5, 8, 13, 21):
        add(
            "trend_pullback",
            w,
            cross_up(c, ema[w]) & c.gt(ema[55]) & ema[21].diff().gt(0),
            f"EMA {w} recovery inside rising EMA21/55 trend",
        )
    for w in (6, 12, 20, 36, 60):
        boundary = h.shift().rolling(w).max()
        add("donchian_breakout", w, cross_up(c, boundary), f"Close breaks previous {w}-bar high")
    for w in (3, 4, 6, 8, 12):
        narrow = span.shift().le(span.shift().rolling(w).min())
        add(
            "narrow_range_break",
            w,
            narrow & c.gt(h.shift()),
            f"Break prior high after narrowest prior range in {w} bars",
        )
    for threshold in (-2, -1.5, -1, -0.75, -0.5):
        add(
            "bollinger_recovery",
            threshold,
            cross_up(z, threshold) & body.gt(0),
            f"20-bar price z-score recovers above {threshold}",
        )
    for threshold in (20, 25, 30, 35, 40):
        add(
            "rsi_recovery", threshold, cross_up(rsi, threshold), f"RSI14 recovers above {threshold}"
        )
    for threshold in (20, 30, 40, 50, 60):
        add(
            "stochastic_recovery",
            threshold,
            cross_up(k, k.rolling(3).mean()) & k.shift().lt(threshold),
            f"Stochastic14 crosses its three-bar mean after below {threshold}",
        )
    for fast, slow, signal in ((3, 8, 3), (5, 13, 4), (8, 21, 5), (12, 26, 9), (13, 34, 9)):
        macd = (
            c.ewm(span=fast, adjust=False, min_periods=fast).mean()
            - c.ewm(span=slow, adjust=False, min_periods=slow).mean()
        )
        add(
            "macd_turn",
            f"{fast}_{slow}_{signal}",
            cross_up(macd, macd.ewm(span=signal, adjust=False, min_periods=signal).mean()),
            f"MACD {fast}/{slow} crosses signal EMA {signal}",
        )
    for threshold in (0.3, 0.5, 0.75, 1, 1.5):
        add(
            "selloff_reversal",
            threshold,
            body.shift().lt(-threshold) & body.gt(0) & location.gt(0.65),
            f"Positive recovery after prior body below -{threshold} ATR",
        )
    engulf = c.gt(o) & c.shift().lt(o.shift()) & o.le(c.shift()) & c.ge(o.shift())
    for threshold in (0, 0.1, 0.2, 0.3, 0.5):
        add(
            "bullish_engulfing",
            threshold,
            engulf & body.shift().lt(-threshold),
            f"Bullish engulfing after prior downside body >{threshold} ATR",
        )
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        add(
            "lower_wick_rejection",
            threshold,
            lower_wick.ge(threshold) & location.gt(0.65) & body.gt(0),
            f"Lower wick >= {threshold} of range with bullish close",
        )
    for threshold in (0, 0.75, 1, 1.25, 1.5):
        add(
            "vwap_reclaim",
            threshold,
            cross_up(c, vwap) & volume_ratio.ge(threshold) & same_session,
            f"Session VWAP reclaim with volume ratio >= {threshold}",
        )
    for threshold in (0.25, 0.5, 1, 1.5, 2):
        add(
            "vwap_dip_recovery",
            threshold,
            distance.shift().lt(-threshold) & body.gt(0) & location.gt(0.7) & same_session,
            f"Bullish recovery after VWAP dip >{threshold} ATR",
        )
    elapsed = (bars.index - pd.DatetimeIndex(anchors)).total_seconds() / 60
    first30 = None
    for duration in (5, 10, 15, 30, 60):
        first = (
            bars.loc[(elapsed >= 0) & (elapsed < duration)]
            .assign(anchor=anchors[(elapsed >= 0) & (elapsed < duration)].to_numpy())
            .groupby("anchor")
            .agg(
                high=("high", "max"),
                close=("close", "last"),
                open=("open", "first"),
                count=("close", "size"),
            )
        )
        first.index = pd.DatetimeIndex(first.index)
        boundary = pd.Series(pd.DatetimeIndex(anchors).map(first.high), index=bars.index)
        count = pd.Series(pd.DatetimeIndex(anchors).map(first["count"]), index=bars.index)
        ready = (
            ((clock - pd.DatetimeIndex(anchors)).total_seconds() / 60 >= duration)
            & same_session
            & count.eq(duration // 5)
        )
        add(
            "opening_range_break",
            duration,
            cross_up(c, boundary) & ready,
            f"Completed close breaks fixed first-{duration}-minute session high",
        )
        if duration == 30:
            first30 = (
                pd.Series(pd.DatetimeIndex(anchors).map(first.close - first.open), index=bars.index)
                / atr
            ).where(count.eq(6))
    for threshold in (1, 1.25, 1.5, 2, 3):
        add(
            "volume_momentum",
            threshold,
            body.gt(0.3) & ret.rolling(3).sum().gt(0) & volume_ratio.ge(threshold) & c.gt(ema[21]),
            f"Positive momentum/body with volume ratio >= {threshold}",
        )
    for threshold in (0, 0.75, 1, 1.25, 1.5):
        add(
            "es_confirmed_trend",
            threshold,
            ret.gt(0) & es_ret.gt(0) & ema[21].gt(ema[55]) & es_volume.ge(threshold),
            f"NQ/ES completed bars positive with ES relative volume >= {threshold}",
        )
    for threshold in (0.5, 0.75, 1, 1.5, 2):
        add(
            "es_residual_recovery",
            threshold,
            residual.lt(-threshold) & es_ret.rolling(3).sum().gt(0) & ret.gt(0) & c.gt(ema[55]),
            f"NQ positive recovery while lagging beta-adjusted ES by >{threshold} sigma",
        )
    for threshold in (0, 0.05, 0.1, 0.2, 0.3):
        add(
            "same_clock_momentum",
            threshold,
            seasonal.gt(threshold * sigma) & ret.gt(0) & volume_ratio.ge(1),
            f"Prior same-clock mean positive by >{threshold} sigma, with momentum/volume",
        )
    remaining = (pd.DatetimeIndex(end_meta.session_end) - clock).total_seconds() / 60
    for threshold in (0, 0.1, 0.25, 0.5, 1):
        add(
            "late_session_momentum",
            threshold,
            first30.gt(threshold) & (remaining <= 60) & distance.gt(0) & same_session,
            f"Last session hour after positive first30 >{threshold} ATR and above VWAP",
        )
    if len(registry) != 100 or len(signals) != 100:
        raise AssertionError("Entry catalog must contain exactly 100 named rules")
    matrix = pd.DataFrame(signals, index=bars.index)
    matrix.index = clock
    warm = atr.notna() & ema[55].notna()
    matrix.loc[~warm.to_numpy()] = False
    metadata = end_meta.copy()
    metadata["atr"] = atr.to_numpy()
    return matrix, metadata, pd.DataFrame(registry)


def trade_metrics(frame):
    valid = frame.loc[frame.scorable].copy()
    pnl = valid.net_dollars.to_numpy()
    if not len(pnl):
        return {
            "trades": 0,
            "unscored_trades": len(frame),
            "net_dollars": 0.0,
            "expectancy": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "trade_days": 0,
            "closed_trade_drawdown": np.nan,
        }
    equity = np.r_[0, np.cumsum(pnl)]
    return {
        "trades": len(pnl),
        "unscored_trades": len(frame) - len(valid),
        "trade_days": valid.anchor.nunique(),
        "net_dollars": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum())
        if (pnl < 0).any()
        else np.nan,
        "win_rate": float((pnl > 0).mean()),
        "stress_dollars": float(valid.stress_dollars.sum()),
        "delayed_net_dollars": float(valid.delayed_net_dollars.sum()),
        "closed_trade_drawdown": float((np.maximum.accumulate(equity) - equity).max()),
        "worst_trade": float(pnl.min()),
        "worst_adverse_points": float(valid.mae_points.max()),
        "net_without_best5": float(pnl.sum() - np.sort(pnl)[-5:].sum()),
        "mean_effective_minutes": float(valid.effective_minutes.mean()),
        "capped_trades": int(valid.capped.sum()),
    }


def matched_control(outcomes):
    valid = outcomes.scorable
    frame = outcomes.loc[valid]
    keys = ["split", "quarter", "slot", "vol_bucket"]
    sums = frame.groupby(keys).gross_dollars.transform("sum")
    counts = frame.groupby(keys).gross_dollars.transform("count")
    daily_sum = frame.groupby(keys + ["anchor"]).gross_dollars.transform("sum")
    daily_count = frame.groupby(keys + ["anchor"]).gross_dollars.transform("count")
    control = (sums - daily_sum) / (counts - daily_count).replace(0, np.nan) - 25
    return control.reindex(outcomes.index)


def incremental_test(trades, all_days, seed):
    valid = trades.loc[trades.scorable & trades.control.notna()]
    if len(valid) < 20:
        return {
            "matched_trades": len(valid),
            "incremental_expectancy": np.nan,
            "incremental_ci_low": np.nan,
            "incremental_ci_high": np.nan,
            "p_hac": 1.0,
        }
    delta = valid.net_dollars - valid.control
    daily = (
        pd.DataFrame({"difference": delta, "count": 1, "day": valid.anchor})
        .groupby("day")[["difference", "count"]]
        .sum()
        .reindex(all_days, fill_value=0)
    )
    total = daily["count"].sum()
    mean = daily.difference.sum() / total
    residual = daily.difference.to_numpy() - mean * daily["count"].to_numpy()
    variance = float(residual @ residual)
    for lag in range(1, min(10, len(daily) - 1) + 1):
        variance += 2 * (1 - lag / 11) * float(residual[lag:] @ residual[:-lag])
    se = np.sqrt(max(variance, 1e-9)) / total
    pvalue = float(student_t.sf(mean / se, df=max(1, len(daily) - 1)))
    rng = np.random.default_rng(seed)
    n = len(daily)
    starts = rng.integers(0, n, (2000, int(np.ceil(n / 10))))
    indices = ((starts[:, :, None] + np.arange(10)) % n).reshape(2000, -1)[:, :n]
    sampled = daily.to_numpy()[indices].sum(axis=1)
    estimates = sampled[:, 0] / np.where(sampled[:, 1] > 0, sampled[:, 1], np.nan)
    return {
        "matched_trades": len(valid),
        "incremental_expectancy": float(mean),
        "incremental_ci_low": float(np.nanquantile(estimates, 0.025)),
        "incremental_ci_high": float(np.nanquantile(estimates, 0.975)),
        "p_hac": pvalue,
    }


def fdr(pvalues):
    p = np.asarray(pvalues)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1)
    return result


def run():
    out = BASE / "entry_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    minute = load_minutes("NQ")
    bars = aggregate(minute, 5)
    signals, meta, registry = build_signals(bars, aggregate(load_minutes("ES"), 5))
    opened = minute.open.reindex(signals.index)
    keep = opened.notna() & meta.kind.ne("break") & meta.atr.notna()
    signals, meta = signals.loc[keep], meta.loc[keep]
    registry.to_csv(out / "signals.csv", index=False)
    signatures = {}
    for name in signals:
        signature = np.packbits(signals[name].to_numpy()).tobytes()
        signatures.setdefault(signature, []).append(name)
    (out / "duplicate_signal_vectors.json").write_text(
        json.dumps([v for v in signatures.values() if len(v) > 1], indent=2), encoding="utf-8"
    )
    signals.to_parquet(out / "signal_events.parquet")
    meta.to_parquet(out / "metadata.parquet")
    discovery = meta.index < FIRST_TEST
    cuts = np.quantile(meta.loc[discovery, "atr"], [1 / 3, 2 / 3])
    meta["vol_bucket"] = np.searchsorted(cuts, meta.atr)
    meta["split"] = np.where(discovery, "development", "later_test")
    boundaries = list(pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))) + [END]
    test_quarter = (
        np.searchsorted(pd.DatetimeIndex(boundaries).asi8, meta.index.asi8, side="right") - 1
    )
    earlier_quarter = meta.index.year * 4 + (meta.index.month - 1) // 3
    meta["quarter"] = np.where(discovery, earlier_quarter, test_quarter)
    protocol = {
        "rules": 100,
        "frequency_requirement": (
            "At least one signal-triggered entry in every available session, RTH "
            "and Globex separately; no forced fallback entries"
        ),
        "families": 20,
        "sessions": {"RTH": "09:30-16:00 NY", "Globex": "18:00-09:30 NY"},
        "holds": list(HOLDS),
        "entry": (
            "Completed 5m signal, next observed minute open at the boundary; one "
            "long contract, no overlapping positions or stops"
        ),
        "exit": (
            "Fixed 5/15/60/240 elapsed minutes, capped at session close; sell final "
            "held minute close"
        ),
        "cost": (
            "$25/round trip = 2 NQ ticks per side plus $2.50 commission per side; "
            "$50 stress; $20/point"
        ),
        "latency": "One-minute entry delay with same scheduled exit, sensitivity only",
        "split": {"development_before": str(FIRST_TEST), "later_test_end_exclusive": str(END)},
        "matching": (
            "Same session/clock/quarter/development-volatility bucket, leave own "
            "session out; descriptive benchmark, not a tradable portfolio"
        ),
        "statistics": (
            "10-session block bootstrap; approximate 10-lag HAC one-sided p-values; "
            "BH FDR across all 800 tests"
        ),
        "qualification": (
            "100 later trades, 50 trading sessions, positive net and stress/delay "
            "results, PF>=1.1, 3/4 positive quarters and both halves positive, "
            "survives removing best5 and bestquarter, incremental CI>0 and FDR "
            "q<=.05, >=95% closed-data coverage"
        ),
        "data": (
            "Reused development only; NQ legacy continuous export has unverified "
            "volume roll mapping; no sealed data read"
        ),
        "seed": SEED,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    rows, quarter_rows, trades_saved, baselines, selected_rows = [], [], [], [], []
    for kind in ("RTH", "Globex"):
        mask = meta.kind.eq(kind)
        sm, mm = signals.loc[mask], meta.loc[mask]
        for hold in HOLDS:
            planned = pd.DatetimeIndex(
                np.minimum(
                    (sm.index + pd.Timedelta(minutes=hold)).asi8,
                    pd.DatetimeIndex(mm.session_end).asi8,
                ),
                tz="UTC",
            )
            outcome = fixed_hold_outcomes(minute, sm.index, planned).join(mm)
            outcome["capped"] = outcome.effective_minutes.lt(hold)
            # Entries are selected before this coverage flag is inspected.
            outcome.loc[
                (outcome.split.eq("development") & outcome.planned_exit.ge(FIRST_TEST))
                | outcome.planned_exit.ge(END),
                "scorable",
            ] = False
            outcome["control"] = matched_control(outcome)
            outcome.to_parquet(out / f"outcomes_{kind}_{hold}.parquet")
            print(f"Testing {kind} hold {hold}: {len(outcome)} opportunities", flush=True)
            dev_rank = []
            for split in ("development", "later_test"):
                scope = outcome.loc[outcome.split.eq(split)]
                scope_signals = sm.reindex(scope.index)
                all_days = pd.DatetimeIndex(scope.anchor.unique()).sort_values()
                always = select_nonoverlapping(
                    np.ones(len(scope), dtype=bool),
                    scope.index.asi8,
                    pd.DatetimeIndex(scope.planned_exit).asi8,
                )
                baselines.append(
                    {
                        "session": kind,
                        "hold": hold,
                        "split": split,
                        **trade_metrics(scope.iloc[always]),
                    }
                )
                for number, name in enumerate(sm, 1):
                    selected = select_nonoverlapping(
                        scope_signals[name].to_numpy(dtype=bool),
                        scope.index.asi8,
                        pd.DatetimeIndex(scope.planned_exit).asi8,
                    )
                    trades = scope.iloc[selected].copy()
                    metrics = trade_metrics(trades)
                    entry_sessions = trades.anchor.nunique()
                    metrics.update(
                        {
                            "entry_sessions": entry_sessions,
                            "sessions_without_entry": len(all_days) - entry_sessions,
                            "session_entry_coverage": entry_sessions / len(all_days)
                            if len(all_days)
                            else 0.0,
                            "entries_per_session": len(trades) / len(all_days)
                            if len(all_days)
                            else 0.0,
                        }
                    )
                    record = {
                        "session": kind,
                        "hold": hold,
                        "split": split,
                        "signal": name,
                        "candidate_days": len(all_days),
                        **metrics,
                    }
                    if split == "later_test":
                        record.update(incremental_test(trades, all_days, SEED + number))
                        quarters = (
                            trades.loc[trades.scorable]
                            .groupby("quarter")
                            .net_dollars.sum()
                            .reindex(range(4), fill_value=0)
                        )
                        record.update(
                            {
                                "positive_quarters": int(quarters.gt(0).sum()),
                                "first_half_net": float(quarters.iloc[:2].sum()),
                                "second_half_net": float(quarters.iloc[2:].sum()),
                                "net_without_bestquarter": float(quarters.sum() - quarters.max()),
                            }
                        )
                        quarter_rows.extend(
                            {
                                "session": kind,
                                "hold": hold,
                                "signal": name,
                                "quarter": int(q) + 1,
                                "net_dollars": float(v),
                            }
                            for q, v in quarters.items()
                        )
                        trades["signal"], trades["hold"], trades["session"] = name, hold, kind
                        trades_saved.append(trades)
                    elif (
                        metrics["trades"] >= 50
                        and metrics.get("stress_dollars", 0) > 0
                        and metrics["sessions_without_entry"] == 0
                    ):
                        dev_rank.append(record)
                    rows.append(record)
            winner = max(dev_rank, key=lambda r: r["expectancy"]) if dev_rank else None
            if winner:
                selected_rows.append(
                    {
                        "session": kind,
                        "hold": hold,
                        "selected_before_later_test": winner["signal"],
                        "development_expectancy": winner["expectancy"],
                    }
                )
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
            pd.DataFrame(quarter_rows).to_csv(out / "quarters.csv", index=False)
    results = pd.DataFrame(rows)
    later = results.split.eq("later_test")
    results.loc[later, "fdr_q"] = fdr(results.loc[later, "p_hac"].fillna(1).to_numpy())
    results["worth_further_validation"] = False
    z = results.loc[later]
    qualified = (
        z.sessions_without_entry.eq(0)
        & z.trades.ge(100)
        & z.trade_days.ge(50)
        & z.net_dollars.gt(0)
        & z.stress_dollars.gt(0)
        & z.delayed_net_dollars.gt(0)
        & z.profit_factor.ge(1.1)
        & z.positive_quarters.ge(3)
        & z.first_half_net.gt(0)
        & z.second_half_net.gt(0)
        & z.net_without_best5.gt(0)
        & z.net_without_bestquarter.gt(0)
        & z.incremental_ci_low.gt(0)
        & z.fdr_q.le(0.05)
        & (z.trades / (z.trades + z.unscored_trades).replace(0, np.nan)).ge(0.95)
    )
    results.loc[z.index, "worth_further_validation"] = qualified
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(baselines).to_csv(out / "always_long_baselines.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(out / "development_selections.csv", index=False)
    pd.concat(trades_saved).to_parquet(out / "later_trades.parquet")
    print(f"COMPLETE {out}; qualified={int(qualified.sum())}/800", flush=True)


if __name__ == "__main__":
    run()
