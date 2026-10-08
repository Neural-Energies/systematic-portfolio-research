"""Exact NQ next-15-minute open/close probability research; development data only."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from direction_expanded import (
    EXTENDED,
    ORIGINAL,
    ProbabilityCalibration,
    contract_safe_bars,
    feature_matrices,
    load_extended,
    screen_features,
)
from experiment import END, ROOT, SEED, aggregate, load_minutes
from medium_frequency import FIRST_TEST, raw_scores, session_keys
from range_forecast import carry_completed, compact_features
from scipy.special import logit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

BASE = ROOT / "work/nq_long_ml"
RULES = ("noise_breakout", "opening_breakout", "dip_recovery", "fast_momentum", "late_momentum")


def slot_history(value: pd.Series, slots: np.ndarray, window: int) -> pd.Series:
    """Past observations of the same clock slot; current session is excluded."""
    return value.groupby(slots).transform(
        lambda s: s.shift(1).rolling(window, min_periods=5).mean()
    )


def intraday_features(minutes: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """Price/volume, RTH noise area, opening range, same-clock and reversal features."""
    clock = pd.DatetimeIndex(bars.available_at)
    local = bars.index.tz_convert("America/New_York")
    slot = local.hour * 60 + local.minute
    day = local.tz_localize(None).normalize()
    ret = np.log(bars.close / bars.open)
    span = (bars.high - bars.low).replace(0, np.nan)
    scale = span.rolling(16, min_periods=8).mean().clip(lower=0.25)
    values = {
        "return": ret,
        "body_scale": (bars.close - bars.open) / scale,
        "close_location": (bars.close - bars.low) / span,
        "upper_wick": (bars.high - bars[["open", "close"]].max(axis=1)) / span,
        "lower_wick": (bars[["open", "close"]].min(axis=1) - bars.low) / span,
        "range_scale": span / scale,
        "slot_sin": np.sin(2 * np.pi * slot / 1440),
        "slot_cos": np.cos(2 * np.pi * slot / 1440),
        "weekday": local.dayofweek.astype(float),
        "same_clock_return5": slot_history(ret, slot, 5),
        "same_clock_return20": slot_history(ret, slot, 20),
        "same_clock_up20": slot_history(ret.gt(0).astype(float), slot, 20),
        "same_clock_volume_ratio": bars.volume
        / slot_history(bars.volume, slot, 20).replace(0, np.nan),
    }
    delta = bars.close.diff()
    for w in (2, 4, 8, 16, 32, 64):
        mean = bars.close.ewm(span=w, adjust=False).mean()
        values[f"momentum{w}"] = ret.rolling(w).sum()
        values[f"volatility{w}"] = ret.rolling(w).std()
        values[f"ema_distance{w}"] = (bars.close - mean) / scale
        values[f"volume_z{w}"] = (
            bars.volume - bars.volume.rolling(w).mean()
        ) / bars.volume.rolling(w).std().replace(0, np.nan)
        values[f"rsi{w}"] = delta.clip(lower=0).rolling(w).mean() / delta.abs().rolling(
            w
        ).mean().replace(0, np.nan)
    for lag in (1, 2, 3, 4):
        values[f"return_lag{lag}"] = ret.shift(lag)
    # A rolling AR(1) is estimated entirely from completed bars, without full-sample fitting.
    beta = ret.rolling(128, min_periods=64).cov(ret.shift(1)) / ret.shift(1).rolling(
        128, min_periods=64
    ).var().replace(0, np.nan)
    values["ar1_forecast"] = ret.rolling(128).mean() + beta * (
        ret - ret.shift(1).rolling(128).mean()
    )
    values["scale"] = scale
    x = pd.DataFrame(values, index=bars.index)

    mlocal = minutes.index.tz_convert("America/New_York")
    mslot = mlocal.hour * 60 + mlocal.minute
    regular = minutes.loc[(mslot >= 570) & (mslot < 960)].copy()
    rday = regular.index.tz_convert("America/New_York").tz_localize(None).normalize()
    typical = (regular.high + regular.low + regular.close) / 3
    vwap = (typical * regular.volume).groupby(rday).cumsum() / regular.volume.groupby(
        rday
    ).cumsum().replace(0, np.nan)
    # All minute-derived information is timestamped AFTER that minute completes.
    rv = pd.DataFrame({"rth_vwap": vwap}, index=regular.index)
    rv.index = rv.index + pd.Timedelta(minutes=1)
    aligned = carry_completed(rv, clock, 0.25)
    x["vwap_distance"] = (bars.close.to_numpy() - aligned.rth_vwap.to_numpy()) / scale.to_numpy()
    open_at = regular.loc[
        (regular.index.tz_convert("America/New_York").hour == 9)
        & (regular.index.tz_convert("America/New_York").minute == 30),
        "open",
    ]
    open_at.index = open_at.index.tz_convert("America/New_York").tz_localize(None).normalize()
    in_regular = (slot >= 570) & (slot < 960)
    rth_open = pd.Series(day.map(open_at), index=bars.index).where(in_regular)
    move = (bars.close / rth_open - 1).where(in_regular)
    noise = slot_history(move.abs(), slot, 14)
    # Previous 15:59 close only; missing holiday closes remain missing.
    prev_close = regular.loc[
        (regular.index.tz_convert("America/New_York").hour == 15)
        & (regular.index.tz_convert("America/New_York").minute == 59),
        "close",
    ]
    prev_close.index = prev_close.index.tz_convert("America/New_York").tz_localize(None).normalize()
    lookup = (
        prev_close.reindex(prev_close.index.union(pd.DatetimeIndex(day.unique())))
        .sort_index()
        .ffill()
        .shift(1)
    )
    prior = pd.Series(day.map(lookup), index=bars.index)
    upper = pd.concat([rth_open, prior], axis=1).max(axis=1) * (1 + noise)
    x["rth_move"] = move
    x["noise_ratio"] = move / noise.replace(0, np.nan)
    x["noise_upper_distance"] = (bars.close - upper) / scale
    x["overnight_gap"] = rth_open / prior - 1
    opening = regular.loc[
        (
            regular.index.tz_convert("America/New_York").hour * 60
            + regular.index.tz_convert("America/New_York").minute
            < 600
        )
    ].copy()
    opening_day = opening.index.tz_convert("America/New_York").tz_localize(None).normalize()
    first = opening.groupby(opening_day).agg(
        open=("open", "first"),
        high=("high", "max"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "size"),
    )
    first = first.where(first["count"].eq(30))
    # Mapping later-known daily aggregates is safe ONLY behind the 10:00 availability gate.
    ready = (slot >= 585) & (slot < 960)  # bar ending 10:00 is the earliest eligible input
    first_ret = pd.Series(day.map(first.close / first.open - 1), index=bars.index).where(ready)
    x["opening_return"] = first_ret
    x["opening_from_prior_close"] = (
        pd.Series(day.map(first.close), index=bars.index).where(ready) / prior - 1
    )
    x["opening_high_distance"] = (
        bars.close - pd.Series(day.map(first.high), index=bars.index).where(ready)
    ) / scale
    x["opening_volume_relative"] = pd.Series(
        day.map(first.volume / first.volume.shift(1).rolling(20, min_periods=5).mean()),
        index=bars.index,
    ).where(ready)
    x["regular_session"] = in_regular.astype(float)
    x["rule_noise_breakout"] = (
        in_regular & x.noise_upper_distance.gt(0) & x.vwap_distance.gt(0)
    ).astype(float)
    x["rule_opening_breakout"] = (
        in_regular & x.opening_high_distance.gt(0) & x.vwap_distance.gt(0)
    ).astype(float)
    x["rule_dip_recovery"] = (
        in_regular & x.vwap_distance.lt(-0.5) & x["return"].gt(0) & x.close_location.gt(0.65)
    ).astype(float)
    x["rule_fast_momentum"] = (
        x.ema_distance8.gt(0) & x.momentum4.gt(0) & x.same_clock_volume_ratio.gt(1)
    ).astype(float)
    x["rule_late_momentum"] = (
        (slot >= 915) & (slot < 960) & x.opening_from_prior_close.gt(0)
    ).astype(float)
    minute_ret = np.log(minutes.close / minutes.open)
    signed = np.sign(minute_ret) * minutes.volume
    # Assign arrays explicitly: Series alignment to a shifted index would leak the next minute.
    micro = pd.DataFrame(
        {
            "minute_return5": minute_ret.rolling(5).sum().to_numpy(),
            "minute_signed_volume15": (
                signed.rolling(15).sum() / minutes.volume.rolling(15).sum().replace(0, np.nan)
            ).to_numpy(),
            "minute_realized_vol15": minute_ret.pow(2).rolling(15).sum().pow(0.5).to_numpy(),
        },
        index=minutes.index + pd.Timedelta(minutes=1),
    )
    x.index = clock
    x = x.join(carry_completed(micro, clock, 0))
    return x.replace([np.inf, -np.inf], np.nan)


def exact_labels(minutes: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Forecast time IS next open; target is strictly close > open, independent of costs."""
    bars = aggregate(minutes, 15)
    out = x.join(bars[["open", "close", "available_at"]], how="inner")
    out["target"] = out.close.gt(out.open).astype(int)
    out["flat"] = out.close.eq(out.open)
    out["label_end"] = out.pop("available_at")
    out["gross_dollars"] = (out.close - out.open) * 20
    out["net_dollars"] = out.gross_dollars - 25
    out["stress_dollars"] = out.gross_dollars - 50
    delayed = minutes.open.reindex(out.index + pd.Timedelta(minutes=1)).to_numpy()
    out["delayed_net_dollars"] = (out.close - delayed) * 20 - 25
    out["normalized_return"] = (out.close - out.open) / out.scale
    out["session"] = session_keys(out.index)
    return out.loc[out.scale.notna() & out.momentum64.notna()]


def prepare():
    cache = BASE / "fifteen_features_v1.joblib"
    if cache.exists():
        return joblib.load(cache)
    minute = load_minutes("NQ")
    bars = aggregate(minute, 15)
    x = intraday_features(minute, bars)
    clock = x.index
    for duration, age in ((60, 2), (240, 8), (0, 96)):
        other = contract_safe_bars(minute, duration)
        x = x.join(carry_completed(compact_features(other, f"nq_{duration}"), clock, age))
    markets, coverage = {}, []
    for symbol in ORIGINAL + EXTENDED:
        print(f"15m predictors: {symbol}", flush=True)
        market = load_extended(symbol) if symbol in EXTENDED else load_minutes(symbol)
        b = contract_safe_bars(market, 15, sparse_predictor=True)
        compact = compact_features(b, f"market_{symbol}")
        compact[f"market_{symbol}_lag1"] = compact[f"market_{symbol}_return"].shift(1)
        compact[f"market_{symbol}_coverage"] = b.coverage_fraction.to_numpy()
        compact[f"market_{symbol}_quote_age"] = b.quote_age_minutes.to_numpy()
        matched = carry_completed(compact, clock, 0)
        x = x.join(matched)
        mr = matched[f"market_{symbol}_return"]
        markets[symbol] = mr
        nr = x["return"].where(mr.notna())
        beta = nr.rolling(128, min_periods=32).cov(mr) / mr.rolling(
            128, min_periods=32
        ).var().replace(0, np.nan)
        x[f"market_{symbol}_beta"] = beta
        x[f"market_{symbol}_residual"] = x["return"] - beta * mr
        x[f"market_{symbol}_correlation"] = nr.rolling(128, min_periods=32).corr(mr)
        coverage.append(
            {
                "symbol": symbol,
                "matched_15m": float(mr.notna().mean()),
                "start": str(market.index.min()),
            }
        )
        if symbol in ("ES", "RTY", "ZN"):
            for duration, age in ((60, 2), (240, 8), (0, 96)):
                longer = contract_safe_bars(market, duration, sparse_predictor=True)
                x = x.join(
                    carry_completed(
                        compact_features(longer, f"market_{symbol}_{duration}"), clock, age
                    )
                )
    x = x.replace([np.inf, -np.inf], np.nan)
    result = exact_labels(minute, x), x, pd.DataFrame(markets), coverage
    joblib.dump(result, cache)
    return result


def catalog():
    return {
        "logistic": LogisticRegression(C=0.05, max_iter=1000, random_state=SEED),
        "shrinkage_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "gaussian_nb": GaussianNB(var_smoothing=0.01),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            max_features=0.7,
            n_jobs=4,
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=40, n_jobs=4, random_state=SEED
        ),
        "hist_boost": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            l2_regularization=20,
            early_stopping=False,
            random_state=SEED,
        ),
    }


def quantile_probability(predicted: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Interpolate P(return<=0) across monotone quantiles; conservative tail bounds."""
    ordered = np.sort(predicted, axis=1)
    return np.array(
        [1 - np.interp(0, row, levels, left=levels[0], right=levels[-1]) for row in ordered]
    )


def measure(part: pd.DataFrame, p: np.ndarray, gate: np.ndarray | None = None) -> dict:
    if gate is None:
        gate = np.ones(len(part), dtype=bool)
    trade = (p >= 0.66) & gate
    pnl = part.net_dollars.to_numpy()[trade]
    y = part.target.to_numpy()
    return {
        "observations": len(part),
        "accuracy": float((y == (p >= 0.5)).mean()),
        "brier": float(brier_score_loss(y, p)),
        "up_baseline": float(y.mean()),
        "trades": int(trade.sum()),
        "coverage": float(trade.mean()),
        "direction_win": float(y[trade].mean()) if trade.any() else np.nan,
        "forecast_probability": float(p[trade].mean()) if trade.any() else np.nan,
        "net_dollars": float(pnl.sum()),
        "net_win": float((pnl > 0).mean()) if trade.any() else np.nan,
        "stress_dollars": float(part.stress_dollars.to_numpy()[trade].sum()),
        "delayed_net_dollars": float(part.delayed_net_dollars.to_numpy()[trade].sum()),
        "profit_factor": float(pnl[pnl > 0].sum() / -pnl[pnl < 0].sum())
        if (pnl < 0).any()
        else np.nan,
    }


def time_parts(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, window: int = 180):
    a, b = start - pd.Timedelta(days=42), start - pd.Timedelta(days=14)
    edges = [(start - pd.Timedelta(days=window), a), (a, b), (b, start), (start, end)]
    return [
        data.loc[(data.index >= lo) & (data.index < hi) & data.label_end.lt(hi)].copy()
        for lo, hi in edges
    ]


def run():
    out = BASE / "fifteen_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    data, x, markets, coverage = prepare()
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    protocol = {
        "target": "next 15-minute close strictly above open; flat is not up",
        "policy": "completed-bar information at open; long if calibrated P(up)>=0.66; exit close",
        "execution": (
            "literal bar open/close; ideal instantaneous decision at boundary; "
            "1-minute delay sensitivity separately"
        ),
        "cost": "$25 one-contract round-trip; $50 stress; NQ $20/point",
        "data": (
            "development only 2023-08-22 to 2025-08-16 exclusive; "
            "reused exploratory period, not a fresh holdout"
        ),
        "roll": (
            "extension uses previous UTC-day volume leader; "
            "NQ legacy continuous data lacks verified volume-roll map"
        ),
        "retraining": (
            "monthly: fit past 180 days until -42d; calibrate -42d/-14d; "
            "select -14d/start; purge labels ending at boundary"
        ),
        "selection": (
            "probability champion by prior tune Brier; selective champion requires "
            ">=30 tune trades and Wilson 80% lower bound >=0.55 then maximize lower bound"
        ),
        "families": list(catalog())
        + ["quantile_distribution", "clock_regime_beta", "conditional_hist_boost"],
        "features": len(x.columns),
        "threshold": 0.66,
        "seed": SEED,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"RUN {out}; {len(data)} rows; {len(x.columns)} raw features", flush=True)
    boundaries = list(pd.date_range(FIRST_TEST, periods=12, freq=pd.DateOffset(months=1))) + [END]
    forecasts, records, selected, rankrows, baselines = [], [], [], [], []
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        parts = time_parts(data, start, end)
        if any(len(p) < 100 for p in parts):
            raise ValueError("Insufficient temporal partition support")
        fit, calpart, tune, test = parts
        matrices, pca, observed = feature_matrices(parts, x, markets)
        xf, xc, xu, xt = matrices
        split = int(len(fit) * 0.75)
        # This screening validation is contained in FIT, before all calibration/tuning/test data.
        rank = screen_features(
            xf.iloc[:split], fit.target.iloc[:split], xf.iloc[split:], fit.target.iloc[split:]
        )
        rankrows.extend({"fold": fold, "feature": f, "importance": v} for f, v in rank.items())
        mandatory = [
            c
            for c in xf
            if c.startswith("rule_") or c in ("slot_sin", "slot_cos", "regular_session")
        ]
        columns100 = list(dict.fromkeys(list(rank.head(100).index) + mandatory))
        candidates = {}

        def store_candidate(
            name,
            score_cal,
            score_tune,
            score_test,
            gate_tune=None,
            gate_test=None,
            raw=False,
            calpart=calpart,
            tune=tune,
            fit=fit,
            test=test,
            fold=fold,
            candidates=candidates,
        ):
            methods = (
                ("raw", "sigmoid", "binned_isotonic") if raw else ("sigmoid", "binned_isotonic")
            )
            for method in methods:
                if method == "raw":
                    from scipy.special import expit

                    pu, pt = expit(score_tune), expit(score_test)
                else:
                    calibrator = ProbabilityCalibration(method).fit(score_cal, calpart.target)
                    pu, pt = calibrator.predict(score_tune), calibrator.predict(score_test)
                config = f"{name}_{method}"
                stats = measure(tune, pu, gate_tune)
                record = {
                    "fold": fold,
                    "config": config,
                    "fit_start": str(fit.index.min()),
                    "fit_end": str(fit.label_end.max()),
                    "cal_end": str(calpart.label_end.max()),
                    "tune_end": str(tune.label_end.max()),
                    "test_start": str(test.index.min()),
                    **{f"tune_{k}": v for k, v in stats.items()},
                }
                records.append(record)
                candidates[config] = (pt, gate_test, stats)
                frame = test[
                    [
                        "target",
                        "gross_dollars",
                        "net_dollars",
                        "stress_dollars",
                        "delayed_net_dollars",
                        "session",
                        "label_end",
                    ]
                ].copy()
                frame["probability"] = pt
                frame["gate"] = True if gate_test is None else gate_test
                frame["config"], frame["fold"] = config, fold
                forecasts.append(frame)

        for count in (20, 100):
            cols = list(rank.head(20).index) if count == 20 else columns100
            for name, estimator in catalog().items():
                print(f"month {fold}: top{count} {name}", flush=True)
                model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)
                model.fit(xf[cols], fit.target)
                scores = [raw_scores(model, m[cols]) for m in (xc, xu, xt)]
                store_candidate(f"top{count}_{name}", *scores, raw=True)
        # Predict the return distribution, then infer the probability of being above zero.
        levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        preds = [[], [], []]
        for quantile in levels:
            print(f"month {fold}: return quantile {quantile}", flush=True)
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile,
                    max_iter=80,
                    max_leaf_nodes=7,
                    min_samples_leaf=100,
                    l2_regularization=20,
                    early_stopping=False,
                    random_state=SEED,
                ),
            )
            model.fit(xf[columns100], fit.normalized_return)
            for j, m in enumerate((xc, xu, xt)):
                preds[j].append(model.predict(m[columns100]))
        scores = [
            logit(np.clip(quantile_probability(np.column_stack(arr), levels), 1e-6, 1 - 1e-6))
            for arr in preds
        ]
        store_candidate("quantile_distribution", *scores, raw=True)

        # Shrunk empirical probabilities by time, volume regime and momentum: interpretable control.
        keys = [
            pd.DataFrame(
                {
                    "slot": (
                        p.index.tz_convert("America/New_York").hour * 4
                        + p.index.tz_convert("America/New_York").minute // 15
                    ),
                    "momentum": p.momentum4.gt(0).astype(int).to_numpy(),
                    "volume": p.same_clock_volume_ratio.gt(1).astype(int).to_numpy(),
                },
                index=p.index,
            )
            for p in parts
        ]
        table = (
            keys[0]
            .assign(target=fit.target.to_numpy())
            .groupby(["slot", "momentum", "volume"])
            .target.agg(["sum", "count"])
        )
        prior = float(fit.target.mean())
        table["p"] = (table["sum"] + 100 * prior) / (table["count"] + 100)
        scores = [
            logit(k.join(table.p, on=["slot", "momentum", "volume"]).p.fillna(prior).to_numpy())
            for k in keys[1:]
        ]
        store_candidate("clock_regime_beta", *scores, raw=True)

        for rule in RULES:
            masks = [p[f"rule_{rule}"].eq(1).to_numpy() for p in parts]
            # Sparse strategies are still reported as fixed-rule baselines, never quietly dropped.
            if masks[0].sum() >= 300 and masks[1].sum() >= 80:
                print(f"month {fold}: conditional {rule}", flush=True)
                model = make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingClassifier(
                        max_iter=80,
                        max_leaf_nodes=7,
                        min_samples_leaf=40,
                        l2_regularization=20,
                        early_stopping=False,
                        random_state=SEED,
                    ),
                )
                model.fit(xf.loc[masks[0], columns100], fit.target.loc[masks[0]])
                score_cal = raw_scores(model, xc[columns100])
                # Fit calibration on the same strategy's candidates, rather than unrelated states.
                for method in ("sigmoid", "binned_isotonic"):
                    calibration = ProbabilityCalibration(method).fit(
                        score_cal[masks[1]], calpart.target.loc[masks[1]]
                    )
                    pu, pt = [
                        calibration.predict(raw_scores(model, m[columns100])) for m in (xu, xt)
                    ]
                    config = f"conditional_{rule}_{method}"
                    stats = measure(tune, pu, masks[2])
                    records.append(
                        {
                            "fold": fold,
                            "config": config,
                            **{f"tune_{k}": v for k, v in stats.items()},
                        }
                    )
                    candidates[config] = (pt, masks[3], stats)
                    frame = test[
                        [
                            "target",
                            "gross_dollars",
                            "net_dollars",
                            "stress_dollars",
                            "delayed_net_dollars",
                            "session",
                            "label_end",
                        ]
                    ].copy()
                    frame["probability"], frame["gate"], frame["config"], frame["fold"] = (
                        pt,
                        masks[3],
                        config,
                        fold,
                    )
                    forecasts.append(frame)
            for p, mask, period in ((tune, masks[2], "tune"), (test, masks[3], "test")):
                base = measure(p, np.ones(len(p)), mask)
                baselines.append({"fold": fold, "rule": rule, "period": period, **base})
        base = measure(test, np.ones(len(test)))
        baselines.append({"fold": fold, "rule": "always_long", "period": "test", **base})
        best_probability = min(candidates, key=lambda k: candidates[k][2]["brier"])
        qualifying = {
            k: v
            for k, v in candidates.items()
            if v[2]["trades"] >= 30 and wilson_lower(v[2]["direction_win"], v[2]["trades"]) >= 0.55
        }
        best_selective = (
            max(
                qualifying,
                key=lambda k: wilson_lower(
                    qualifying[k][2]["direction_win"], qualifying[k][2]["trades"]
                ),
            )
            if qualifying
            else None
        )
        for policy, chosen in (
            ("probability_champion", best_probability),
            ("qualified_selective", best_selective),
        ):
            if chosen is None:
                ptest, gate = np.full(len(test), prior), np.zeros(len(test), dtype=bool)
            else:
                ptest, gate, _ = candidates[chosen]
            frame = test[
                [
                    "target",
                    "gross_dollars",
                    "net_dollars",
                    "stress_dollars",
                    "delayed_net_dollars",
                    "session",
                    "label_end",
                ]
            ].copy()
            frame["probability"], frame["gate"], frame["config"], frame["fold"] = (
                ptest,
                True if gate is None else gate,
                chosen or "abstain",
                fold,
            )
            frame["policy"] = policy
            selected.append(frame)
        pd.DataFrame(records).to_csv(out / "monthly_tuning.csv", index=False)
        pd.DataFrame(rankrows).to_csv(out / "feature_ranking.csv", index=False)
        pd.concat(selected).to_parquet(out / "selected.parquet")
        print(
            f"month {fold} completed; probability={best_probability}; qualified={best_selective}",
            flush=True,
        )
    predicted = pd.concat(forecasts)
    predicted.to_parquet(out / "forecasts.parquet")
    pd.DataFrame(baselines).to_csv(out / "rule_baselines.csv", index=False)
    summaries = [
        {"config": name, **measure(group, group.probability.to_numpy(), group.gate.to_numpy())}
        for name, group in predicted.groupby("config")
    ]
    pd.DataFrame(summaries).to_csv(out / "model_summary.csv", index=False)
    pd.DataFrame(
        [
            {"policy": name, **measure(group, group.probability.to_numpy(), group.gate.to_numpy())}
            for name, group in pd.concat(selected).groupby("policy")
        ]
    ).to_csv(out / "policy_summary.csv", index=False)
    print(f"COMPLETE {out}", flush=True)


def wilson_lower(rate: float, n: int, z: float = 1.281551565545):
    if not n or not np.isfinite(rate):
        return 0.0
    return (rate + z * z / (2 * n) - z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))) / (
        1 + z * z / n
    )


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        run()
