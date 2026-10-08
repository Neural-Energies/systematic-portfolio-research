"""Fixed, session-anchored high forecasts; asymmetric one-sided 20-point success."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from experiment import END, SEED, load_minutes
from fifteen_probability import BASE, prepare
from hour_high import feature_table
from medium_frequency import FIRST_TEST
from range_forecast import carry_completed
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HORIZONS = (5, 15, 60, 240, 0)
SESSIONS = ("RTH", "Globex")
CACHE = BASE / "session_high_features_v1.joblib"
TICK = .25


def session_periods(minutes: pd.DataFrame, session: str, horizon: int):
    """Civil-time session anchors, clipped ends, strict complete minute coverage."""
    local = minutes.index.tz_convert("America/New_York")
    civil = local.tz_localize(None)
    slot = civil.hour * 60 + civil.minute
    date = civil.normalize()
    if session == "RTH":
        keep = (slot >= 570) & (slot < 960)
        anchor = date + pd.Timedelta(minutes=570)
        stop = date + pd.Timedelta(minutes=960)
    elif session == "Globex":
        keep = (slot >= 1080) | (slot < 570)
        day = date - pd.to_timedelta((slot < 570).astype(int), unit="D")
        anchor = day + pd.Timedelta(hours=18)
        stop = day + pd.Timedelta(days=1, minutes=570)
    else:
        raise ValueError("Unknown session")
    frame = minutes.loc[keep].copy()
    anchor = anchor[keep].tz_localize("America/New_York").tz_convert("UTC")
    stop = stop[keep].tz_localize("America/New_York").tz_convert("UTC")
    if horizon:
        elapsed = (frame.index - anchor).total_seconds() / 60
        opened = anchor + pd.to_timedelta(np.floor(elapsed / horizon) * horizon, unit="min")
        ended = pd.DatetimeIndex(np.minimum((opened + pd.Timedelta(minutes=horizon)).asi8, stop.asi8), tz="UTC")
    else:
        opened, ended = anchor, stop
    frame = frame.assign(period_open=opened, label_end=ended, session_open=anchor, quoted_at=frame.index)
    periods = frame.groupby("period_open").agg(
        open=("open", "first"), actual_high=("high", "max"),
        label_end=("label_end", "first"), session_open=("session_open", "first"),
        count=("close", "size"), first_quote=("quoted_at", "min"), last_quote=("quoted_at", "max"),
    )
    periods["effective_minutes"] = (periods.label_end - periods.index).dt.total_seconds() / 60
    valid = periods["count"].eq(periods.effective_minutes)
    valid &= periods.first_quote.eq(periods.index)
    valid &= periods.last_quote.eq(periods.label_end - pd.Timedelta(minutes=1))
    audit = {"session": session, "horizon": horizon or "session", "observed_periods": len(periods), "complete_periods": int(valid.sum()), "excluded_incomplete": int((~valid).sum())}
    periods = periods.loc[valid].copy()
    if periods.actual_high.lt(periods.open).any():
        raise ValueError("Observed high is below its opening price")
    periods["session"] = session
    periods["horizon"] = horizon
    periods["capped"] = periods.effective_minutes.lt(horizon) if horizon else False
    return periods, audit


def publish(raw: np.ndarray, opened: np.ndarray) -> np.ndarray:
    """Prices are floored at the observed opening trade and rounded down to an NQ tick."""
    return np.maximum(opened, np.floor(np.maximum(raw, opened) / TICK + 1e-9) * TICK)


def measures(frame: pd.DataFrame, raw: np.ndarray):
    opened = frame.open.to_numpy()
    predicted = publish(np.asarray(raw), opened)
    error = predicted - frame.actual_high.to_numpy()
    win = (error <= 1e-9) & (error >= -20 - 1e-9)
    elevated = predicted > opened + 1e-9
    return {
        "forecasts": len(frame), "usable_accuracy": float(win.mean()),
        "overprediction_rate": float((error > 1e-9).mean()),
        "under_by_more_than20_rate": float((error < -20 - 1e-9).mean()),
        "reached_rate": float((error <= 1e-9).mean()),
        "mae_points": float(np.abs(error).mean()), "mse_points_squared": float(np.square(error).mean()),
        "rmse_points": float(np.sqrt(np.square(error).mean())),
        "asymmetric_loss": float((3 * np.maximum(error, 0) + np.maximum(-error, 0)).mean()),
        "floor_correction_rate": float((raw < opened).mean()),
        "at_open_rate": float((~elevated).mean()), "above_open_forecasts": int(elevated.sum()),
        "above_open_usable_accuracy": float(win[elevated].mean()) if elevated.any() else np.nan,
        "mean_forecast_above_open": float((predicted - opened).mean()),
    }


def prepare_session_data():
    if CACHE.exists():
        return joblib.load(CACHE)
    minute = load_minutes("NQ")
    _, previous, _, _ = prepare()
    datasets, audits = {}, []
    for session in SESSIONS:
        for horizon in HORIZONS:
            print(f"Preparing {session} {horizon or 'session'}", flush=True)
            data, audit = session_periods(minute, session, horizon)
            datasets[f"{session}_{horizon}"] = data
            audits.append(audit)
    clock = pd.DatetimeIndex(sorted(set().union(*(set(d.index) for d in datasets.values()))))
    core = ["scale", "return", "body_scale", "close_location", "range_scale", "same_clock_return20", "same_clock_up20", "same_clock_volume_ratio", "vwap_distance", "rth_move", "noise_ratio", "overnight_gap", "opening_return", "opening_volume_relative", "minute_return5", "minute_signed_volume15", "minute_realized_vol15", "ar1_forecast"]
    core += [f"{prefix}{window}" for prefix in ("momentum", "volatility", "ema_distance", "volume_z") for window in (4, 16)]
    es = [c for c in previous if c.startswith("market_ES_") and not any(str(n) in c for n in (60, 240))]
    market_returns = [c for c in previous if c.startswith("market_") and c.endswith("_return") and c.count("_") == 2]
    cols = list(dict.fromkeys([c for c in core if c in previous] + es + market_returns))
    x = feature_table(minute, previous[cols], clock)
    # Prior completed snapshots remain available between 15-minute updates.
    x[market_returns] = carry_completed(previous[market_returns], clock, .5)
    ret = np.log(minute.close / minute.open)
    extra = pd.DataFrame(index=minute.index)
    for duration in (5, 15):
        window = f"{duration}min"
        extra[f"fast_return_{duration}"] = ret.rolling(window, min_periods=2).sum()
        extra[f"fast_sigma_{duration}"] = ret.pow(2).rolling(window, min_periods=2).sum().pow(.5)
        extra[f"fast_volume_{duration}"] = minute.volume.rolling(window, min_periods=2).sum()
    extra["reference"] = minute.close
    extra.index += pd.Timedelta(minutes=1)
    x = x.join(carry_completed(extra, clock, 96))
    for key, frame in datasets.items():
        frame = frame.loc[x.scale.reindex(frame.index).notna()].copy()
        frame["scale"] = x.scale.reindex(frame.index) * np.sqrt(frame.effective_minutes / 15)
        frame["target"] = (frame.actual_high - frame.open) / frame.scale
        datasets[key] = frame
    result = x, datasets, pd.DataFrame(audits)
    joblib.dump(result, CACHE)
    return result


def matrices(x, parts):
    fit = parts[0]
    market = [c for c in x if c.startswith("market_") and c.endswith("_return") and c.count("_") == 2 and x.loc[fit.index, c].notna().sum() >= 50]
    raw_market = [c for c in x if c.startswith("market_") and c.endswith("_return") and c.count("_") == 2]
    usable = [c for c in x if c not in raw_market and c not in ("reference",) and x.loc[fit.index, c].notna().sum() >= 50 and x.loc[fit.index, c].nunique() > 1]
    pca = None
    if market:
        pca = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), PCA(n_components=min(3, len(market)), random_state=SEED))
        pca.fit(x.loc[fit.index, market])
    result = []
    for p in parts:
        m = x.loc[p.index, usable].copy()
        m["opening_gap_scaled"] = (p.open - x.reference.reindex(p.index)) / p.scale
        m["effective_minutes"] = p.effective_minutes
        m["minutes_from_session_open"] = (p.index - pd.DatetimeIndex(p.session_open)).total_seconds() / 60
        if pca is not None:
            pcs = pca.transform(x.loc[p.index, market])
            for i in range(pcs.shape[1]):
                m[f"factor_pca_{i+1}"] = pcs[:, i]
        result.append(m.replace([np.inf, -np.inf], np.nan))
    return result, pca, market, usable


def catalog(nfit, threads):
    leaf = max(5, min(50, nfit // 20))
    models = {
        "ridge": Ridge(alpha=100),
        "random_forest": RandomForestRegressor(n_estimators=60, max_depth=8, min_samples_leaf=leaf, max_features=.7, n_jobs=threads, random_state=SEED),
        "extra_trees": ExtraTreesRegressor(n_estimators=60, max_depth=8, min_samples_leaf=leaf, max_features=.7, n_jobs=threads, random_state=SEED),
        "boost_mean": HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=7, min_samples_leaf=leaf, l2_regularization=20, early_stopping=False, random_state=SEED),
    }
    for q in (.1, .25, .5, .7):
        models[f"boost_q{int(q*100)}"] = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=80, max_leaf_nodes=7, min_samples_leaf=leaf, l2_regularization=20, early_stopping=False, random_state=SEED)
    return models


def temporal_parts(data, start, end):
    tune_start = start - pd.Timedelta(days=60)
    fit_start = start - pd.Timedelta(days=330)
    fit = data.loc[(data.index >= fit_start) & (data.index < tune_start) & data.label_end.lt(tune_start)]
    if len(fit) > 20000:
        fit = fit.iloc[np.linspace(0, len(fit)-1, 20000, dtype=int)]
    tune = data.loc[(data.index >= tune_start) & (data.index < start) & data.label_end.lt(start)]
    test = data.loc[(data.index >= start) & (data.index < end) & data.label_end.lt(end)]
    return [fit.copy(), tune.copy(), test.copy()]


def run(session, horizon, threads):
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = BASE / "session_high_runs" / f"{stamp}_{session}_{horizon}"
    out.mkdir(parents=True)
    x, datasets, coverage = prepare_session_data()
    data = datasets[f"{session}_{horizon}"]
    data.to_parquet(out / "labels.parquet")
    coverage.to_csv(out / "source_coverage.csv", index=False)
    protocol = {
        "session": session, "horizon_minutes": horizon or "entire_session", "timezone": "America/New_York",
        "sessions": {"RTH": "09:30-16:00", "Globex": "18:00-09:30 next day"},
        "rule": "One fixed forecast at each session-anchored period open; clip period at session close",
        "success": "0 <= actual_high - published_prediction <=20; ANY overprediction fails",
        "opening_trade": "Observed opening price is known at forecast time; no current-period high/close/volume is used",
        "publication": "opening-price floor, rounded downward to 0.25 NQ tick; corrections and open-level forecasts reported separately",
        "selection": "Earlier tune maximum one-sided accuracy; overshoot and 3:1 asymmetric loss break ties",
        "history": "Fit preceding 330d through -60d; tune last 60d; four later quarterly tests; purge all labels at boundaries",
        "training_cap": "20000 chronologically spaced fit observations; no test/tune subsampling",
        "data": "Development only; reused 2023-08-22 through 2025-08-15; no fresh holdout",
        "NQ_roll": "Legacy continuous export has no verified dated volume-roll map", "seed": SEED,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    limits = list(pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))) + [END]
    forecasts, selected, tuning, support = [], [], [], []
    for fold, (start, end) in enumerate(zip(limits[:-1], limits[1:], strict=True), 1):
        parts = temporal_parts(data, start, end)
        fit, tune, test = parts
        support.append({"fold": fold, "fit": len(fit), "tune": len(tune), "test": len(test)})
        if not len(test):
            continue
        candidates = {}

        def store(config, raw_tune, raw_test, model=None, fold=fold, fit=fit, tune=tune, test=test, candidates=candidates):
            mt = measures(tune, raw_tune) if len(tune) else {"usable_accuracy": 0, "overprediction_rate": 1, "asymmetric_loss": np.inf}
            tuning.append({"fold": fold, "config": config, **{f"tune_{k}": v for k, v in mt.items()}, "fit_end": str(fit.label_end.max()), "tune_end": str(tune.label_end.max()), "test_start": str(test.index.min())})
            frame = test[["open", "actual_high", "scale", "session_open", "effective_minutes", "capped", "label_end"]].copy()
            frame["raw_prediction"] = raw_test
            frame["prediction"] = publish(raw_test, test.open.to_numpy())
            frame["config"], frame["fold"] = config, fold
            forecasts.append(frame)
            candidates[config] = (mt, frame, model)

        for offset in (0, 5, 10, 20):
            store(f"opening_plus{offset}", tune.open.to_numpy() + offset, test.open.to_numpy() + offset)
        if len(fit):
            slots_fit = fit.index.tz_convert("America/New_York").hour * 60 + fit.index.tz_convert("America/New_York").minute
            slots_tune = tune.index.tz_convert("America/New_York").hour * 60 + tune.index.tz_convert("America/New_York").minute
            slots_test = test.index.tz_convert("America/New_York").hour * 60 + test.index.tz_convert("America/New_York").minute
            for q in (.1, .25, .5):
                global_q = fit.target.quantile(q)
                by_slot = fit.target.groupby(slots_fit).quantile(q)
                counts = fit.target.groupby(slots_fit).count()
                weight = counts / (counts + 20)
                by_slot = by_slot * weight + (1-weight) * global_q
                pu = pd.Series(slots_tune).map(by_slot).fillna(global_q).to_numpy()
                pt = pd.Series(slots_test).map(by_slot).fillna(global_q).to_numpy()
                store(f"clock_q{int(q*100)}", tune.open.to_numpy() + pu*tune.scale.to_numpy(), test.open.to_numpy()+pt*test.scale.to_numpy())
        models, pca, markets, columns = {}, None, [], []
        if len(fit) >= 50 and len(tune) >= 8:
            ms, pca, markets, columns = matrices(x, parts)
            xf, xu, xt = ms
            for name, estimator in catalog(len(fit), threads).items():
                print(f"{session} {horizon or 'session'} fold {fold}: {name} fit={len(fit)} test={len(test)}", flush=True)
                model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), clone(estimator))
                model.fit(xf, fit.target)
                pu = tune.open.to_numpy() + model.predict(xu) * tune.scale.to_numpy()
                pt = test.open.to_numpy() + model.predict(xt) * test.scale.to_numpy()
                store(name, pu, pt, model)
                models[name] = model
        ml = [k for k in candidates if k in models]
        for policy, allowed in (("earlier_selected_ml", ml), ("earlier_selected_with_baselines", list(candidates))):
            if not allowed:
                continue
            winner = max(allowed, key=lambda k: (candidates[k][0]["usable_accuracy"], -candidates[k][0]["overprediction_rate"], -candidates[k][0]["asymmetric_loss"]))
            chosen = candidates[winner][1].copy()
            chosen["policy"] = policy
            selected.append(chosen)
            if policy == "earlier_selected_ml":
                joblib.dump({"model": models[winner], "config": winner, "pca": pca, "market_columns": markets, "base_columns": columns, "feature_columns": list(ms[0]), "session": session, "horizon": horizon, "test_start": str(start)}, out / f"model_fold_{fold}.joblib")
        pd.concat(forecasts).to_parquet(out / "forecasts.parquet")
        if selected:
            pd.concat(selected).to_parquet(out / "selected.parquet")
        pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)
        pd.DataFrame(support).to_csv(out / "support.csv", index=False)
        print(f"{session} {horizon or 'session'} fold {fold} complete", flush=True)
    summary = []
    if forecasts:
        for config, frame in pd.concat(forecasts).groupby("config"):
            summary.append({"config": config, **measures(frame, frame.raw_prediction.to_numpy())})
    if selected:
        for policy, frame in pd.concat(selected).groupby("policy"):
            summary.append({"config": policy, **measures(frame, frame.raw_prediction.to_numpy())})
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
    (out / "complete.json").write_text(json.dumps({"session": session, "horizon": horizon, "folds": len(support), "summaries": len(summary)}), encoding="utf-8")
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--session", choices=SESSIONS)
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        if args.prepare:
            prepare_session_data()
        elif args.session is not None and args.horizon is not None:
            run(args.session, args.horizon, args.threads)
        else:
            parser.error("Specify --prepare or both --session and --horizon")
