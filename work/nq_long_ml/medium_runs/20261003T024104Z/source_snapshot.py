"""Local frequency-constrained research; all multi-day P&L remains provisional."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from experiment import (
    BENCHMARKS,
    END,
    SEED,
    aggregate,
    block_importance,
    classifiers,
    features,
    load_minutes,
)
from scipy.special import logit
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPIRY_MONTHS = (3, 6, 9, 12)
FIRST_TEST = pd.Timestamp("2024-08-22", tz="UTC")
MODELS = ("rf", "rf_flexible", "extra_trees", "hist_boost", "logistic", "lda", "svm")


def session_keys(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    local = index.tz_convert("America/Chicago").tz_localize(None)
    return (local - pd.Timedelta(hours=17)).normalize()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build an hourly decision clock; carry only already-completed bar features."""
    cache = Path(__file__).parent / "medium_feature_cache_v1.joblib"
    if cache.exists():
        return joblib.load(cache)
    minutes = load_minutes("NQ")
    bars = aggregate(minutes, 60)
    factors = pd.DataFrame(
        {s: np.log((b := aggregate(load_minutes(s), 60))["close"] / b["open"]) for s in BENCHMARKS}
    ).reindex(bars.index)
    raw = features(bars, factors)
    r = np.log(bars["close"] / bars["open"])
    raw["slow_momentum_115"] = r.rolling(115).sum()
    raw["slow_momentum_230"] = r.rolling(230).sum()
    raw["slow_volatility_115"] = r.rolling(115).std()
    raw["slow_downside_115"] = r.clip(upper=0).pow(2).rolling(115).mean().pow(0.5)
    v = bars["volume"].astype(float)
    raw["slow_volume_z_115"] = (v - v.rolling(115).mean()) / v.rolling(115).std()
    raw.index = pd.DatetimeIndex(bars["available_at"])
    factors.index = raw.index
    clock = pd.date_range(raw.index.min(), END, freq="h", inclusive="left")
    raw["feature_timestamp"] = raw.index
    grid = raw.reindex(clock).ffill()
    grid["feature_age_hours"] = (
        pd.Series(clock, index=clock) - grid.pop("feature_timestamp")
    ).dt.total_seconds() / 3600
    grid = grid.loc[grid["feature_age_hours"].le(3) & grid["momentum_64"].notna()]
    factors = factors.reindex(clock).reindex(grid.index)
    joblib.dump((minutes, grid, factors), cache)
    return minutes, grid, factors


def build_labels(minutes: pd.DataFrame, grid: pd.DataFrame, days: int) -> pd.DataFrame:
    """Delayed fills; 4 trading-hour or session-based label maturity, no roll claim."""
    times = minutes.index
    minute_sessions = session_keys(times)
    sessions = pd.DatetimeIndex(minute_sessions.unique()).sort_values()
    sid = sessions.get_indexer(minute_sessions)
    frame = grid.copy()
    decision = frame.index
    requested_entry = decision + pd.Timedelta(minutes=1)
    entry = times.searchsorted(requested_entry)
    entry = np.minimum(entry, len(times) - 1)
    entry_time = times[entry]
    entry_ok = (entry_time >= requested_entry) & (
        (entry_time - requested_entry) <= pd.Timedelta(minutes=2)
    )
    target = entry + 240
    allow = np.zeros(len(frame), dtype=bool)
    if days:
        for j, pos in enumerate(entry):
            current = sid[pos]
            future = current + days
            if future >= len(sessions):
                continue  # known dataset-end guard uses four hours
            path = sessions[current : future + 1] + pd.Timedelta(days=1)
            if path.month.isin(EXPIRY_MONTHS).any():
                continue
            current_anchor = (
                (sessions[current] + pd.Timedelta(hours=17))
                .tz_localize("America/Chicago")
                .tz_convert("UTC")
            )
            future_anchor = (
                (sessions[future] + pd.Timedelta(hours=17))
                .tz_localize("America/Chicago")
                .tz_convert("UTC")
            )
            desired = future_anchor + (times[pos] - current_anchor)
            found = times.searchsorted(desired)
            if (
                found < len(times)
                and sid[found] == future
                and times[found] - desired <= pd.Timedelta(minutes=2)
            ):
                target[j], allow[j] = found, True
    target_ok = target < len(times)
    safe_target = np.minimum(target, len(times) - 1)
    # Four-hour fallback cannot cross a session during an expiration month.
    quarter = (minute_sessions[entry] + pd.Timedelta(days=1)).month.isin(EXPIRY_MONTHS)
    target_ok &= ~quarter | (sid[safe_target] == sid[entry]) | allow
    frame["overnight_allowed"] = allow.astype(float)
    frame["entry_position"] = entry
    frame["max_exit_position"] = safe_target
    frame["entry_time"] = entry_time
    frame["label_end"] = times[safe_target]
    frame["session"] = minute_sessions[entry]
    frame["execution_valid"] = entry_ok & target_ok
    frame["effective_days"] = np.where(allow, days, 0)
    price = minutes["open"].to_numpy()
    net = (price[safe_target] - price[entry]) * 20 - 25
    frame["target"] = np.where(frame["execution_valid"], (net > 0).astype(float), np.nan)
    return frame


def model_catalog() -> dict:
    models = classifiers()
    models["rf_flexible"] = RandomForestClassifier(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=15,
        max_features=0.7,
        n_jobs=4,
        random_state=SEED,
    )
    return {name: models[name] for name in MODELS}


def raw_scores(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return logit(np.clip(model.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6))
    return model.decision_function(x)


def hypothetical_exits(frame: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Replay future model signals; only signals observed at exit time can trigger it."""
    ordered = frame.sort_index().copy()
    p = ordered["probability"].to_numpy()
    entry_pos = ordered["entry_position"].to_numpy(dtype=int)
    planned = ordered["max_exit_position"].to_numpy(dtype=int)
    decision_ns = ordered.index.asi8
    times = minutes.index
    prices = minutes["open"].to_numpy()
    exits = planned.copy()
    reasons = np.full(len(ordered), "time_limit", dtype=object)
    for i in range(len(ordered)):
        if not ordered["execution_valid"].iloc[i]:
            continue
        minimum = min(entry_pos[i] + 240, len(times) - 1)
        first = np.searchsorted(entry_pos, minimum)
        last = np.searchsorted(decision_ns, times[planned[i]].value, side="right")
        quote_delays = ordered["entry_time"].iloc[first:last].array.asi8 - decision_ns[first:last]
        adverse = np.flatnonzero(
            (p[first:last] < 0.45) & (quote_delays <= pd.Timedelta(minutes=3).value)
        )
        if len(adverse):
            j = first + adverse[0]
            candidate = entry_pos[j]
            if minimum <= candidate <= planned[i]:
                exits[i], reasons[i] = candidate, "probability_exit"
    ordered["exit_position"] = exits
    ordered["exit_time"] = times[exits]
    ordered["entry_price"] = prices[entry_pos]
    ordered["exit_price"] = prices[exits]
    ordered["exit_reason"] = reasons
    ordered["observed_minutes_held"] = exits - entry_pos
    return ordered


def daily_policy(frame: pd.DataFrame, threshold: float | None, quota: bool) -> pd.DataFrame:
    """Forward-only scheduled entries; never rank a day's future opportunities."""
    opened, used, active = [], set(), []
    for decision, row in frame.sort_index().iterrows():
        local = decision.tz_convert("America/New_York")
        if local.hour not in (8, 10, 12):
            continue
        day = row["session"]
        if day in used:
            continue
        cutoff = float(row["selected_threshold"]) if threshold is None else threshold
        confident = row["probability"] >= cutoff
        deadline = 8 if local.month in EXPIRY_MONTHS else 12
        fallback = quota and local.hour >= deadline
        if not confident and not fallback:
            continue
        if not row["execution_valid"]:
            # An unscorable attempted entry is recorded as a coverage failure.
            used.add(day)
            continue
        active = [t for t in active if t > row["entry_time"]]
        if len(active) >= 5:
            continue
        record = row.to_dict()
        record.update(
            decision_time=decision,
            quota_fallback=not confident,
            concurrent_at_entry=len(active) + 1,
        )
        opened.append(record)
        active.append(row["exit_time"])
        used.add(day)
    return pd.DataFrame(opened)


def trade_summary(trades: pd.DataFrame, sessions: pd.DatetimeIndex, cost: float = 25) -> dict:
    if trades.empty:
        return {
            "trades": 0,
            "trades_per_day": 0.0,
            "daily_coverage": 0.0,
            "missing_days": len(sessions),
            "net_dollars": 0.0,
            "expectancy": np.nan,
            "quota_fallback_fraction": np.nan,
            "max_concurrent": 0,
        }
    pnl = (trades["exit_price"] - trades["entry_price"]) * 20 - cost
    coverage = trades["session"].nunique() / len(sessions)
    losses = -pnl.clip(upper=0).sum()
    return {
        "trades": len(trades),
        "trades_per_day": len(trades) / len(sessions),
        "daily_coverage": coverage,
        "missing_days": len(sessions) - trades["session"].nunique(),
        "net_dollars": float(pnl.sum()),
        "expectancy": float(pnl.mean()),
        "win_rate": float(pnl.gt(0).mean()),
        "profit_factor": float(pnl.clip(lower=0).sum() / losses) if losses else np.nan,
        "quota_fallback_fraction": float(trades["quota_fallback"].mean()),
        "max_concurrent": int(trades["concurrent_at_entry"].max()),
        "mean_trading_hours_held": float(trades["observed_minutes_held"].mean() / 60),
    }


def marked_equity(trades: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Mark open positions each session; do not hide unrealized losses."""
    market = minutes.loc[minutes.index >= FIRST_TEST]
    closes = market.assign(session=session_keys(market.index)).groupby("session").tail(1)
    rows = []
    for t, row in closes.iterrows():
        if trades.empty:
            rows.append(
                {
                    "session": session_keys(pd.DatetimeIndex([t]))[0],
                    "equity": 0.0,
                    "open_contracts": 0,
                }
            )
            continue
        exited = trades["exit_time"].le(t)
        live = trades["entry_time"].le(t) & trades["exit_time"].gt(t)
        realized = (
            (trades.loc[exited, "exit_price"] - trades.loc[exited, "entry_price"]) * 20 - 25
        ).sum()
        floating = ((row["close"] - trades.loc[live, "entry_price"]) * 20 - 12.5).sum()
        rows.append(
            {
                "session": session_keys(pd.DatetimeIndex([t]))[0],
                "equity": float(realized + floating),
                "open_contracts": int(live.sum()),
            }
        )
    result = pd.DataFrame(rows)
    result["drawdown"] = result["equity"].cummax().clip(lower=0) - result["equity"]
    return result


def run_horizon(
    days: int, minutes: pd.DataFrame, grid: pd.DataFrame, factors: pd.DataFrame, out: Path
) -> tuple[list[dict], list[dict]]:
    name = "4h" if days == 0 else f"{days}d"
    print(f"{name}: preparing targets", flush=True)
    data = build_labels(minutes, grid, days)
    cols = list(grid) + ["overnight_allowed"]
    forecasts, importance_rows, tuning_rows = [], [], []
    boundaries = pd.date_range("2024-08-22", periods=5, freq=pd.DateOffset(months=3), tz="UTC")
    boundaries = list(boundaries[:-1]) + [END]
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        training = data.loc[
            (data.index < start) & data["label_end"].lt(start) & data["target"].notna()
        ]
        a, b, c = [int(len(training) * fraction) for fraction in (0.55, 0.70, 0.85)]
        fit, select, cal, tune = [
            training.iloc[lo:hi].copy() for lo, hi in ((0, a), (a, b), (b, c), (c, len(training)))
        ]
        fit = fit.loc[fit["label_end"].lt(select.index.min())]
        select = select.loc[select["label_end"].lt(cal.index.min())]
        cal = cal.loc[cal["label_end"].lt(tune.index.min())]
        test = data.loc[(data.index >= start) & (data.index < end)].copy()
        if any(len(part) < 80 or part["target"].nunique() < 2 for part in (fit, select, cal, tune)):
            raise ValueError(f"{name} fold {fold}: inadequate training segment")
        pc = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=3, random_state=SEED),
        )
        pc.fit(factors.reindex(fit.index))
        matrices = []
        for part in (fit, select, cal, tune, test):
            matrix = part[cols].copy()
            scores = pc.transform(factors.reindex(part.index))
            for component in range(3):
                matrix[f"pca_factor_{component + 1}"] = scores[:, component]
            matrices.append(matrix)
        xf, xs, xc, xu, xt = matrices
        assert xf.shape[1] == 100
        if xf.isna().all().any():
            raise ValueError("Unobserved training feature")
        screen = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), model_catalog()["rf"]
        )
        screen.fit(xf, fit["target"])
        imp = block_importance(screen, xs, select["target"])
        selected = list(imp.head(20).index)
        ranking = imp.rename_axis("feature").rename("importance").reset_index()
        ranking["fold"], ranking["selected"] = fold, ranking["feature"].isin(selected)
        importance_rows.append(ranking)
        for model_name, estimator in model_catalog().items():
            print(f"{name} fold {fold}: {model_name}", flush=True)
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), clone(estimator)
            )
            model.fit(pd.concat([xf, xs])[selected], pd.concat([fit, select])["target"])
            calibrator = LogisticRegression(C=1, max_iter=1000)
            calibrator.fit(raw_scores(model, xc[selected]).reshape(-1, 1), cal["target"])
            tune_prob = calibrator.predict_proba(raw_scores(model, xu[selected]).reshape(-1, 1))[
                :, 1
            ]
            tuning = tune.drop(columns=cols, errors="ignore").copy()
            tuning["probability"] = tune_prob
            tuning = hypothetical_exits(tuning, minutes)
            tuning_sessions = pd.DatetimeIndex(tune["session"].unique())
            trial_records = []
            for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.66):
                trial_trades = daily_policy(tuning, threshold, quota=True)
                score = trade_summary(trial_trades, tuning_sessions, cost=50)
                record = {
                    "horizon": name,
                    "fold": fold,
                    "model": model_name,
                    "threshold": threshold,
                    **score,
                }
                tuning_rows.append(record)
                trial_records.append(record)
            best = max(
                trial_records,
                key=lambda rec: (
                    rec["daily_coverage"] >= 1,
                    rec["net_dollars"],
                    -rec["quota_fallback_fraction"],
                    rec["threshold"],
                ),
            )
            prediction = test.drop(columns=cols, errors="ignore").copy()
            prediction["probability"] = calibrator.predict_proba(
                raw_scores(model, xt[selected]).reshape(-1, 1)
            )[:, 1]
            prediction["selected_threshold"] = best["threshold"]
            prediction["model"], prediction["fold"], prediction["horizon"] = model_name, fold, name
            forecasts.append(prediction)
            if fold == 4:
                joblib.dump(
                    {
                        "model": model,
                        "calibrator": calibrator,
                        "pca": pc,
                        "selected_features": selected,
                        "threshold": best["threshold"],
                        "status": "development_only_not_live_ready",
                    },
                    out / f"model_{name}_{model_name}.joblib",
                )
        pd.DataFrame(tuning_rows).to_csv(out / f"tuning_{name}.csv", index=False)
    predictions = pd.concat(forecasts).sort_index()
    predictions.to_parquet(out / f"forecasts_{name}.parquet")
    pd.concat(importance_rows).to_csv(out / f"importance_{name}.csv", index=False)
    session_denominator = pd.DatetimeIndex(
        session_keys(minutes.loc[minutes.index >= FIRST_TEST].index).unique()
    )
    summary, folds = [], []
    for model_name, prediction in predictions.groupby("model"):
        hypothetical = hypothetical_exits(prediction, minutes)
        for policy, cutoff, quota in (
            ("daily_quota_tuned", None, True),
            ("strict_066", 0.66, False),
            ("daily_schedule_benchmark", 0.0, True),
        ):
            policy_frame = hypothetical
            if policy == "daily_schedule_benchmark":
                policy_frame = hypothetical.copy()
                positions = policy_frame["max_exit_position"].to_numpy(dtype=int)
                policy_frame["exit_time"] = minutes.index[positions]
                policy_frame["exit_price"] = minutes["open"].to_numpy()[positions]
                policy_frame["exit_position"] = positions
                policy_frame["observed_minutes_held"] = positions - policy_frame[
                    "entry_position"
                ].to_numpy(dtype=int)
                policy_frame["exit_reason"] = "fixed_horizon_benchmark"
            trades = daily_policy(policy_frame, cutoff, quota)
            stats = trade_summary(trades, session_denominator)
            stress = trade_summary(trades, session_denominator, cost=50)
            equity = marked_equity(trades, minutes)
            valid = prediction["target"].notna()
            result = {
                "horizon": name,
                "model": model_name,
                "policy": policy,
                **stats,
                "double_cost_net": stress["net_dollars"],
                "daily_drawdown": float(equity["drawdown"].max()),
                "accuracy": accuracy_score(
                    prediction.loc[valid, "target"], prediction.loc[valid, "probability"].ge(0.5)
                ),
                "brier": brier_score_loss(
                    prediction.loc[valid, "target"], prediction.loc[valid, "probability"]
                ),
                "frequency_pass": stats["missing_days"] == 0,
                "roll_verification": "unverified_legacy_continuous_series",
            }
            summary.append(result)
            trades.to_csv(out / f"trades_{name}_{model_name}_{policy}.csv", index=False)
            equity.to_csv(out / f"equity_{name}_{model_name}_{policy}.csv", index=False)
            counts = trades.groupby("session").size() if not trades.empty else pd.Series(dtype=int)
            counts.reindex(session_denominator, fill_value=0).rename("entries").to_csv(
                out / f"coverage_{name}_{model_name}_{policy}.csv"
            )
            if not trades.empty:
                for fold, group in trades.groupby("fold"):
                    folds.append(
                        {
                            "horizon": name,
                            "model": model_name,
                            "policy": policy,
                            "fold": fold,
                            "trades": len(group),
                            "net_dollars": float(
                                ((group.exit_price - group.entry_price) * 20 - 25).sum()
                            ),
                        }
                    )
    pd.DataFrame(summary).to_csv(out / f"summary_{name}.csv", index=False)
    print(f"{name}: complete", flush=True)
    return summary, folds


def main() -> None:
    out = Path(__file__).parent / "medium_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "source_snapshot.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    manifest = {
        "status": "running",
        "development_only": True,
        "min_trading_minutes": 240,
        "max_trading_sessions": 5,
        "max_contracts": 5,
        "entry_latency_minutes": 1,
        "models": list(MODELS),
        "features": 100,
        "selected_features": 20,
        "source_authority": "NQU6_Export.csv legacy continuous, roll mapping unverified",
        "fresh_holdout_available": False,
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries, fold_results = [], []
    try:
        print("Preparing local hourly features", flush=True)
        minutes, grid, factors = prepare()
        for days in (0, 1, 3, 5):
            summary, folds = run_horizon(days, minutes, grid, factors, out)
            summaries.extend(summary)
            fold_results.extend(folds)
            pd.DataFrame(summaries).to_csv(out / "combined_summary.csv", index=False)
            pd.DataFrame(fold_results).to_csv(out / "fold_results.csv", index=False)
        manifest["status"] = "complete_development_provisional"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", repr(error)
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Output: {out}", flush=True)


if __name__ == "__main__":
    main()
