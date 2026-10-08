"""Quarterly retrained, mixed-frequency NQ direction and favorable-excursion models.

Research only: the continuous NQ export has no verified dated volume-roll map.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from experiment import END, ROOT, SEED, aggregate, block_importance, load_minutes
from medium_frequency import (
    FIRST_TEST,
    build_labels,
    daily_policy,
    hypothetical_exits,
    marked_equity,
    model_catalog,
    prepare,
    raw_scores,
    session_keys,
    trade_summary,
)
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.metrics import accuracy_score, brier_score_loss, mean_pinball_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

QUANTILES = (0.25, 0.50, 0.75)
FAMILIES = ("historical_quantile", "linear_quantile", "boost_shallow", "boost_flexible")


def daily_bars(minutes: pd.DataFrame) -> pd.DataFrame:
    """Daily information becomes usable only after the scheduled 16:00 CT close."""
    key = session_keys(minutes.index)
    bars = (
        minutes.assign(session=key)
        .groupby("session")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            count=("close", "size"),
        )
    )
    bars = bars.loc[bars["count"].ge(240)].copy()
    bars["available_at"] = (
        (bars.index + pd.Timedelta(days=1, hours=16))
        .tz_localize("America/Chicago")
        .tz_convert("UTC")
    )
    return bars


def carry_completed(values: pd.DataFrame, clock: pd.DatetimeIndex, hours: float) -> pd.DataFrame:
    """Backward as-of alignment with an explicit maximum age; no future backfill."""
    values = values.sort_index().copy()
    values["_available"] = values.index
    aligned = values.reindex(values.index.union(clock)).sort_index().ffill().reindex(clock)
    age = (pd.Series(clock, index=clock) - aligned.pop("_available")).dt.total_seconds() / 3600
    return aligned.where(age.le(hours), np.nan, axis=0)


def compact_features(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ret = np.log(bars["close"] / bars["open"])
    volume = bars["volume"].astype(float)
    values = pd.DataFrame(
        {
            f"{prefix}_return": ret,
            f"{prefix}_momentum4": ret.rolling(4).sum(),
            f"{prefix}_volatility16": ret.rolling(16).std(),
            f"{prefix}_volume_z16": (volume - volume.rolling(16).mean()) / volume.rolling(16).std(),
            f"{prefix}_range16": ((bars["high"] - bars["low"]) / bars["close"]).rolling(16).mean(),
        }
    )
    values.index = pd.DatetimeIndex(bars["available_at"])
    return values.replace([np.inf, -np.inf], np.nan)


def mixed_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    minutes, grid, factors = prepare()
    clock = grid.index
    # 66 hourly technical + 4 ARIMA + 15 mixed-period NQ + 12 ES + 3 PCA = 100.
    columns = [
        c
        for c in grid
        if c.startswith("arima_")
        or (not c.startswith(("factor_", "slow_")) and c != "feature_age_hours")
    ]
    x = grid[columns].copy()
    assert len(columns) == 70
    for duration, prefix, age in ((15, "nq15", 1), (240, "nq4h", 8), (0, "nqday", 96)):
        bars = daily_bars(minutes) if duration == 0 else aggregate(minutes, duration)
        x = x.join(carry_completed(compact_features(bars, prefix), clock, age))
    es = load_minutes("ES")
    es_hour = aggregate(es, 60)
    es_compact = carry_completed(compact_features(es_hour, "es_hour"), clock, 3)
    x = x.join(es_compact.drop(columns="es_hour_range16"))
    for col in ("factor_beta_ES", "factor_alpha_ES", "factor_residual_ES", "factor_correlation_ES"):
        x[col] = grid[col]
    x["es_relative_strength4"] = x["momentum_4"] - x["es_hour_momentum4"]
    for duration, prefix, age in ((240, "es4h", 8), (0, "esday", 96)):
        bars = daily_bars(es) if duration == 0 else aggregate(es, duration)
        value = compact_features(bars, prefix)[[f"{prefix}_return"]]
        x = x.join(carry_completed(value, clock, age))
    x["es_hour_available"] = x["es_hour_return"].notna().astype(float)
    assert x.shape[1] == 97
    nq_hour = aggregate(minutes, 60)
    span = nq_hour["high"] - nq_hour["low"]
    prev = nq_hour["close"].shift(1)
    tr = pd.concat(
        [span, (nq_hour["high"] - prev).abs(), (nq_hour["low"] - prev).abs()], axis=1
    ).max(axis=1)
    scale = pd.DataFrame({"scale": tr.rolling(16).mean().clip(lower=2)})
    scale.index = pd.DatetimeIndex(nq_hour["available_at"])
    scale = carry_completed(scale, clock, 3)["scale"]
    valid = scale.notna() & x["nqday_volatility16"].notna()
    return minutes, x.loc[valid], factors.reindex(x.index[valid]), scale.loc[valid]


def excursion_labels(frame: pd.DataFrame, minutes: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """Future high is a training label, never a price chosen by an execution rule."""
    result = frame.copy()
    result["scale"] = scale.reindex(frame.index)
    high, low, opened = [minutes[c].to_numpy() for c in ("high", "low", "open")]
    mfe, mae = np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    for i, row in enumerate(result.itertuples()):
        if not row.execution_valid:
            continue
        a, b = int(row.entry_position), int(row.max_exit_position)
        if b <= a:
            continue
        # At the terminal minute open the trade is already closed; its high is excluded.
        mfe[i] = max(0.0, high[a:b].max() - opened[a]) / row.scale
        mae[i] = max(0.0, opened[a] - low[a:b].min()) / row.scale
    result["mfe"], result["mae"] = mfe, mae
    return result


def split_training(data: pd.DataFrame, start: pd.Timestamp, target: str):
    train = data.loc[(data.index < start) & data["label_end"].lt(start) & data[target].notna()]
    cuts = [0] + [int(len(train) * f) for f in (0.55, 0.70, 0.85)] + [len(train)]
    parts = [train.iloc[a:b].copy() for a, b in zip(cuts[:-1], cuts[1:], strict=True)]
    for i in range(3):
        parts[i] = parts[i].loc[parts[i]["label_end"].lt(parts[i + 1].index.min())]
    if any(len(p) < 80 for p in parts):
        raise ValueError("Insufficient mature training labels")
    return parts


def pca_matrices(x: pd.DataFrame, factors: pd.DataFrame, parts: list[pd.DataFrame], use_es: bool):
    markets = list(factors) if use_es else [c for c in factors if c != "ES"]
    pca = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), PCA(n_components=3, random_state=SEED)
    )
    pca.fit(factors.reindex(parts[0].index)[markets])
    base = list(x) if use_es else [c for c in x if not (c.startswith("es") or c.endswith("_ES"))]
    matrices = []
    for part in parts:
        matrix = x.reindex(part.index)[base].copy()
        scores = pca.transform(factors.reindex(part.index)[markets])
        for k in range(3):
            matrix[f"pca_factor_{k + 1}"] = scores[:, k]
        matrices.append(matrix)
    return matrices, pca, markets


def regressor(family: str, quantile: float):
    if family == "linear_quantile":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            QuantileRegressor(quantile=quantile, alpha=0.015, solver="highs"),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            max_iter=100,
            max_leaf_nodes=7 if family == "boost_shallow" else 15,
            min_samples_leaf=50,
            l2_regularization=10,
            early_stopping=False,
            random_state=SEED,
        ),
    )


def range_importance(xf, yf, xs, ys) -> pd.Series:
    screen = regressor("boost_shallow", 0.5).fit(xf, yf)
    base = mean_pinball_loss(ys, screen.predict(xs), alpha=0.5)
    rng = np.random.default_rng(SEED)
    blocks = np.array_split(np.arange(len(xs)), max(2, len(xs) // 32))
    rank = {}
    for col in xs:
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        shuffled = xs.copy()
        shuffled[col] = xs[col].to_numpy()[order]
        rank[col] = mean_pinball_loss(ys, screen.predict(shuffled), alpha=0.5) - base
    return pd.Series(rank).sort_values(ascending=False)


def quantile_predict(models: list, x: pd.DataFrame, offset: np.ndarray) -> np.ndarray:
    raw = np.column_stack(
        [np.full(len(x), model) if np.isscalar(model) else model.predict(x) for model in models]
    )
    # Monotone rearrangement removes crossing; calibration is empirical, not a guarantee.
    return np.sort(np.maximum(raw + offset, 0), axis=1)


def pinball_score(y: pd.Series, values: np.ndarray) -> float:
    return float(
        np.mean([mean_pinball_loss(y, values[:, i], alpha=q) for i, q in enumerate(QUANTILES)])
    )


def fit_direction(data, x, factors, test, start, fold, out):
    parts = split_training(data, start, "target")
    candidates, forecasts, rankings = [], [], []
    for use_es in (False, True):
        matrices, pca, markets = pca_matrices(x, factors, parts + [test], use_es)
        xf, xs, xc, xu, xt = matrices
        screen = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), model_catalog()["rf"]
        )
        screen.fit(xf, parts[0]["target"])
        rank = block_importance(screen, xs, parts[1]["target"])
        selected = list(rank.head(20).index)
        rankings.extend(
            {
                "fold": fold,
                "use_es": use_es,
                "feature": c,
                "importance": v,
                "selected": c in selected,
            }
            for c, v in rank.items()
        )
        for name, estimator in model_catalog().items():
            print(f"Direction fold {fold}: ES={use_es}, {name}", flush=True)
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), clone(estimator)
            )
            model.fit(pd.concat([xf, xs])[selected], pd.concat(parts[:2])["target"])
            cal = LogisticRegression(C=1, max_iter=1000)
            cal.fit(raw_scores(model, xc[selected]).reshape(-1, 1), parts[2]["target"])
            tune_p = cal.predict_proba(raw_scores(model, xu[selected]).reshape(-1, 1))[:, 1]
            p = cal.predict_proba(raw_scores(model, xt[selected]).reshape(-1, 1))[:, 1]
            valid = test["target"].notna()
            row = {
                "fold": fold,
                "model": name,
                "use_es": use_es,
                "selection_brier": brier_score_loss(parts[3]["target"], tune_p),
                "test_brier": brier_score_loss(test.loc[valid, "target"], p[valid]),
                "test_accuracy": accuracy_score(test.loc[valid, "target"], p[valid] >= 0.5),
                "test_majority": max(
                    test.loc[valid, "target"].mean(), 1 - test.loc[valid, "target"].mean()
                ),
                "selected_features": selected,
            }
            candidates.append(row)
            prediction = test.drop(columns=list(x), errors="ignore").copy()
            (
                prediction["probability"],
                prediction["model"],
                prediction["use_es"],
                prediction["fold"],
            ) = p, name, use_es, fold
            forecasts.append(prediction)
            joblib.dump(
                {
                    "model": model,
                    "calibrator": cal,
                    "pca": pca,
                    "markets": markets,
                    "selected": selected,
                    "trained_before": str(start),
                },
                out / f"direction_{fold}_{name}_es{use_es}.joblib",
            )
    best = min(candidates, key=lambda r: r["selection_brier"])
    prediction = next(
        f
        for f in forecasts
        if f["model"].iloc[0] == best["model"] and f["use_es"].iloc[0] == best["use_es"]
    )
    for r in candidates:
        r["chosen_before_test"] = r["model"] == best["model"] and r["use_es"] == best["use_es"]
    pd.concat(forecasts).to_parquet(out / f"direction_candidates_{fold}.parquet")
    return prediction, candidates, rankings


def fit_range(data, x, factors, test, start, fold, name, out):
    parts = split_training(data, start, "mfe")
    candidates, forecasts, rankings = [], [], []
    for use_es in (False, True):
        matrices, pca, markets = pca_matrices(x, factors, parts + [test], use_es)
        xf, xs, xc, xu, xt = matrices
        rank = range_importance(xf, parts[0]["mfe"], xs, parts[1]["mfe"])
        selected = list(rank.head(20).index)
        rankings.extend(
            {
                "horizon": name,
                "fold": fold,
                "use_es": use_es,
                "feature": c,
                "importance": v,
                "selected": c in selected,
            }
            for c, v in rank.items()
        )
        for family in FAMILIES:
            print(f"Range {name} fold {fold}: ES={use_es}, {family}", flush=True)
            yf = pd.concat(parts[:2])["mfe"]
            models = [
                float(yf.quantile(q))
                if family == "historical_quantile"
                else regressor(family, q).fit(pd.concat([xf, xs])[selected], yf)
                for q in QUANTILES
            ]
            raw_cal = quantile_predict(models, xc[selected], np.zeros(3))
            offset = np.array(
                [
                    np.quantile(parts[2]["mfe"].to_numpy() - raw_cal[:, i], q)
                    for i, q in enumerate(QUANTILES)
                ]
            )
            tuned = quantile_predict(models, xu[selected], offset)
            values = quantile_predict(models, xt[selected], offset)
            valid = test["mfe"].notna()
            row = {
                "horizon": name,
                "fold": fold,
                "family": family,
                "use_es": use_es,
                "selection_pinball": pinball_score(parts[3]["mfe"], tuned),
                "test_pinball": pinball_score(test.loc[valid, "mfe"], values[valid]),
                "selected_features": selected,
            }
            for i, q in enumerate(QUANTILES):
                row[f"coverage_q{int(q * 100)}"] = float(
                    (test.loc[valid, "mfe"].to_numpy() <= values[valid, i]).mean()
                )
                row[f"touch_rate_q{int(q * 100)}"] = float(
                    (test.loc[valid, "mfe"].to_numpy() >= values[valid, i]).mean()
                )
            candidates.append(row)
            prediction = test.drop(columns=list(x), errors="ignore").copy()
            for i, q in enumerate(QUANTILES):
                prediction[f"q{int(q * 100)}"] = values[:, i]
            prediction["range_model"], prediction["use_es"], prediction["fold"] = (
                family,
                use_es,
                fold,
            )
            forecasts.append(prediction)
            joblib.dump(
                {
                    "models": models,
                    "offset": offset,
                    "pca": pca,
                    "markets": markets,
                    "selected": selected,
                    "trained_before": str(start),
                },
                out / f"range_{name}_{fold}_{family}_es{use_es}.joblib",
            )
    best = min(candidates, key=lambda r: r["selection_pinball"])
    prediction = next(
        f
        for f in forecasts
        if f["range_model"].iloc[0] == best["family"] and f["use_es"].iloc[0] == best["use_es"]
    )
    for r in candidates:
        r["chosen_before_test"] = r["family"] == best["family"] and r["use_es"] == best["use_es"]
    pd.concat(forecasts).to_parquet(out / f"range_candidates_{name}_{fold}.parquet")
    return prediction, candidates, rankings


def target_exits(frame: pd.DataFrame, minutes: pd.DataFrame, quantile: int) -> pd.DataFrame:
    """Freeze the target at entry. A minute high can test touch, never define the target."""
    result = hypothetical_exits(frame, minutes)
    high = minutes["high"].to_numpy()
    target_prices = (
        np.ceil((result["entry_price"] + result[f"q{quantile}"] * result["scale"]) * 4) / 4
    )
    result["target_price"] = target_prices
    result["target_quantile"] = quantile
    for decision, row in result.iterrows():
        if not row.execution_valid:
            continue
        # Preserve a four-trading-hour minimum. Touches before then are ineligible.
        a = int(row.entry_position) + 240
        b = int(row.exit_position)
        target = float(row.target_price)
        hits = np.flatnonzero(high[a:b] >= target)
        if len(hits):
            j = a + int(hits[0])
            # Limit fill uses target even when the market gaps favorably: conservative.
            fill = target
            result.loc[
                decision,
                [
                    "exit_position",
                    "exit_time",
                    "exit_price",
                    "exit_reason",
                    "observed_minutes_held",
                ],
            ] = [j, minutes.index[j], fill, "predicted_target", j - int(row.entry_position)]
    return result


def evaluate(frame: pd.DataFrame, minutes: pd.DataFrame, name: str, out: Path):
    sessions = pd.DatetimeIndex(
        session_keys(minutes.loc[minutes.index >= FIRST_TEST].index).unique()
    )
    summary = []
    for exit_policy in ("direction_only", "q25", "q50", "q75"):
        simulated = (
            hypothetical_exits(frame, minutes)
            if exit_policy == "direction_only"
            else target_exits(frame, minutes, int(exit_policy[1:]))
        )
        for entry_policy, threshold, quota in (
            ("strict_066", 0.66, False),
            ("daily_quota", 0.55, True),
        ):
            trades = daily_policy(simulated, threshold, quota, conservative_roll_guard=False)
            equity = marked_equity(trades, minutes)
            stats = trade_summary(trades, sessions)
            row = {
                "horizon": name,
                "exit_policy": exit_policy,
                "entry_policy": entry_policy,
                **stats,
                "double_cost_net": trade_summary(trades, sessions, cost=50)["net_dollars"],
                "daily_drawdown": float(equity["drawdown"].max()),
                "frequency_pass": stats["missing_days"] == 0,
                "target_exit_fraction": float(trades["exit_reason"].eq("predicted_target").mean())
                if not trades.empty
                else 0,
                "status": "development_only_volume_roll_mapping_unverified",
            }
            summary.append(row)
            if not trades.empty:
                assert trades["observed_minutes_held"].ge(240).all()
                assert (
                    trades["entry_time"] >= trades["decision_time"] + pd.Timedelta(minutes=1)
                ).all()
                assert trades["concurrent_at_entry"].le(5).all()
            trades.to_csv(out / f"trades_{name}_{entry_policy}_{exit_policy}.csv", index=False)
            equity.to_csv(out / f"equity_{name}_{entry_policy}_{exit_policy}.csv", index=False)
    return summary


def main():
    out = Path(__file__).parent / "range_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "source_snapshot.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    manifest = {
        "status": "running",
        "development_only": True,
        "fresh_holdout_available": False,
        "retrain": "quarterly_expanding_purged",
        "mixed_periods_minutes": [15, 60, 240, 1440],
        "range_families": list(FAMILIES),
        "quantiles": QUANTILES,
        "direction_models": list(model_catalog()),
        "roll_rule": "volume_based_user_specified",
        "roll_map_verified": False,
        "min_observed_minutes_held": 240,
        "max_sessions": 5,
        "max_contracts": 5,
        "cost_per_roundtrip": 25,
        "entry_delay_minutes": 1,
        "entry_policy": "strict66 comparator; daily quota55 with disclosed fallback",
        "target_freeze": "entry",
        "target_touch_before_minimum_hold": "ignored",
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    direction_rows, range_rows, direction_ranks, range_ranks = [], [], [], []
    directions, ranges = [], {name: [] for name in ("4h", "1d", "3d", "5d")}
    try:
        with threadpool_limits(limits=4):
            print("Building causal mixed-frequency NQ and ES inputs", flush=True)
            minutes, x, factors, scale = mixed_features()
            labels = {}
            for days in (0, 1, 3, 5):
                name = "4h" if days == 0 else f"{days}d"
                # No invented quarter-month roll rule. Historical P&L stays provisional.
                labels[name] = excursion_labels(
                    build_labels(minutes, x, days, conservative_roll_guard=False), minutes, scale
                )
            manifest["feature_candidates"] = len(x) + 3
            manifest["development_rows"] = len(x)
            manifest["es_hour_coverage"] = float(x["es_hour_return"].notna().mean())
            files = {
                symbol: str(
                    ROOT
                    / "data/processed/databento_research/development_minute_returns"
                    / f"symbol={symbol}/returns.parquet"
                )
                for symbol in ("NQ", "ES")
            }
            manifest["inputs"] = {
                s: {
                    "path": p,
                    "size": Path(p).stat().st_size,
                    "mtime_ns": Path(p).stat().st_mtime_ns,
                }
                for s, p in files.items()
            }
            boundaries = list(
                pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))
            ) + [END]
            for fold, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:], strict=True), 1
            ):
                test = labels["4h"].loc[(x.index >= start) & (x.index < end)].copy()
                direction, rows, rank = fit_direction(
                    labels["4h"], x, factors, test, start, fold, out
                )
                directions.append(direction)
                direction_rows.extend(rows)
                direction_ranks.extend(rank)
                pd.DataFrame(direction_rows).to_csv(out / "direction_diagnostics.csv", index=False)
                for name, data in labels.items():
                    test = data.loc[(x.index >= start) & (x.index < end)].copy()
                    prediction, rows, rank = fit_range(
                        data, x, factors, test, start, fold, name, out
                    )
                    prediction["probability"] = direction["probability"]
                    prediction["direction_model"] = direction["model"]
                    prediction["selected_threshold"] = 0.55
                    ranges[name].append(prediction)
                    range_rows.extend(rows)
                    range_ranks.extend(rank)
                    pd.DataFrame(range_rows).to_csv(out / "range_diagnostics.csv", index=False)
            pd.concat(directions).to_parquet(out / "chosen_direction.parquet")
            summaries = []
            for name, predictions in ranges.items():
                frame = pd.concat(predictions).sort_index()
                frame.to_parquet(out / f"chosen_range_{name}.parquet")
                summaries.extend(evaluate(frame, minutes, name, out))
                pd.DataFrame(summaries).to_csv(out / "trade_summary.csv", index=False)
            pd.DataFrame(direction_ranks).to_csv(out / "direction_feature_ranks.csv", index=False)
            pd.DataFrame(range_ranks).to_csv(out / "range_feature_ranks.csv", index=False)
            manifest["status"] = "complete_development_provisional"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", repr(error)
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Output: {out}", flush=True)


if __name__ == "__main__":
    main()
