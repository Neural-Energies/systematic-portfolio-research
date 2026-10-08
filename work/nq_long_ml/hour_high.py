"""Next-hour NQ HIGH regression, all complete hourly periods; no trading policy."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from experiment import END, SEED, aggregate, load_minutes
from fifteen_probability import BASE, prepare
from medium_frequency import FIRST_TEST, session_keys
from range_forecast import carry_completed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def high_labels(minutes: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """Target is max high of [forecast, forecast+60m); reference uses earlier minutes."""
    bars = aggregate(minutes, 60)
    clock = bars.index
    # A minute's OHLC is usable only after it completes, never at its timestamped open.
    known = minutes[["close"]].rename(columns={"close": "reference"}).copy()
    known["known_at"] = known.index + pd.Timedelta(minutes=1)
    known.index = pd.DatetimeIndex(known.known_at)
    reference = carry_completed(known, clock, 96)
    out = pd.DataFrame(index=clock)
    out["actual_high"] = bars.high
    out["label_end"] = bars.available_at
    out["reference"] = reference.reference
    out["known_at"] = reference.known_at
    out["reference_age_minutes"] = (
        pd.Series(clock, index=clock) - out.known_at
    ).dt.total_seconds() / 60
    out["scale"] = x.reindex(clock).scale * 2
    out["move_points"] = out.actual_high - out.reference
    out["target"] = out.move_points / out.scale
    out["session"] = session_keys(clock)
    return out.loc[out.scale.notna() & out.reference.notna() & out.scale.gt(0)]


def feature_table(
    minutes: pd.DataFrame, previous: pd.DataFrame, forecast_clock: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Complete past feature snapshots plus clock features and realized semivariance."""
    clock = aggregate(minutes, 60).index if forecast_clock is None else forecast_clock
    nq_cols = [c for c in previous if not c.startswith("market_") and not c.startswith("rule_")]
    # Carry prior NQ state across closures and explicitly expose its age.
    past = previous[nq_cols].copy()
    past["snapshot_at"] = past.index
    x = carry_completed(past, clock, 96)
    x["snapshot_age_minutes"] = (
        pd.Series(clock, index=clock) - x.pop("snapshot_at")
    ).dt.total_seconds() / 60
    market_cols = [c for c in previous if c.startswith("market_")]
    x = x.join(previous[market_cols].reindex(clock))
    local = clock.tz_convert("America/New_York")
    minute_of_day = local.hour * 60 + local.minute
    x["forecast_clock_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    x["forecast_clock_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    x["forecast_weekday"] = local.dayofweek
    x["forecast_rth"] = ((minute_of_day >= 570) & (minute_of_day < 960)).astype(float)
    ret = np.log(minutes.close / minutes.open)
    micro = pd.DataFrame(index=minutes.index)
    for hours in (1, 4, 24):
        window = f"{hours}h"
        variance = ret.pow(2).rolling(window, min_periods=15).sum()
        micro[f"realized_sigma_{hours}h"] = np.sqrt(variance / hours)
        up = ret.clip(lower=0).pow(2).rolling(window, min_periods=15).sum()
        down = ret.clip(upper=0).pow(2).rolling(window, min_periods=15).sum()
        micro[f"semivariance_balance_{hours}h"] = (up - down) / variance.replace(0, np.nan)
        micro[f"observed_minutes_{hours}h"] = ret.rolling(window, min_periods=15).count() / (
            60 * hours
        )
    micro.index = micro.index + pd.Timedelta(minutes=1)
    x = x.join(carry_completed(micro, clock, 96))
    return x.replace([np.inf, -np.inf], np.nan)


def temporal_parts(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    a, b = start - pd.Timedelta(days=56), start - pd.Timedelta(days=28)
    intervals = [(start - pd.Timedelta(days=350), a), (a, b), (b, start), (start, end)]
    return [
        data.loc[(data.index >= lo) & (data.index < hi) & data.label_end.lt(hi)].copy()
        for lo, hi in intervals
    ]


def clustered_columns(fit: pd.DataFrame) -> list[str]:
    """Unsupervised redundancy reduction: fit-only absolute Spearman correlations."""
    usable = [c for c in fit if fit[c].notna().sum() >= 60 and fit[c].nunique() > 1]
    if len(usable) <= 1:
        return usable
    corr = fit[usable].corr(method="spearman").fillna(0).abs()
    distance = np.sqrt(np.maximum(0, (1 - corr.to_numpy()) / 2))
    np.fill_diagonal(distance, 0)
    groups = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=np.sqrt((1 - 0.80) / 2),
        criterion="distance",
    )
    representatives = []
    for label in np.unique(groups):
        members = [c for c, group in zip(usable, groups, strict=True) if group == label]
        # Medoid among best-covered members; outcome labels never enter the clustering.
        coverage = fit[members].notna().mean()
        eligible = coverage.index[coverage >= coverage.max() - 0.05]
        representative = corr.loc[eligible, members].mean(axis=1).idxmax()
        representatives.append(representative)
    return representatives


def model_catalog():
    return {
        "ridge": Ridge(alpha=100),
        "elastic_net": ElasticNet(alpha=0.02, l1_ratio=0.2, max_iter=3000, random_state=SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=120, max_depth=10, min_samples_leaf=30, n_jobs=4, random_state=SEED
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=120, max_depth=12, min_samples_leaf=25, n_jobs=4, random_state=SEED
        ),
        "boost_mean": HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=10,
            early_stopping=False,
            random_state=SEED,
        ),
        "boost_median": HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=10,
            early_stopping=False,
            random_state=SEED,
        ),
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray, reference: np.ndarray) -> dict:
    error = predicted - actual
    return {
        "observations": len(actual),
        "mae_points": mean_absolute_error(actual, predicted),
        "mse_points_squared": mean_squared_error(actual, predicted),
        "rmse_points": np.sqrt(mean_squared_error(actual, predicted)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.9)),
        "bias_points": float(error.mean()),
        "within_10_points": float((np.abs(error) <= 10).mean()),
        "within_20_points": float((np.abs(error) <= 20).mean()),
        "within_25_points": float((np.abs(error) <= 25).mean()),
        "within_50_points": float((np.abs(error) <= 50).mean()),
        "actual_below_forecast": float((actual <= predicted).mean()),
        "move_r2": r2_score(actual - reference, predicted - reference),
    }


def corrected_quantiles(
    predicted: np.ndarray, actual: np.ndarray, quantiles: np.ndarray
) -> np.ndarray:
    """Calibration-period residual shifts; coverage is evaluated, not guaranteed."""
    return np.array(
        [np.quantile(actual - predicted[:, i], q, method="higher") for i, q in enumerate(quantiles)]
    )


def run(folds: list[int] | None = None, threads: int = 4):
    suffix = "" if folds is None else "_folds_" + "-".join(map(str, folds))
    out = BASE / "hour_high_runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + suffix)
    out.mkdir(parents=True)
    _, previous, _, _ = prepare()
    minutes = load_minutes("NQ")
    x = feature_table(minutes, previous)
    data = high_labels(minutes, x)
    protocol = {
        "task": "Forecast maximum NQ high over next complete 60 minutes at each hourly open",
        "reference": "last completed minute close; next-hour open is NOT a predictor",
        "target": "(future hourly high - reference)/past-only range scale; reconstruct price",
        "models": list(model_catalog()),
        "groups": ["nq_only", "nq_es", "all_clustered"],
        "quantiles": [0.1, 0.5, 0.8, 0.9],
        "selection": "point model by earlier tune MAE; quantile group by earlier average pinball",
        "splits": (
            "Monthly: fit past 350d to -56d; calibrate -56d/-28d; "
            "tune -28d/start; purge every label at boundary"
        ),
        "goal": "absolute high forecast error <=20 NQ points, with monthly consistency",
        "data": "Reused development 2023-08-22/2025-08-16; no fresh holdout; no sealed data",
        "roll": "NQ legacy continuous export without verified dated volume-roll map",
        "source_papers": [
            "https://arxiv.org/abs/2005.12483",
            "https://doi.org/10.1016/j.jempfin.2023.03.001",
            "https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/0210.pdf",
            "https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html",
        ],
        "seed": SEED,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    data.to_parquet(out / "labels.parquet")
    print(f"RUN {out}: {len(data)} complete hourly labels; {x.shape[1]} predictors", flush=True)
    boundaries = list(pd.date_range(FIRST_TEST, periods=12, freq=pd.DateOffset(months=1))) + [END]
    tuning, forecasts, selected, quantile_forecasts, features = [], [], [], [], []
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        if folds is not None and fold not in folds:
            continue
        fit, cal, tune, test = temporal_parts(data, start, end)
        usable = [
            c
            for c in x
            if x.loc[fit.index, c].notna().sum() >= 60 and x.loc[fit.index, c].nunique() > 1
        ]
        groups = {
            "nq_only": [c for c in usable if not c.startswith("market_")],
            "nq_es": [
                c for c in usable if not c.startswith("market_") or c.startswith("market_ES_")
            ],
            "all_clustered": clustered_columns(x.loc[fit.index, usable]),
        }
        candidates, quantile_candidates = {}, {}
        # Honest baselines are estimated only on FIT, and predict every test hour.
        for name, q in (("historical_median", 0.5), ("historical_mean", None)):
            normalized = (
                float(fit.target.quantile(q)) if q is not None else float(fit.target.mean())
            )
            pred = test.reference + normalized * test.scale
            frame = test[["actual_high", "reference", "scale", "session", "label_end"]].copy()
            frame["predicted_high"], frame["config"], frame["fold"] = pred, name, fold
            forecasts.append(frame)
        for name, pred in (
            ("last_price", test.reference),
            (
                "volatility_maximum",
                test.reference * (1 + x.loc[test.index, "realized_sigma_1h"] * np.sqrt(2 / np.pi)),
            ),
        ):
            frame = test[["actual_high", "reference", "scale", "session", "label_end"]].copy()
            frame["predicted_high"], frame["config"], frame["fold"] = pred, name, fold
            forecasts.append(frame)
        # Same forecast clock baseline shrunk toward the global median for small samples.
        slots = fit.index.tz_convert("America/New_York").hour
        table = fit.target.groupby(slots).agg(["median", "count"])
        weight = table["count"] / (table["count"] + 30)
        table["value"] = weight * table["median"] + (1 - weight) * fit.target.median()
        value = (
            pd.Series(test.index.tz_convert("America/New_York").hour)
            .map(table.value)
            .fillna(fit.target.median())
            .to_numpy()
        )
        frame = test[["actual_high", "reference", "scale", "session", "label_end"]].copy()
        frame["predicted_high"], frame["config"], frame["fold"] = (
            test.reference + value * test.scale,
            "clock_median",
            fold,
        )
        forecasts.append(frame)
        for group, cols in groups.items():
            features.extend({"fold": fold, "group": group, "feature": c} for c in cols)
            for name, estimator in model_catalog().items():
                if "n_jobs" in estimator.get_params():
                    estimator.set_params(n_jobs=threads)
                print(f"month {fold}: {group} {name}", flush=True)
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    clone(estimator),
                )
                model.fit(x.loc[fit.index, cols], fit.target)
                pu = (
                    tune.reference.to_numpy()
                    + model.predict(x.loc[tune.index, cols]) * tune.scale.to_numpy()
                )
                pt = (
                    test.reference.to_numpy()
                    + model.predict(x.loc[test.index, cols]) * test.scale.to_numpy()
                )
                config = f"{group}_{name}"
                tune_mae = mean_absolute_error(tune.actual_high, pu)
                tune_within20 = float((np.abs(tune.actual_high.to_numpy() - pu) <= 20).mean())
                tuning.append(
                    {
                        "fold": fold,
                        "config": config,
                        "tune_mae": tune_mae,
                        "tune_within20": tune_within20,
                        "fit_end": str(fit.label_end.max()),
                        "cal_end": str(cal.label_end.max()),
                        "tune_end": str(tune.label_end.max()),
                        "test_start": str(test.index.min()),
                    }
                )
                frame = test[["actual_high", "reference", "scale", "session", "label_end"]].copy()
                frame["predicted_high"], frame["config"], frame["fold"] = pt, config, fold
                forecasts.append(frame)
                candidates[config] = (tune_mae, frame, model, cols, tune_within20)
            qs = np.array([0.1, 0.5, 0.8, 0.9])
            pc, pu, pt, fitted_quantiles = [], [], [], []
            for q in qs:
                print(f"month {fold}: {group} quantile {q}", flush=True)
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=q,
                        max_iter=120,
                        max_leaf_nodes=7,
                        min_samples_leaf=40,
                        l2_regularization=10,
                        early_stopping=False,
                        random_state=SEED,
                    ),
                )
                model.fit(x.loc[fit.index, cols], fit.target)
                fitted_quantiles.append(model)
                for target, part in ((pc, cal), (pu, tune), (pt, test)):
                    target.append(model.predict(x.loc[part.index, cols]))
            shifts = corrected_quantiles(np.column_stack(pc), cal.target.to_numpy(), qs)
            normalized_tune = np.sort(np.column_stack(pu) + shifts, axis=1)
            normalized_test = np.sort(np.column_stack(pt) + shifts, axis=1)
            losses = [
                mean_pinball_loss(tune.target, normalized_tune[:, i], alpha=q)
                for i, q in enumerate(qs)
            ]
            qframe = test[["actual_high", "reference", "scale", "session", "label_end"]].copy()
            for i, q in enumerate(qs):
                qframe[f"q{int(q * 100)}"] = test.reference + normalized_test[:, i] * test.scale
            qframe["config"], qframe["fold"] = group, fold
            quantile_forecasts.append(qframe)
            quantile_candidates[group] = (
                float(np.mean(losses)),
                qframe,
                fitted_quantiles,
                shifts,
                cols,
            )
            tuning.append(
                {
                    "fold": fold,
                    "config": f"{group}_quantiles",
                    "tune_pinball": float(np.mean(losses)),
                    "fit_end": str(fit.label_end.max()),
                    "cal_end": str(cal.label_end.max()),
                    "tune_end": str(tune.label_end.max()),
                    "test_start": str(test.index.min()),
                }
            )
        winner = min(candidates, key=lambda k: candidates[k][0])
        point = candidates[winner][1].copy()
        point["policy"] = "earlier_selected"
        selected.append(point)
        tolerance_winner = max(candidates, key=lambda k: (candidates[k][4], -candidates[k][0]))
        tolerance_point = candidates[tolerance_winner][1].copy()
        tolerance_point["policy"] = "earlier_selected_within20"
        selected.append(tolerance_point)
        qw = min(quantile_candidates, key=lambda k: quantile_candidates[k][0])
        chosen_quantile = quantile_candidates[qw][1].copy()
        chosen_quantile["config"] = "earlier_selected"
        quantile_forecasts.append(chosen_quantile)
        joblib.dump(
            {
                "model": candidates[winner][2],
                "columns": candidates[winner][3],
                "config": winner,
                "quantile_models": quantile_candidates[qw][2],
                "quantile_columns": quantile_candidates[qw][4],
                "quantile_shifts": quantile_candidates[qw][3],
                "quantile_group": qw,
                "fit_end": str(fit.label_end.max()),
                "test_start": str(test.index.min()),
            },
            out / f"model_month_{fold:02d}.joblib",
        )
        pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)
        pd.concat(forecasts).to_parquet(out / "forecasts.parquet")
        pd.concat(selected).to_parquet(out / "selected.parquet")
        pd.concat(quantile_forecasts).to_parquet(out / "quantiles.parquet")
        pd.DataFrame(features).to_csv(out / "features.csv", index=False)
        print(f"month {fold} completed: point={winner}; quantiles={qw}", flush=True)
    all_forecasts = pd.concat(forecasts + selected)
    chosen_mask = all_forecasts["policy"].notna()
    all_forecasts.loc[chosen_mask, "config"] = all_forecasts.loc[chosen_mask, "policy"]
    summaries = [
        {
            "config": config,
            **regression_metrics(
                g.actual_high.to_numpy(), g.predicted_high.to_numpy(), g.reference.to_numpy()
            ),
        }
        for config, g in all_forecasts.groupby("config")
    ]
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", help="Comma-separated independent monthly folds, from 1 to 12")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    requested = None if args.folds is None else [int(n) for n in args.folds.split(",")]
    if requested is not None and (not requested or any(n < 1 or n > 12 for n in requested)):
        parser.error("folds must be between 1 and 12")
    if args.threads < 1:
        parser.error("threads must be positive")
    with threadpool_limits(limits=args.threads):
        run(requested, args.threads)
