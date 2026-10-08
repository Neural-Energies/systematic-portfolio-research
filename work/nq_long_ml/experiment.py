"""Isolated development-only NQ experiment. See README for the research contract."""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from statsmodels.tsa.arima.model import ARIMA

from systematic_research.research_partitions import guard_research_sample

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2023-08-22", tz="UTC")
END = pd.Timestamp("2025-08-16", tz="UTC")
SEED = 20261002
BENCHMARKS = ("ES", "ZN", "GC", "CL", "6J")


def load_minutes(symbol: str) -> pd.DataFrame:
    """Only development partitions are allowed; also predicate-filter dates."""
    path = (
        ROOT
        / "data/processed/databento_research/development_minute_returns"
        / f"symbol={symbol}/returns.parquet"
    )
    frame = pd.read_parquet(
        path,
        columns=["timestamp_utc", "open", "high", "low", "close", "volume"],
        filters=[("timestamp_utc", ">=", START), ("timestamp_utc", "<", END)],
    )
    ts = pd.to_datetime(frame["timestamp_utc"], utc=True)
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError(f"{symbol}: duplicate or unsorted timestamps")
    if frame[["open", "high", "low", "close"]].le(0).any().any():
        raise ValueError(f"{symbol}: nonpositive price")
    if not ts.between(START, END, inclusive="left").all():
        raise ValueError("Development boundary violated")
    frame = frame.set_index("timestamp_utc")
    stamps = pd.DatetimeIndex(frame.index)
    if stamps.tz is None:
        stamps = stamps.tz_localize("UTC")
    guard_research_sample(stamps, "fit")
    return frame


def aggregate(minutes: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Session-anchored complete bars, with timestamps denoting their open."""
    local = minutes.index.tz_convert("America/Chicago")
    session_date = (local.tz_localize(None) - pd.Timedelta(hours=17)).normalize()
    anchor = (session_date + pd.Timedelta(hours=17)).tz_localize("America/Chicago")
    elapsed = (minutes.index - anchor.tz_convert("UTC")).total_seconds() / 60
    keys = anchor.tz_convert("UTC") + pd.to_timedelta(
        np.floor(elapsed / horizon) * horizon, unit="min"
    )
    grouped = minutes.assign(bar_open=keys, session=anchor.tz_convert("UTC")).groupby("bar_open")
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "size"),
        session=("session", "first"),
    )
    # With unique timestamps, matching endpoints and count prove minute continuity.
    endpoints = pd.Series(minutes.index, index=minutes.index).groupby(keys).agg(["min", "max"])
    complete = bars["count"].eq(horizon)
    complete &= endpoints["min"].eq(bars.index)
    complete &= endpoints["max"].eq(bars.index + pd.Timedelta(minutes=horizon - 1))
    bars = bars.loc[complete].copy()
    bars["available_at"] = bars.index + pd.Timedelta(minutes=horizon)
    return bars


def arima_features(returns: pd.Series, sessions: pd.Series) -> pd.DataFrame:
    """Session-refitted ARIMA(1,0,0); fixed parameters update one-step forecasts."""
    out = pd.DataFrame(
        index=returns.index,
        columns=["arima_mean", "arima_sigma", "arima_z", "arima_phi"],
        dtype=float,
    )
    for _, positions in sessions.groupby(sessions).groups.items():
        first = returns.index.get_loc(positions[0])
        history = returns.iloc[max(0, first - 256) : first].dropna()
        if len(history) < 64:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(history.to_numpy(), order=(1, 0, 0), trend="c").fit()
        params = dict(zip(fit.param_names, fit.params, strict=True))
        if not fit.mle_retvals.get("converged", False):
            continue
        mean, phi = params["const"], params["ar.L1"]
        sigma = np.sqrt(max(params["sigma2"], 1e-16))
        forecast = mean + phi * (returns.loc[positions] - mean)
        out.loc[positions, "arima_mean"] = forecast
        out.loc[positions, "arima_sigma"] = sigma
        out.loc[positions, "arima_z"] = forecast / sigma
        out.loc[positions, "arima_phi"] = phi
    return out


def features(bars: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """60 technical, 20 factor-proxy and 4 ARIMA features, all backward-looking."""
    r = np.log(bars["close"] / bars["open"])
    close, volume = bars["close"], bars["volume"].astype(float)
    span = (bars["high"] - bars["low"]).replace(0, np.nan)
    values: dict[str, pd.Series] = {}
    for window in (2, 4, 8, 16, 32, 64):
        roll = r.rolling(window, min_periods=window)
        vol = roll.std().replace(0, np.nan)
        momentum = roll.sum()
        vroll = volume.rolling(window)
        er = close.ewm(span=window, adjust=False).mean()
        values[f"momentum_{window}"] = momentum
        values[f"volatility_{window}"] = vol
        values[f"return_z_{window}"] = r / vol
        values[f"efficiency_{window}"] = momentum / roll.apply(
            lambda x: np.abs(x).sum(), raw=True
        ).replace(0, np.nan)
        values[f"volume_z_{window}"] = (volume - vroll.mean()) / vroll.std().replace(0, np.nan)
        values[f"ema_distance_{window}"] = close / er - 1
        values[f"range_mean_{window}"] = (span / close).rolling(window).mean()
        values[f"close_location_{window}"] = ((close - bars["low"]) / span).rolling(window).mean()
        values[f"downside_risk_{window}"] = r.clip(upper=0).pow(2).rolling(window).mean().pow(0.5)
        values[f"positive_fraction_{window}"] = r.gt(0).astype(float).rolling(window).mean()
        centered = np.arange(window) - (window - 1) / 2
        values[f"regression_slope_{window}"] = roll.apply(
            lambda x, c=centered: np.dot(c, np.cumsum(x)) / np.dot(c, c), raw=True
        )
    for symbol in BENCHMARKS:
        market = factors[symbol]
        paired = r.where(market.notna())
        beta = paired.rolling(64, min_periods=16).cov(market) / market.rolling(
            64, min_periods=16
        ).var().replace(0, np.nan)
        alpha = (
            paired.rolling(64, min_periods=16).mean()
            - beta * market.rolling(64, min_periods=16).mean()
        )
        values[f"factor_beta_{symbol}"] = beta
        values[f"factor_alpha_{symbol}"] = alpha
        values[f"factor_residual_{symbol}"] = r - alpha - beta * market
        values[f"factor_correlation_{symbol}"] = paired.rolling(64, min_periods=16).corr(market)
    result = pd.DataFrame(values, index=bars.index)
    result = result.join(arima_features(r, bars["session"]))
    return result.replace([np.inf, -np.inf], np.nan)


def labeled_frame(bars: pd.DataFrame, x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    future = bars.shift(-1)
    valid = future["session"].eq(bars["session"])
    valid &= pd.Series(bars.index, index=bars.index).shift(-1).eq(bars["available_at"])
    net_points = future["close"] - future["open"] - 1.25
    frame = x.copy()
    frame["target"] = net_points.gt(0).astype(int)
    frame["available_at"] = bars["available_at"]
    frame["label_end"] = bars["available_at"] + pd.Timedelta(minutes=horizon)
    frame["trade_open"] = future["open"]
    frame["trade_close"] = future["close"]
    frame["session"] = bars["session"]
    return frame.loc[valid & frame["momentum_64"].notna()].copy()


def classifiers() -> dict[str, object]:
    return {
        "logistic": LogisticRegression(C=0.1, max_iter=1000),
        "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "naive_bayes": GaussianNB(),
        "rf": RandomForestClassifier(
            n_estimators=120, max_depth=6, min_samples_leaf=40, n_jobs=4, random_state=SEED
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=120, max_depth=6, min_samples_leaf=40, n_jobs=4, random_state=SEED
        ),
        "hist_boost": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, l2_regularization=10, random_state=SEED
        ),
        "adaboost": AdaBoostClassifier(n_estimators=80, learning_rate=0.05, random_state=SEED),
        "svm": LinearSVC(C=0.1, max_iter=5000, random_state=SEED),
        "knn": KNeighborsClassifier(n_neighbors=101, weights="uniform", n_jobs=4),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=10, max_iter=200, shuffle=False, random_state=SEED
        ),
    }


def block_importance(model: object, x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Unseen selection data, temporal blocks, fixed seeds; not training impurity."""
    base = log_loss(y, model.predict_proba(x), labels=[0, 1])
    rng = np.random.default_rng(SEED)
    blocks = np.array_split(np.arange(len(x)), max(2, len(x) // 32))
    importance = {}
    for col in x:
        changes = []
        for _ in range(3):
            order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            shuffled = x.copy()
            shuffled[col] = x[col].to_numpy()[order]
            changes.append(log_loss(y, model.predict_proba(shuffled), labels=[0, 1]) - base)
        importance[col] = float(np.mean(changes))
    return pd.Series(importance).sort_values(ascending=False)


def calibrated_probability(
    model: object, calibration: pd.DataFrame, y: pd.Series, test: pd.DataFrame
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = logit(np.clip(model.predict_proba(calibration)[:, 1], 1e-6, 1 - 1e-6))
        test_raw = logit(np.clip(model.predict_proba(test)[:, 1], 1e-6, 1 - 1e-6))
    else:
        raw = model.decision_function(calibration)
        test_raw = model.decision_function(test)
    platt = LogisticRegression(C=1, max_iter=1000)
    platt.fit(raw.reshape(-1, 1), y)
    return platt.predict_proba(test_raw.reshape(-1, 1))[:, 1]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    p, y = frame["probability"], frame["target"]
    entry = p.ge(0.66)
    pnl = (frame["trade_close"] - frame["trade_open"]) * 20 - 25
    traded = pnl.loc[entry]
    daily = traded.groupby(frame.loc[entry, "session"]).sum()
    daily = daily.reindex(pd.Index(frame["session"].unique()), fill_value=0).sort_index()
    cumulative = daily.cumsum()
    peak = cumulative.cummax().clip(lower=0)
    losers = -traded.loc[traded.lt(0)].sum()
    return {
        "observations": len(frame),
        "trades": int(entry.sum()),
        "accuracy": accuracy_score(y, p.ge(0.5)),
        "balanced_accuracy": balanced_accuracy_score(y, p.ge(0.5)),
        "always_up_accuracy": float(y.mean()),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "precision_at_066": float(y.loc[entry].mean()) if entry.any() else np.nan,
        "coverage": float(entry.mean()),
        "net_dollars_1_contract": float(traded.sum()),
        "double_cost_net_dollars": float((traded - 25).sum()),
        "always_long_net_dollars": float(pnl.sum()),
        "expectancy_dollars": float(traded.mean()) if entry.any() else np.nan,
        "profit_factor": float(traded.clip(lower=0).sum() / losers) if losers > 0 else np.nan,
        "max_drawdown_dollars": float((peak - cumulative).max()),
        "daily_sharpe": float(daily.mean() / daily.std() * np.sqrt(252))
        if daily.std() > 0
        else np.nan,
    }


def signal_backtest(frame: pd.DataFrame, filter_name: str, stress: int = 1) -> dict[str, object]:
    """One long at a time; causal exits at following opens, capped at four bars."""
    ordered = frame.sort_values("available_at")
    trades = []
    rows = list(ordered.to_dict("records"))
    i = 0
    while i < len(rows):
        row = rows[i]
        trend_ok = filter_name == "none" or float(row[filter_name]) > 0
        if row["probability"] < 0.66 or not trend_ok:
            i += 1
            continue
        entry = float(row["trade_open"])
        j, exit_price = i, float(row["trade_close"])
        while j + 1 < len(rows) and j - i + 1 < 4:
            following = rows[j + 1]
            if (
                following["session"] != row["session"]
                or following["available_at"] != rows[j]["label_end"]
            ):
                break
            keep_trend = filter_name == "none" or float(following[filter_name]) > 0
            if following["probability"] < 0.5 or not keep_trend:
                exit_price = float(following["trade_open"])
                # Exit at this open. Do not re-enter at the same open.
                j += 1
                break
            j += 1
            exit_price = float(following["trade_close"])
        trades.append({"session": row["session"], "pnl": (exit_price - entry) * 20 - 25 * stress})
        i = j + 1
    pnl = pd.DataFrame(trades, columns=["session", "pnl"])
    daily = (
        pnl.groupby("session")["pnl"]
        .sum()
        .reindex(pd.Index(ordered["session"].unique()), fill_value=0)
        .sort_index()
    )
    curve = daily.cumsum()
    rng = np.random.default_rng(SEED)
    # Session blocks preserve short-range serial dependence; exploratory interval.
    simulated = []
    if len(daily) >= 20:
        arr = daily.to_numpy()
        for _ in range(1000):
            starts = rng.integers(0, len(arr), size=int(np.ceil(len(arr) / 10)))
            sample = np.concatenate([arr[(s + np.arange(10)) % len(arr)] for s in starts])[
                : len(arr)
            ]
            simulated.append(sample.sum())
    losses = -pnl["pnl"].clip(upper=0).sum()
    return {
        "filter": filter_name,
        "cost_multiplier": stress,
        "trades": len(pnl),
        "net_dollars": float(pnl["pnl"].sum()),
        "win_rate": float(pnl["pnl"].gt(0).mean()) if len(pnl) else np.nan,
        "profit_factor": float(pnl["pnl"].clip(lower=0).sum() / losses) if losses else np.nan,
        "max_drawdown_dollars": float((curve.cummax().clip(lower=0) - curve).max()),
        "bootstrap_net_p05": float(np.quantile(simulated, 0.05)) if simulated else np.nan,
        "bootstrap_net_p95": float(np.quantile(simulated, 0.95)) if simulated else np.nan,
    }


def run(horizon: int, model_names: list[str], out: Path) -> None:
    print(f"{horizon}m: loading development bars", flush=True)
    bars = aggregate(load_minutes("NQ"), horizon)
    factors = pd.DataFrame(
        {
            symbol: np.log((b := aggregate(load_minutes(symbol), horizon))["close"] / b["open"])
            for symbol in BENCHMARKS
        }
    ).reindex(bars.index)
    raw = features(bars, factors)
    frame = labeled_frame(bars, raw, horizon)
    raw.notna().mean().rename("observed_fraction").to_csv(out / f"feature_coverage_{horizon}m.csv")
    factors.notna().mean().rename("observed_fraction").to_csv(
        out / f"factor_coverage_{horizon}m.csv"
    )
    feature_names = list(raw)
    all_predictions, rankings, fold_scores, clusters = [], [], [], []
    for fold, (start, end) in enumerate(
        zip(
            ["2024-08-22", "2024-11-22", "2025-02-22", "2025-05-22"],
            ["2024-11-22", "2025-02-22", "2025-05-22", "2025-08-16"],
            strict=True,
        ),
        start=1,
    ):
        start, end = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        training = frame.loc[frame["label_end"].lt(start)].copy()
        test = frame.loc[frame["available_at"].ge(start) & frame["label_end"].lt(end)].copy()
        a, b = int(len(training) * 0.6), int(len(training) * 0.8)
        selection_start, calibration_start = (
            training.iloc[a]["available_at"],
            training.iloc[b]["available_at"],
        )
        fit = training.iloc[:a]
        fit = fit.loc[fit["label_end"].lt(selection_start)]
        selection = training.iloc[a:b]
        selection = selection.loc[selection["label_end"].lt(calibration_start)]
        calibration = training.iloc[b:]
        if any(
            len(z) < 100 or z["target"].nunique() < 2 for z in (fit, selection, calibration, test)
        ):
            raise ValueError(f"{horizon}m fold {fold}: insufficient observations/classes")
        # Factor PCA fits only earlier training observations, never evaluation rows.
        factor_processor = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            PCA(n_components=3, random_state=SEED),
        )
        factor_processor.fit(factors.reindex(fit.index))
        matrices = []
        for part in (fit, selection, calibration, test):
            matrix = part[feature_names].copy()
            pcs = factor_processor.transform(factors.reindex(part.index))
            for i in range(3):
                matrix[f"pca_factor_{i + 1}"] = pcs[:, i]
            matrices.append(matrix)
        xfit, xselect, xcal, xtest = matrices
        unavailable = xfit.columns[xfit.notna().sum().eq(0)].tolist()
        if unavailable:
            raise ValueError(f"Unusable training features: {unavailable}; investigate coverage")
        screen = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), classifiers()["rf"]
        )
        screen.fit(xfit, fit["target"])
        importance = block_importance(screen, xselect, selection["target"])
        selected = list(importance.head(20).index)
        ranks = importance.rename_axis("feature").rename("log_loss_degradation").reset_index()
        ranks["fold"], ranks["selected"] = fold, ranks["feature"].isin(selected)
        rankings.append(ranks)
        corr = xfit.corr().abs()
        for i, col in enumerate(corr):
            for other in corr.columns[i + 1 :]:
                if corr.loc[col, other] >= 0.9:
                    clusters.append(
                        {
                            "fold": fold,
                            "feature": col,
                            "correlated_feature": other,
                            "abs_correlation": corr.loc[col, other],
                        }
                    )
        print(
            f"{horizon}m fold {fold}: {len(xfit)} fit, {len(test)} evaluation; "
            f"selected 20/{len(xfit.columns)}",
            flush=True,
        )
        cases = [(name, selected) for name in model_names] + [("rf_full", list(xfit))]
        for name, cols in cases:
            print(f"{horizon}m fold {fold}: fitting {name}", flush=True)
            estimator = classifiers()["rf" if name == "rf_full" else name]
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(), clone(estimator)
            )
            model.fit(pd.concat([xfit, xselect])[cols], pd.concat([fit, selection])["target"])
            prob = calibrated_probability(model, xcal[cols], calibration["target"], xtest[cols])
            prediction = test[
                ["target", "trade_open", "trade_close", "session", "available_at", "label_end"]
            ].copy()
            prediction["probability"], prediction["model"], prediction["fold"] = prob, name, fold
            prediction["ema_distance_4"] = test["ema_distance_4"]
            prediction["regression_slope_8"] = test["regression_slope_8"]
            all_predictions.append(prediction)
            fold_scores.append(
                {"horizon_minutes": horizon, "fold": fold, "model": name, **metrics(prediction)}
            )
        pd.DataFrame(fold_scores).to_csv(out / f"fold_scores_{horizon}m.csv", index=False)
    predictions = pd.concat(all_predictions)
    predictions.to_parquet(out / f"predictions_{horizon}m.parquet")
    pd.concat(rankings).to_csv(out / f"feature_rankings_{horizon}m.csv", index=False)
    stability = (
        pd.concat(rankings)
        .groupby("feature")
        .agg(mean_importance=("log_loss_degradation", "mean"), selected_folds=("selected", "sum"))
        .sort_values(["selected_folds", "mean_importance"], ascending=False)
    )
    stability.to_csv(out / f"feature_stability_{horizon}m.csv")
    pd.DataFrame(clusters).to_csv(out / f"correlated_features_{horizon}m.csv", index=False)
    pd.DataFrame(fold_scores).to_csv(out / f"fold_scores_{horizon}m.csv", index=False)
    pd.DataFrame(
        [
            {"horizon_minutes": horizon, "model": name, **metrics(group)}
            for name, group in predictions.groupby("model")
        ]
    ).to_csv(out / f"summary_{horizon}m.csv", index=False)
    pd.DataFrame(
        [
            {
                "horizon_minutes": horizon,
                "model": name,
                **signal_backtest(group, filter_name, stress),
            }
            for name, group in predictions.groupby("model")
            for filter_name in ("none", "ema_distance_4", "regression_slope_8")
            for stress in (1, 2)
        ]
    ).to_csv(out / f"signal_exits_{horizon}m.csv", index=False)
    raw.iloc[:0].to_csv(out / f"feature_dictionary_{horizon}m.csv")
    print(f"{horizon}m complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=["rf", "all"], default="rf")
    parser.add_argument(
        "--horizons", nargs="+", type=int, choices=[15, 60, 240], default=[15, 60, 240]
    )
    args = parser.parse_args()
    out = Path(__file__).parent / "runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running",
        "development_only": True,
        "fresh_holdout_available": False,
        "start": str(START),
        "end_exclusive": str(END),
        "horizons": args.horizons,
        "models": list(classifiers()) if args.models == "all" else ["rf"],
        "seed": SEED,
        "raw_features": 90,
        "fold_pca_features": 3,
        "entry_threshold": 0.66,
        "roundtrip_dollars_per_contract": 25,
        "implemented_exit": "one_bar_and_probability_trend_exit_max_four_bars",
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        for horizon in args.horizons:
            run(horizon, manifest["models"], out)
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", repr(error)
        raise
    else:
        manifest["status"] = "complete_development_fixed_exit"
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Output: {out}", flush=True)


if __name__ == "__main__":
    main()
