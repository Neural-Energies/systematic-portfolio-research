"""Broader next-bar direction gauntlet: causal intermarket inputs and calibrated probabilities."""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from experiment import END, ROOT, SEED, START, aggregate, load_minutes
from medium_frequency import FIRST_TEST, raw_scores, session_keys
from range_forecast import carry_completed, compact_features, daily_bars, mixed_features
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning
)
EXTENDED = ("RTY", "ZB", "ZF", "ZT", "SR3", "ZW", "ZS", "LE")
ORIGINAL = ("ES", "NKD", "ZN", "GC", "HG", "CL", "NG", "6E", "6J", "ZC")


def load_extended(symbol: str) -> pd.DataFrame:
    """Select each UTC day's contract using the previous completed day's volume leader."""
    path = (
        ROOT
        / "data/processed/databento_portfolio_extension/minute_bars"
        / f"symbol={symbol}/bars.parquet"
    )
    # A file handle avoids Arrow inferring the root's hive symbol over raw contract symbols.
    with path.open("rb") as stream:
        frame = pd.read_parquet(
            stream,
            columns=["ts_event", "symbol", "open", "high", "low", "close", "volume"],
            filters=[("ts_event", ">=", START - pd.Timedelta(days=7)), ("ts_event", "<", END)],
        )
    frame["timestamp"] = pd.to_datetime(frame["ts_event"], utc=True)
    frame["utc_day"] = frame["timestamp"].dt.normalize()
    volumes = frame.groupby(["utc_day", "symbol"], observed=True)["volume"].sum().reset_index()
    leaders = volumes.sort_values(
        ["utc_day", "volume", "symbol"], ascending=[True, False, True]
    ).drop_duplicates("utc_day")
    leaders["selected_contract"] = leaders["symbol"].shift(1)
    frame = frame.merge(
        leaders[["utc_day", "selected_contract"]], on="utc_day", validate="many_to_one"
    )
    frame = frame.loc[frame["symbol"].eq(frame["selected_contract"]) & frame.timestamp.ge(START)]
    frame = frame.sort_values("timestamp").set_index("timestamp")
    if frame.index.duplicated().any():
        raise ValueError(f"Duplicate minute after causal contract selection: {symbol}")
    frame["_contract"] = frame["symbol"]
    return frame[["open", "high", "low", "close", "volume", "_contract"]]


def observed_predictor_bars(market: pd.DataFrame, duration: int) -> pd.DataFrame:
    """Completed predictor periods can use sparse observations with age/coverage disclosed."""
    dates = session_keys(market.index)
    anchors = (dates + pd.Timedelta(hours=17)).tz_localize("America/Chicago").tz_convert("UTC")
    elapsed = (market.index - anchors).total_seconds() / 60
    keys = anchors + pd.to_timedelta(np.floor(elapsed / duration) * duration, unit="min")
    frame = market.assign(key=keys, observed_timestamp=market.index)
    bars = frame.groupby("key").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        count=("close", "size"),
        last_quote=("observed_timestamp", "max"),
    )
    bars["available_at"] = bars.index + pd.Timedelta(minutes=duration)
    bars["quote_age_minutes"] = (bars.available_at - bars.last_quote).dt.total_seconds() / 60 - 1
    bars["coverage_fraction"] = bars["count"] / duration
    return bars.loc[bars["count"].ge(2) & bars.quote_age_minutes.le(min(10, duration / 4))]


def contract_safe_bars(
    market: pd.DataFrame, duration: int, sparse_predictor: bool = False
) -> pd.DataFrame:
    bars = (
        daily_bars(market)
        if duration == 0
        else observed_predictor_bars(market, duration)
        if sparse_predictor
        else aggregate(market, duration)
    )
    if "_contract" not in market:
        return bars
    changes = market.index[market["_contract"].ne(market["_contract"].shift())]
    opens = (
        (bars.index + pd.Timedelta(hours=17)).tz_localize("America/Chicago").tz_convert("UTC")
        if duration == 0
        else bars.index
    )
    counts = changes.searchsorted(
        pd.DatetimeIndex(bars.available_at), side="left"
    ) - changes.searchsorted(opens, side="right")
    return bars.loc[counts == 0]


def make_features():
    minutes, x, _, _ = mixed_features()
    x = x.copy()
    clock = x.index
    nq_bars = aggregate(minutes, 60)
    nq_ret = pd.Series(
        np.log(nq_bars.close / nq_bars.open).to_numpy(),
        index=pd.DatetimeIndex(nq_bars.available_at),
    ).reindex(clock)
    market_returns, coverage = {}, []
    for symbol in ORIGINAL + EXTENDED:
        print(f"Preparing intermarket features: {symbol}", flush=True)
        market = load_extended(symbol) if symbol in EXTENDED else load_minutes(symbol)
        bars = contract_safe_bars(market, 60, sparse_predictor=True)
        compact = compact_features(bars, f"market_{symbol}")
        compact[f"market_{symbol}_coverage"] = bars.coverage_fraction.to_numpy()
        compact[f"market_{symbol}_quote_age"] = bars.quote_age_minutes.to_numpy()
        # Contemporaneous features are completed at or before the forecast, never future returns.
        compact[f"market_{symbol}_lag1"] = compact[f"market_{symbol}_return"].shift(1)
        compact[f"market_{symbol}_lag2"] = compact[f"market_{symbol}_return"].shift(2)
        hourly = carry_completed(compact, clock, 2)
        x = x.join(hourly)
        completed = compact[f"market_{symbol}_return"].reindex(clock)
        market_returns[symbol] = completed
        paired = nq_ret.where(completed.notna())
        beta = paired.rolling(128, min_periods=24).cov(completed) / completed.rolling(
            128, min_periods=24
        ).var().replace(0, np.nan)
        x[f"market_{symbol}_beta128"] = beta
        x[f"market_{symbol}_correlation128"] = paired.rolling(128, min_periods=24).corr(completed)
        x[f"market_{symbol}_residual"] = nq_ret - beta * completed
        x[f"market_{symbol}_relative4"] = x.momentum_4 - hourly[f"market_{symbol}_momentum4"]
        x[f"market_{symbol}_available"] = completed.notna().astype(float)
        for duration, prefix, age in ((15, "15m", 1), (240, "4h", 8), (0, "daily", 96)):
            b = contract_safe_bars(market, duration, sparse_predictor=True)
            cf = compact_features(b, f"market_{symbol}_{prefix}")
            x = x.join(carry_completed(cf[[f"market_{symbol}_{prefix}_return"]], clock, age))
        coverage.append(
            {
                "market": symbol,
                "rows": len(market),
                "start": str(market.index.min()),
                "end": str(market.index.max()),
                "matched_hour_fraction": completed.notna().mean(),
            }
        )
    local = clock.tz_convert("America/New_York")
    x["hour_sin"], x["hour_cos"] = (
        np.sin(2 * np.pi * local.hour / 24),
        np.cos(2 * np.pi * local.hour / 24),
    )
    x["weekday"] = local.dayofweek
    x["regular_session"] = ((local.hour >= 9) & (local.hour < 16)).astype(float)
    # Minute price/volume proxies, not claims of bid/ask aggressor order-flow data.
    signed = np.sign(np.log(minutes.close / minutes.open))
    session = session_keys(minutes.index)
    dollar = (minutes.close * minutes.volume).groupby(session).cumsum()
    volume = minutes.volume.groupby(session).cumsum().replace(0, np.nan)
    minute_features = pd.DataFrame(
        {
            "vwap_distance": minutes.close / (dollar / volume) - 1,
            "signed_volume_proxy": signed * minutes.volume,
            "squared_minute_return": np.log(minutes.close / minutes.open).pow(2),
            "volume": minutes.volume,
        },
        index=minutes.index,
    )
    hourly_micro = minute_features.resample("h").agg(
        {
            "vwap_distance": "last",
            "signed_volume_proxy": "sum",
            "squared_minute_return": "sum",
            "volume": "sum",
        }
    )
    hourly_micro["signed_volume_proxy"] /= hourly_micro.pop("volume").replace(0, np.nan)
    hourly_micro.index += pd.Timedelta(hours=1)
    x = x.join(carry_completed(hourly_micro, clock, 1))
    return minutes, x.replace([np.inf, -np.inf], np.nan), pd.DataFrame(market_returns), coverage


def literal_bar_labels(minutes: pd.DataFrame, x: pd.DataFrame, horizon: int):
    bars = aggregate(minutes, horizon)
    data = x.reindex(bars.index).copy()
    data["target"] = (bars.close > bars.open).astype(int)
    data["label_end"] = bars.available_at
    data["session"] = session_keys(bars.index)
    data["bar_open"], data["bar_close"] = bars.open, bars.close
    # At least one minute of latency, same exact bar end; P(up) and P(net profit) differ.
    delayed = minutes.open.reindex(bars.index + pd.Timedelta(minutes=1)).to_numpy()
    data["delayed_entry"] = delayed
    data["net_dollars"] = (bars.close.to_numpy() - delayed) * 20 - 25
    return data.loc[data.momentum_64.notna() & data.delayed_entry.notna()]


def catalog():
    return {
        "elastic_logit": LogisticRegression(
            C=0.15,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.2,
            max_iter=1500,
            random_state=SEED,
        ),
        "shrinkage_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "gaussian_nb": GaussianNB(var_smoothing=0.01),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=10,
            min_samples_leaf=15,
            max_features=0.7,
            n_jobs=4,
            random_state=SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=15,
            max_features=0.8,
            n_jobs=4,
            random_state=SEED,
        ),
        "hist_boost": HistGradientBoostingClassifier(
            max_iter=160,
            max_leaf_nodes=15,
            learning_rate=0.05,
            min_samples_leaf=30,
            l2_regularization=10,
            early_stopping=False,
            random_state=SEED,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=25,
            random_state=SEED,
        ),
        "rbf_svm": SVC(C=0.5, gamma="scale", cache_size=512, random_state=SEED),
        "nearest_neighbors": KNeighborsClassifier(n_neighbors=151, weights="distance", n_jobs=4),
        "neural_mlp": MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=20, max_iter=250, shuffle=False, random_state=SEED
        ),
    }


class ProbabilityCalibration:
    def __init__(self, method: str):
        self.method = method

    def fit(self, score: np.ndarray, y: pd.Series):
        if self.method == "sigmoid":
            # Strong C avoids forcing already narrow scores toward 0.5 via the old C=1 fit.
            self.model = LogisticRegression(C=100, max_iter=1500).fit(score.reshape(-1, 1), y)
        else:
            # Target 75 examples per bin; the two-bin minimum can be smaller for 4h.
            order = np.argsort(score, kind="stable")
            groups = np.array_split(order, max(2, min(10, len(score) // 75)))
            centers = np.array([score[g].mean() for g in groups])
            rates = np.array([(y.iloc[g].sum() + 0.5) / (len(g) + 1) for g in groups])
            counts = np.array([len(g) for g in groups])
            self.model = IsotonicRegression(out_of_bounds="clip").fit(
                centers, rates, sample_weight=counts
            )
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        p = (
            self.model.predict_proba(score.reshape(-1, 1))[:, 1]
            if self.method == "sigmoid"
            else self.model.predict(score)
        )
        return np.clip(p, 1e-6, 1 - 1e-6)


def metrics(data: pd.DataFrame, p: np.ndarray):
    y = data.target.to_numpy()
    long = p >= 0.66
    return {
        "observations": len(y),
        "accuracy": accuracy_score(y, p >= 0.5),
        "balanced_accuracy": balanced_accuracy_score(y, p >= 0.5),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "majority_baseline": max(y.mean(), 1 - y.mean()),
        "signals_066": int(long.sum()),
        "signal_coverage": long.mean(),
        "signal_accuracy": float(y[long].mean()) if long.any() else np.nan,
        "mean_signal_probability": float(p[long].mean()) if long.any() else np.nan,
        "signal_net_dollars": float(data.net_dollars.to_numpy()[long].sum()),
    }


def feature_matrices(parts, x, markets):
    usable = [c for c in x if x.reindex(parts[0].index)[c].notna().sum() >= 60]
    observed = [c for c in markets if markets.reindex(parts[0].index)[c].notna().sum() >= 60]
    pc = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        PCA(n_components=min(5, len(observed)), random_state=SEED),
    )
    pc.fit(markets.reindex(parts[0].index)[observed])
    matrices = []
    for part in parts:
        matrix = x.reindex(part.index)[usable].copy()
        for k, values in enumerate(pc.transform(markets.reindex(part.index)[observed]).T, 1):
            matrix[f"market_pca_{k}"] = values
        matrices.append(matrix)
    return matrices, pc, observed


def screen_features(xf, yf, xs, ys):
    screen = make_pipeline(
        # A newly introduced market can be absent in the earliest screening slice.
        # Preserve the schema rather than silently dropping its all-missing columns.
        SimpleImputer(strategy="median", keep_empty_features=True),
        ExtraTreesClassifier(
            n_estimators=100, max_depth=8, min_samples_leaf=20, n_jobs=4, random_state=SEED
        ),
    )
    screen.fit(xf, yf)
    provisional = pd.Series(screen[-1].feature_importances_, index=xf.columns).nlargest(100).index
    base = log_loss(ys, screen.predict_proba(xs), labels=[0, 1])
    blocks = np.array_split(np.arange(len(xs)), max(2, len(xs) // 32))
    rng, rank = np.random.default_rng(SEED), {}
    for col in provisional:
        shuffled = xs.copy()
        order = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
        shuffled[col] = xs[col].to_numpy()[order]
        rank[col] = log_loss(ys, screen.predict_proba(shuffled), labels=[0, 1]) - base
    return pd.Series(rank).sort_values(ascending=False)


def train_horizon(horizon, minutes, x, markets, out):
    data = literal_bar_labels(minutes, x, horizon)
    boundaries = list(pd.date_range(FIRST_TEST, periods=4, freq=pd.DateOffset(months=3))) + [END]
    forecasts, diagnostics, ranks, chosen_rows = [], [], [], []
    for fold, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True), 1):
        test = data.loc[(data.index >= start) & (data.index < end)]
        for window in ("expanding", "270d"):
            train = data.loc[(data.index < start) & data.label_end.lt(start)]
            if window == "270d":
                train = train.loc[train.index >= start - pd.Timedelta(days=270)]
            a, b, c = [int(len(train) * f) for f in (0.55, 0.70, 0.85)]
            parts = [
                train.iloc[lo:hi].copy() for lo, hi in ((0, a), (a, b), (b, c), (c, len(train)))
            ]
            for k in range(3):
                parts[k] = parts[k].loc[parts[k].label_end.lt(parts[k + 1].index.min())]
            if any(len(p) < 80 or p.target.nunique() != 2 for p in parts):
                raise ValueError("Inadequate training or class coverage")
            matrices, pca, observed = feature_matrices(parts + [test], x, markets)
            xf, xs, xc, xu, xt = matrices
            ranking = screen_features(xf, parts[0].target, xs, parts[1].target)
            ranks.extend(
                {"horizon": horizon, "fold": fold, "window": window, "feature": f, "importance": v}
                for f, v in ranking.items()
            )
            for count in (20, 100):
                selected = list(ranking.head(count).index)
                for name, estimator in catalog().items():
                    print(f"{horizon}m fold {fold} {window} top{count}: {name}", flush=True)
                    model = make_pipeline(
                        SimpleImputer(strategy="median"), StandardScaler(), clone(estimator)
                    )
                    model.fit(pd.concat([xf, xs])[selected], pd.concat(parts[:2]).target)
                    sc, su, st = [raw_scores(model, m[selected]) for m in (xc, xu, xt)]
                    for method in ("sigmoid", "binned_isotonic"):
                        cal = ProbabilityCalibration(method).fit(sc, parts[2].target)
                        ptune, ptest = cal.predict(su), cal.predict(st)
                        config = f"{window}_{count}_{name}_{method}"
                        record = {
                            "horizon": horizon,
                            "fold": fold,
                            "config": config,
                            "model": name,
                            "window": window,
                            "features": len(selected),
                            "calibration": method,
                            "tune_brier": brier_score_loss(parts[3].target, ptune),
                            "tune_accuracy": accuracy_score(parts[3].target, ptune >= 0.5),
                            "tune_signals": int((ptune >= 0.66).sum()),
                            "tune_signal_accuracy": float(
                                parts[3].target.to_numpy()[ptune >= 0.66].mean()
                            )
                            if (ptune >= 0.66).any()
                            else np.nan,
                            **metrics(test, ptest),
                            "selected_features": selected,
                            "observed_markets": observed,
                        }
                        diagnostics.append(record)
                        forecast = test.drop(columns=list(x), errors="ignore").copy()
                        forecast["probability"], forecast["config"], forecast["fold"] = (
                            ptest,
                            config,
                            fold,
                        )
                        forecasts.append(forecast)
                        joblib.dump(
                            {
                                "model": model,
                                "calibration": cal,
                                "pca": pca,
                                "pca_markets": observed,
                                "selected_features": selected,
                                "trained_before": str(start),
                                "target": "next_bar_close_above_open",
                                "live_ready": False,
                            },
                            out / f"model_{horizon}_{fold}_{config}.joblib",
                            compress=1,
                        )
            pd.DataFrame(diagnostics).to_csv(out / f"diagnostics_{horizon}.csv", index=False)
        fold_rows = [r for r in diagnostics if r["fold"] == fold]
        best_probability = min(fold_rows, key=lambda r: r["tune_brier"])
        best_direction = max(fold_rows, key=lambda r: (r["tune_accuracy"], -r["tune_brier"]))
        qualified = [
            r for r in fold_rows if r["tune_signals"] >= 40 and r["tune_signal_accuracy"] >= 0.60
        ]
        best_signal = (
            max(qualified, key=lambda r: (r["tune_signal_accuracy"], r["tune_signals"]))
            if qualified
            else best_probability
        )
        for policy, best in (
            ("probability", best_probability),
            ("direction_accuracy", best_direction),
            ("selective_066", best_signal),
        ):
            prediction = next(
                f
                for f in forecasts
                if f.fold.iloc[0] == fold and f.config.iloc[0] == best["config"]
            ).copy()
            prediction["selection_policy"] = policy
            prediction["signal_tuning_qualified"] = (
                bool(qualified) if policy == "selective_066" else False
            )
            chosen_rows.append(prediction)
        pd.concat(forecasts).to_parquet(out / f"forecasts_{horizon}.parquet")
        pd.concat(chosen_rows).to_parquet(out / f"chosen_{horizon}.parquet")
    pd.DataFrame(ranks).to_csv(out / f"feature_ranks_{horizon}.csv", index=False)
    summary = []
    chosen = pd.concat(chosen_rows)
    for policy, frame in chosen.groupby("selection_policy"):
        summary.append(
            {"horizon": horizon, "policy": policy, **metrics(frame, frame.probability.to_numpy())}
        )
    # Every candidate is preserved; observed maxima are exploratory, not independent selections.
    all_forecasts = pd.concat(forecasts)
    for config, frame in all_forecasts.groupby("config"):
        summary.append(
            {
                "horizon": horizon,
                "policy": "candidate",
                "config": config,
                **metrics(frame, frame.probability.to_numpy()),
            }
        )
    pd.DataFrame(summary).to_csv(out / f"summary_{horizon}.csv", index=False)
    return summary


def main():
    out = Path(__file__).parent / "direction_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    (out / "source_snapshot.py").write_text(Path(__file__).read_text(), encoding="utf-8")
    manifest = {
        "status": "running",
        "target": "next_complete_bar_close_above_open",
        "horizons_minutes": [60, 240],
        "threshold": 0.66,
        "models": list(catalog()),
        "training_windows": ["expanding", "270d"],
        "features": [20, 100],
        "calibration": ["sigmoid", "binned_isotonic"],
        "fresh_holdout_available": False,
        "markets": list(ORIGINAL + EXTENDED),
        "extension_roll": "prior_completed_UTC_day_volume_leader",
        "source_NQ_roll_map_verified": False,
        "evaluation": "quarterly_development_not_fresh_lockout",
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        with threadpool_limits(limits=4):
            minutes, x, markets, coverage = make_features()
            pd.DataFrame(coverage).to_csv(out / "market_coverage.csv", index=False)
            x.to_parquet(out / "causal_features.parquet")
            markets.to_parquet(out / "market_returns.parquet")
            manifest["raw_feature_candidates"] = x.shape[1]
            summaries = []
            for horizon in (60, 240):
                summaries.extend(train_horizon(horizon, minutes, x, markets, out))
                pd.DataFrame(summaries).to_csv(out / "combined_summary.csv", index=False)
            manifest["status"] = "complete_development_only"
    except Exception as error:
        manifest["status"], manifest["error"] = "failed", repr(error)
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(out, flush=True)


if __name__ == "__main__":
    main()
