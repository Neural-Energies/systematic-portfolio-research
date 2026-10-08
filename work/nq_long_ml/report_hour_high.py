"""Fresh replay, saved-model verification and honest next-hour high error report."""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd
from experiment import load_minutes
from fifteen_probability import BASE, prepare
from hour_high import feature_table, high_labels, regression_metrics
from threadpoolctl import threadpool_limits

matplotlib.use("Agg")


def plot_results(chosen: pd.DataFrame, monthly: pd.DataFrame, out):
    """Point-error distribution and monthly tolerance rate, without price-level inflation."""
    import matplotlib.pyplot as plt

    error = (chosen.actual_high - chosen.predicted_high).abs().sort_values().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    axes[0].plot(error, np.arange(1, len(error) + 1) / len(error) * 100, color="#126e82")
    axes[0].axvline(20, color="#c45c26", linestyle="--", label="20-point goal")
    axes[0].set(
        xlabel="Absolute high forecast error (NQ points)",
        ylabel="Forecasts within error (%)",
        title="How often are forecasts close?",
        xlim=(0, 150),
        ylim=(0, 100),
    )
    axes[0].legend()
    axes[1].bar(monthly.month, monthly.within_20_points * 100, color="#126e82")
    axes[1].set(
        ylabel="Forecasts within ±20 points (%)", title="Consistency by month", ylim=(0, 100)
    )
    axes[1].tick_params(axis="x", rotation=60)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["right", "top"]].set_visible(False)
    fig.savefig(out / "accuracy.png", dpi=160)
    plt.close(fig)


def metrics(frame: pd.DataFrame) -> dict:
    return regression_metrics(
        frame.actual_high.to_numpy(), frame.predicted_high.to_numpy(), frame.reference.to_numpy()
    )


def block_intervals(frame: pd.DataFrame) -> dict:
    error = (frame.actual_high - frame.predicted_high).abs()
    daily = (
        pd.DataFrame({"absolute": error, "hit": error.le(20), "count": 1, "session": frame.session})
        .groupby("session")
        .sum()
    )
    rng = np.random.default_rng(20261005)
    n = len(daily)
    starts = rng.integers(0, n, (2000, int(np.ceil(n / 10))))
    indices = ((starts[:, :, None] + np.arange(10)) % n).reshape(2000, -1)[:, :n]
    sums = daily.to_numpy()[indices].sum(axis=1)
    mae = sums[:, 0] / sums[:, 2]
    hit = sums[:, 1] / sums[:, 2]
    return {
        "mae_95_low": float(np.quantile(mae, 0.025)),
        "mae_95_high": float(np.quantile(mae, 0.975)),
        "within20_95_low": float(np.quantile(hit, 0.025)),
        "within20_95_high": float(np.quantile(hit, 0.975)),
    }


def paired_skill(frame: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    baseline = baseline.reindex(frame.index)
    model_error = (frame.predicted_high - frame.actual_high).abs()
    baseline_error = (baseline.predicted_high - baseline.actual_high).abs()
    daily = (
        pd.DataFrame(
            {
                "mae_reduction": baseline_error - model_error,
                "hit_gain": model_error.le(20).astype(int) - baseline_error.le(20).astype(int),
                "count": 1,
                "session": frame.session,
            }
        )
        .groupby("session")
        .sum()
    )
    rng = np.random.default_rng(20261005)
    n = len(daily)
    starts = rng.integers(0, n, (2000, int(np.ceil(n / 10))))
    indices = ((starts[:, :, None] + np.arange(10)) % n).reshape(2000, -1)[:, :n]
    summed = daily.to_numpy()[indices].sum(axis=1)
    rates = summed[:, :2] / summed[:, 2:3]
    return {
        "mae_reduction_95_low": float(np.quantile(rates[:, 0], 0.025)),
        "mae_reduction_95_high": float(np.quantile(rates[:, 0], 0.975)),
        "within20_gain_95_low": float(np.quantile(rates[:, 1], 0.025)),
        "within20_gain_95_high": float(np.quantile(rates[:, 1], 0.975)),
    }


def run():
    completed = [
        path.parent
        for path in (BASE / "hour_high_runs").glob("*/summary.csv")
        if pd.read_csv(path.parent / "tuning.csv").fold.nunique() == 12
    ]
    if not completed:
        raise ValueError("A complete twelve-month run is required for the final report")
    out = sorted(completed)[-1]
    _, previous, _, _ = prepare()
    minute = load_minutes("NQ")
    x = feature_table(minute, previous)
    labels = high_labels(minute, x)
    saved_labels = pd.read_parquet(out / "labels.parquet")
    pd.testing.assert_frame_equal(labels, saved_labels)
    forecasts = pd.read_parquet(out / "forecasts.parquet")
    selected = pd.read_parquet(out / "selected.parquet")
    quantiles = pd.read_parquet(out / "quantiles.parquet")
    for name, frame in (("models", forecasts), ("selected", selected), ("quantiles", quantiles)):
        observed = labels.reindex(frame.index)
        if not np.allclose(observed.actual_high, frame.actual_high):
            raise AssertionError(f"{name} actual-high alignment failed")
        if not np.allclose(observed.reference, frame.reference):
            raise AssertionError(f"{name} future prices entered reference")
        if not (observed.known_at <= frame.index).all():
            raise AssertionError("Unavailable reference price")
        if not ((frame.label_end - frame.index) == pd.Timedelta(hours=1)).all():
            raise AssertionError("Wrong forecast duration")
    tuning = pd.read_csv(out / "tuning.csv")
    point_replays, quantile_replays = [], []
    for fold, chosen in selected.groupby("fold"):
        bundle = joblib.load(out / f"model_month_{fold:02d}.joblib")
        qchoices = tuning.loc[tuning.fold.eq(fold) & tuning.tune_pinball.notna()]
        expected_qgroup = qchoices.loc[qchoices.tune_pinball.idxmin(), "config"].removesuffix(
            "_quantiles"
        )
        if bundle["quantile_group"] != expected_qgroup:
            raise AssertionError("Quantile group selection differs from earlier tuning")
        candidates = tuning.loc[tuning.fold.eq(fold) & tuning.tune_mae.notna()]
        expected_mae = candidates.loc[candidates.tune_mae.idxmin(), "config"]
        expected_hit = (
            candidates.sort_values(["tune_within20", "tune_mae"], ascending=[False, True])
            .iloc[0]
            .config
        )
        for policy, expected in (
            ("earlier_selected", expected_mae),
            ("earlier_selected_within20", expected_hit),
        ):
            frame = chosen.loc[chosen.policy.eq(policy)]
            if not frame.config.eq(expected).all():
                raise AssertionError("Earlier model selection differs from protocol")
            candidate = forecasts.loc[forecasts.fold.eq(fold) & forecasts.config.eq(expected)]
            if not candidate.index.equals(frame.index) or not np.allclose(
                candidate.predicted_high, frame.predicted_high
            ):
                raise AssertionError("Selected forecasts differ from candidate")
        frame = chosen.loc[chosen.policy.eq("earlier_selected")]
        rebuilt = (
            frame.reference.to_numpy()
            + bundle["model"].predict(x.loc[frame.index, bundle["columns"]])
            * frame.scale.to_numpy()
        )
        if not np.allclose(rebuilt, frame.predicted_high):
            raise AssertionError("Saved point model failed prediction replay")
        point_replays.append(fold)
        qframe = quantiles.loc[quantiles.fold.eq(fold) & quantiles.config.eq("earlier_selected")]
        predictions = np.column_stack(
            [
                model.predict(x.loc[qframe.index, bundle["quantile_columns"]])
                for model in bundle["quantile_models"]
            ]
        )
        predictions = np.sort(predictions + bundle["quantile_shifts"], axis=1)
        for i, column in enumerate(("q10", "q50", "q80", "q90")):
            rebuilt = qframe.reference.to_numpy() + predictions[:, i] * qframe.scale.to_numpy()
            if not np.allclose(rebuilt, qframe[column]):
                raise AssertionError("Saved quantile model failed prediction replay")
        quantile_replays.append(fold)
    for field in ("fit_end", "cal_end", "tune_end"):
        if not (
            pd.to_datetime(tuning[field], utc=True) < pd.to_datetime(tuning.test_start, utc=True)
        ).all():
            raise AssertionError("Training used test-period outcomes")
    summary = pd.read_csv(out / "summary.csv").set_index("config")
    for config, group in forecasts.groupby("config"):
        recomputed = metrics(group)
        for field, value in recomputed.items():
            if not np.isclose(summary.loc[config, field], value):
                raise AssertionError("Summary metric replay failed")
    # With a one-sided high and symmetric ±20 tolerance, +20 is a necessary strong control.
    # It uses no fitted parameters and must not be confused with predictive ML skill.
    offset_baseline = forecasts.loc[forecasts.config.eq("last_price")].copy()
    offset_baseline["predicted_high"] = offset_baseline.reference + 20
    offset_baseline["config"] = "last_price_plus20"
    summary.loc["last_price_plus20"] = metrics(offset_baseline)
    offset_baseline.to_parquet(out / "baseline_plus20.parquet")
    summary.to_csv(out / "summary_with_baselines.csv")
    months, regimes = [], []
    for policy, frame in selected.groupby("policy"):
        for month, group in frame.groupby(frame.index.strftime("%Y-%m")):
            months.append({"policy": policy, "month": month, **metrics(group)})
        # Regimes are formed from known range scale, not future realized highs.
        bucket = pd.qcut(frame.scale, 3, labels=["quiet", "medium", "volatile"])
        for regime, group in frame.groupby(bucket, observed=True):
            regimes.append({"policy": policy, "regime": str(regime), **metrics(group)})
        rth = frame.index.tz_convert("America/New_York")
        slot = rth.hour * 60 + rth.minute
        regular = (slot >= 570) & (slot + 60 <= 960)
        opening = (slot < 570) & (slot + 60 > 570)
        for regime, mask in (
            ("RTH_full_hour", regular),
            ("opening_overlap", opening),
            ("outside_RTH", ~(regular | opening)),
        ):
            regimes.append({"policy": policy, "regime": regime, **metrics(frame.loc[mask])})
    monthly = pd.DataFrame(months)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    pd.DataFrame(regimes).to_csv(out / "regime_metrics.csv", index=False)
    intervals = {policy: block_intervals(group) for policy, group in selected.groupby("policy")}
    (out / "block_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    qmetrics = []
    for config, group in quantiles.groupby("config"):
        for column, level in (("q10", 0.1), ("q50", 0.5), ("q80", 0.8), ("q90", 0.9)):
            qmetrics.append(
                {
                    "config": config,
                    "quantile": level,
                    "actual_below_prediction": float(group.actual_high.le(group[column]).mean()),
                    "mae_points": float((group.actual_high - group[column]).abs().mean()),
                }
            )
    pd.DataFrame(qmetrics).to_csv(out / "quantile_coverage.csv", index=False)
    audit = {
        "point_model_months_replayed": point_replays,
        "quantile_months_replayed": quantile_replays,
        "forecast_label_replay": True,
        "prior_only_selection_replay": True,
        "metric_replay": True,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    p = summary.loc["earlier_selected_within20"]
    baseline = summary.loc["last_price_plus20"]
    chosen_months = monthly.loc[monthly.policy.eq("earlier_selected_within20")]
    chosen = selected.loc[selected.policy.eq("earlier_selected_within20")]
    skill = paired_skill(chosen, offset_baseline)
    (out / "paired_baseline_skill.json").write_text(json.dumps(skill, indent=2), encoding="utf-8")
    errors = chosen.copy()
    errors["forecast_time_new_york"] = errors.index.tz_convert("America/New_York").astype(str)
    errors["error_points"] = errors.predicted_high - errors.actual_high
    errors["absolute_error_points"] = errors.error_points.abs()
    errors["within20"] = errors.absolute_error_points.le(20)
    errors.to_csv(out / "forecast_errors.csv", index_label="forecast_time_utc")
    plot_results(chosen, chosen_months, out)
    worst = chosen_months.sort_values("within_20_points").iloc[0]
    best = (
        summary.loc[~summary.index.isin(["earlier_selected", "earlier_selected_within20"])]
        .sort_values("within_20_points", ascending=False)
        .iloc[0]
    )
    rows = [
        "# NQ next-hour high: regression results",
        "",
        "Forecast the maximum price over the next complete 60 minutes, once per hour.",
        "The goal is absolute forecast error <=20 NQ points. No targets, stops or trades.",
        "",
        f"Evaluated {int(p.observations):,} later hourly forecasts with monthly retraining.",
        "Six point-model families, three feature groups and four quantiles were compared.",
        "Feature groups: NQ only, NQ+ES, and all markets with fit-only redundancy clustering.",
        "Point models: ridge, elastic net, random forest, extra trees, boosted mean and median.",
        "Models learn the high's displacement from the last known close, scaled by past range.",
        "The future hour's opening price is not a predictor. Every complete test hour is forecast.",
        "",
        "## Accuracy on later data",
        "",
        "| Forecast | Within +/-20 | MAE, points | MSE, points² | RMSE, points | Move R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for config in (
        "earlier_selected_within20",
        "earlier_selected",
        "clock_median",
        "last_price_plus20",
        "historical_median",
        "historical_mean",
        "volatility_maximum",
        "last_price",
    ):
        r = summary.loc[config]
        rows.append(
            f"| {config} | {r.within_20_points:.1%} | {r.mae_points:.2f} | "
            f"{r.mse_points_squared:.2f} | {r.rmse_points:.2f} | {r.move_r2:.3f} |"
        )
    rows += [
        "",
        "The two selected policies choose models using only the preceding tuning period:",
        "The +20 baseline always forecasts last known price plus 20; it uses no future data.",
        "one minimizes MAE; the other maximizes the fraction within 20 points (MAE breaks ties).",
        f"The within-20 policy's months range from {chosen_months.within_20_points.min():.1%} "
        f"to {chosen_months.within_20_points.max():.1%}; worst month: {worst.month}.",
        f"Ten-session block 95% interval for the overall within-20 rate: "
        f"{intervals['earlier_selected_within20']['within20_95_low']:.1%}–"
        f"{intervals['earlier_selected_within20']['within20_95_high']:.1%}.",
        "These intervals do not adjust for the repeated use of development data.",
        f"The 90th-percentile absolute error is {p.p90_absolute_error:.2f} points; "
        f"the largest miss is {errors.absolute_error_points.max():.2f} points.",
        "",
        "## Monthly consistency",
        "",
        f"![Forecast accuracy]({(out / 'accuracy.png').as_posix()})",
        "",
        "| Month | Forecasts | Within +/-20 | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in chosen_months.itertuples():
        rows.append(
            f"| {r.month} | {r.observations} | {r.within_20_points:.1%} | "
            f"{r.mae_points:.2f} | {r.rmse_points:.2f} |"
        )
    rows += [
        "",
        "## Market conditions",
        "",
        "| Condition | Forecasts | Within +/-20 | MAE | RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in (
        pd.DataFrame(regimes).loc[lambda f: f.policy.eq("earlier_selected_within20")].itertuples()
    ):
        rows.append(
            f"| {r.regime} | {r.observations} | {r.within_20_points:.1%} | "
            f"{r.mae_points:.2f} | {r.rmse_points:.2f} |"
        )
    rows += [
        "",
        "## Reading the metrics",
        "",
        "MAE is average absolute price error. MSE is average squared error in points².",
        "RMSE is in points and penalizes large misses more strongly. Lower is better.",
        "Move R² measures the next-hour high's displacement from the known reference price.",
        "A high R² on raw price levels alone can be misleading because price levels persist.",
        "Within +/-20 counts forecasts whose absolute error is at most 20 points, including 20.",
        "Quantile coverage is separate: an 80% high estimate should contain about 80% of highs;",
        "that does not mean it predicts the exact high within 20 points 80% of the time.",
        "",
        "## Quantile reliability",
        "",
        "| Nominal quantile | Actual high below forecast | MAE |",
        "|---|---:|---:|",
    ]
    for r in pd.DataFrame(qmetrics).loc[lambda f: f.config.eq("earlier_selected")].itertuples():
        rows.append(f"| {r.quantile:.0%} | {r.actual_below_prediction:.1%} | {r.mae_points:.2f} |")
    rows += [
        "",
        "## Research basis",
        "",
        "- [Andersen et al., realized volatility]"
        "(https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/0210.pdf): "
        "motivates intraday variance inputs.",
        "- [Liu et al., semivariance]"
        "(https://www.sciencedirect.com/science/article/pii/S0927539823000245): "
        "motivates positive/negative variance asymmetry.",
        "- [Man and Chan](https://arxiv.org/abs/2005.12483): feature-selection stability matters. "
        "Clustering is our redundancy-reduction adaptation, not a replication of their paper.",
        "- [Quantile regression documentation]"
        "(https://scikit-learn.org/stable/auto_examples/ensemble/"
        "plot_gradient_boosting_quantile.html): conditional quantiles and coverage measurement.",
        "",
        "The papers motivate methods; they do not validate this next-hour NQ high forecast.",
        "Saved forecasts, every monthly fit, feature columns, metrics and replay audit "
        "accompany this report.",
        "",
        "## Limits",
        "",
        "This is reused development data from 2023-08-22 through 2025-08-15.",
        "The consumed sealed year was not opened. There is no verified fresh final holdout.",
        "NQ is a legacy continuous export without a verified dated volume-roll map.",
        "Forecasts assume information from the preceding minute is available at the boundary.",
        "Incomplete future hours are excluded from evaluation; first warmup periods are excluded.",
        "A quantile residual adjustment is calibrated on earlier data; "
        "coverage is measured, not guaranteed.",
        f"Run: {out.name}. The best observed fixed variant was {best.name}; it is exploratory.",
        "Within-20 gain over the fixed +20 baseline: "
        f"{p.within_20_points - baseline.within_20_points:+.1%}.",
        "Paired session-block 95% interval for that gain: "
        f"{skill['within20_gain_95_low']:+.1%} to {skill['within20_gain_95_high']:+.1%}.",
        "Paired session-block 95% interval for MAE reduction versus that baseline: "
        f"{skill['mae_reduction_95_low']:.2f} to {skill['mae_reduction_95_high']:.2f} points.",
    ]
    report = BASE / "HOUR_HIGH_RESULTS.md"
    report.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(report)
    print(summary.sort_values("within_20_points", ascending=False).head(8).to_string())
    print(
        pd.DataFrame(regimes)
        .loc[lambda f: f.policy.eq("earlier_selected_within20")]
        .to_string(index=False)
    )
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
