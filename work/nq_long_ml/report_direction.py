"""Audit saved next-bar forecasts, earlier model selection and probability reliability."""

# Report paragraphs are kept as readable literal strings.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from direction_expanded import ProbabilityCalibration, metrics
from medium_frequency import raw_scores

import __main__ as pickle_main


def session_bootstrap(frame: pd.DataFrame, signal_only: bool = False):
    days = pd.DatetimeIndex(sorted(frame.session.unique()))
    data = frame.assign(
        successes=frame.target
        if signal_only
        else frame.probability.ge(0.5).eq(frame.target).astype(int),
        included=frame.probability.ge(0.66) if signal_only else True,
    )
    data["successes"] *= data.included
    daily = (
        data.groupby("session")
        .agg(success=("successes", "sum"), count=("included", "sum"))
        .reindex(days, fill_value=0)
    )
    rng = np.random.default_rng(20261003)
    ratios = []
    width = min(10, len(days))
    for _ in range(1000):
        starts = rng.integers(0, len(days), size=int(np.ceil(len(days) / width)))
        selected = np.concatenate([(start + np.arange(width)) % len(days) for start in starts])[
            : len(days)
        ]
        s, n = daily.iloc[selected].sum()
        if n:
            ratios.append(float(s / n))
    return np.quantile(ratios, [0.025, 0.975]).tolist() if ratios else [None, None]


def main():
    root = Path(__file__).parent
    out = root / "direction_runs" / "20261003T042947Z"
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete_development_only":
        raise ValueError("Training incomplete")
    x = pd.read_parquet(out / "causal_features.parquet")
    markets = pd.read_parquet(out / "market_returns.parquet")
    # Trusted local research artifacts originally refer to the script's __main__ class.
    pickle_main.ProbabilityCalibration = ProbabilityCalibration
    assert x.index.is_unique and x.index.is_monotonic_increasing
    assert manifest["raw_feature_candidates"] == x.shape[1]
    summary = pd.read_csv(out / "combined_summary.csv")
    reliability, checks, intervals, leadlag = [], [], [], []
    for horizon in (60, 240):
        forecasts = pd.read_parquet(out / f"forecasts_{horizon}.parquet")
        chosen = pd.read_parquet(out / f"chosen_{horizon}.parquet")
        diagnostic = pd.read_csv(out / f"diagnostics_{horizon}.csv")
        assert len(diagnostic) == 320
        assert forecasts.probability.between(0, 1).all()
        assert forecasts.target.eq(forecasts.bar_close.gt(forecasts.bar_open).astype(int)).all()
        assert (
            (
                forecasts.label_end - pd.Series(forecasts.index, index=forecasts.index)
            ).dt.total_seconds()
            == horizon * 60
        ).all()
        assert np.allclose(
            forecasts.net_dollars, (forecasts.bar_close - forecasts.delayed_entry) * 20 - 25
        )
        for policy, frame in chosen.groupby("selection_policy"):
            assert frame.index.is_unique
            for fold, part in frame.groupby("fold"):
                records = diagnostic.loc[diagnostic.fold.eq(fold)]
                config = part.config.unique()
                assert len(config) == 1
                selected = records.loc[records.config.eq(config[0])].iloc[0]
                qualified = records.loc[
                    records.tune_signals.ge(40) & records.tune_signal_accuracy.ge(0.60)
                ]
                if policy == "probability" or (policy == "selective_066" and qualified.empty):
                    assert selected.tune_brier == records.tune_brier.min()
                elif policy == "direction_accuracy":
                    assert selected.tune_accuracy == records.tune_accuracy.max()
                else:
                    assert selected.tune_signal_accuracy == qualified.tune_signal_accuracy.max()
                model_path = out / f"model_{horizon}_{fold}_{config[0]}.joblib"
                bundle = joblib.load(model_path)
                matrix = x.reindex(part.index).copy()
                scores = bundle["pca"].transform(markets.reindex(part.index)[bundle["pca_markets"]])
                for component, values in enumerate(scores.T, 1):
                    matrix[f"market_pca_{component}"] = values
                replay = bundle["calibration"].predict(
                    raw_scores(bundle["model"], matrix[bundle["selected_features"]])
                )
                assert np.allclose(replay, part.probability.to_numpy(), atol=1e-10)
                portable = {k: v for k, v in bundle.items() if k != "calibration"}
                portable["calibration_method"] = bundle["calibration"].method
                portable["calibration_model"] = bundle["calibration"].model
                joblib.dump(
                    portable, out / f"portable_{horizon}_{fold}_{policy}.joblib", compress=1
                )
                checks.append(
                    {
                        "horizon": horizon,
                        "policy": policy,
                        "fold": int(fold),
                        "earlier_selection_verified": True,
                        "saved_model_predictions_reproduced": True,
                    }
                )
            row = summary.loc[summary.horizon.eq(horizon) & summary.policy.eq(policy)].iloc[0]
            independent = metrics(frame, frame.probability.to_numpy())
            assert np.isclose(independent["accuracy"], row.accuracy)
            assert independent["signals_066"] == row.signals_066
            interval = session_bootstrap(frame)
            conditional = session_bootstrap(frame, True)
            signal = frame.loc[frame.probability.ge(0.66)]
            intervals.append(
                {
                    "horizon": horizon,
                    "policy": policy,
                    "accuracy_low": interval[0],
                    "accuracy_high": interval[1],
                    "signal_accuracy_low": conditional[0],
                    "signal_accuracy_high": conditional[1],
                    "signal_days": signal.session.nunique(),
                    "all_days": frame.session.nunique(),
                }
            )
            frame = frame.assign(bin=(np.floor(frame.probability * 20) / 20).clip(0, 0.95))
            for lower, group in frame.groupby("bin"):
                reliability.append(
                    {
                        "horizon": horizon,
                        "policy": policy,
                        "lower": lower,
                        "upper": lower + 0.05,
                        "count": len(group),
                        "mean_probability": group.probability.mean(),
                        "observed_up": group.target.mean(),
                    }
                )
        # Diagnostic only: same-period ES return is known after the forecasted bar ends.
        frame = chosen.loc[chosen.selection_policy.eq("probability")].copy()
        nq_return = np.log(frame.bar_close / frame.bar_open)
        field = "market_ES_return" if horizon == 60 else "market_ES_4h_return"
        past = x[field].reindex(frame.index)
        future = x[field].reindex(pd.DatetimeIndex(frame.label_end))
        future.index = frame.index
        leadlag.append(
            {
                "horizon": horizon,
                "es_already_known_correlation": nq_return.corr(past),
                "es_same_future_bar_correlation_LOOKAHEAD_NOT_A_FEATURE": nq_return.corr(future),
            }
        )
    pd.DataFrame(reliability).to_csv(out / "reliability.csv", index=False)
    intervals = pd.DataFrame(intervals)
    intervals.to_csv(out / "session_bootstrap.csv", index=False)
    pd.DataFrame(leadlag).to_csv(out / "correlation_timing_diagnostic.csv", index=False)
    (out / "validation.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "literal_bar_labels": "pass",
                "probability_bounds": "pass",
                "costed_outcome_separate": "pass",
                "candidate_count": 640,
                "estimator_fits": 320,
                "fresh_holdout_available": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    report = [
        "# Expanded NQ direction results",
        "",
        "Ten model families, 18 outside markets, 410 candidate inputs, 320 estimator fits and 640 calibrated quarterly configurations were tested locally. Compare 20/100 inputs, expanding/recent histories and two calibration methods. No daily quota or reduced entry threshold was used.",
        "",
        "The event is **the next complete hourly or four-hour bar closes above its open**. The earlier research measured a different, costed four-hour return. Every reported result below uses later development periods, not training accuracy.",
        "",
        "## Did the models reach 60%?",
        "",
        "| Horizon | Earlier-selected accuracy policy | Majority baseline | Best observed candidate across tests |",
        "|---|---:|---:|---:|",
    ]
    for horizon in (60, 240):
        rows = summary.loc[summary.horizon.eq(horizon)]
        selected = rows.loc[rows.policy.eq("direction_accuracy")].iloc[0]
        best = (
            rows.loc[rows.policy.eq("candidate")].sort_values("accuracy", ascending=False).iloc[0]
        )
        report.append(
            f"| {horizon // 60} hours | {selected.accuracy:.1%} | {selected.majority_baseline:.1%} | {best.accuracy:.1%} |"
        )
    report += [
        "",
        "The best observed candidate is an exploratory maximum after many comparisons. It was not selected before seeing these test results. The earlier-selected policies are the more defensible evidence. No claim of 60% is justified unless the saved results actually demonstrate it.",
        "",
        "## What did probabilities above 66% mean?",
        "",
        "| Horizon | Earlier-selected policy | Signals >=66% | Mean predicted probability | Actual bars up | Signal days |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary.loc[summary.policy.ne("candidate")].itertuples():
        days = intervals.loc[
            intervals.horizon.eq(r.horizon) & intervals.policy.eq(r.policy), "signal_days"
        ].iloc[0]
        mean = "—" if pd.isna(r.mean_signal_probability) else f"{r.mean_signal_probability:.1%}"
        actual = "—" if pd.isna(r.signal_accuracy) else f"{r.signal_accuracy:.1%}"
        report.append(
            f"| {r.horizon // 60} hours | {r.policy} | {r.signals_066} | {mean} | {actual} | {days} |"
        )
    report += [
        "",
        "A forecast of 68% is a model estimate. It becomes useful evidence only when later similar forecasts go up at roughly that rate with enough independent observations. Raising a score or using an overconfident calibrator does not create predictive information. Session-block intervals and bucket reliability are saved alongside the report.",
        "",
        "## Correlated does not necessarily mean predictive",
        "",
        "| Horizon | Known preceding ES return versus future NQ | ES return during the same future bar versus NQ |",
        "|---|---:|---:|",
    ]
    for r in leadlag:
        report.append(
            f"| {r['horizon'] // 60} hours | {r['es_already_known_correlation']:.3f} | {r['es_same_future_bar_correlation_LOOKAHEAD_NOT_A_FEATURE']:.3f} |"
        )
    report += [
        "",
        "The same-future-bar ES comparison is a timing diagnostic only. It cannot be used to forecast that NQ bar because its result is known afterward. The model uses completed ES observations, earlier lags and completed mixed-period context.",
        "",
        "## Coverage repair",
        "",
        "The first expanded run required every predictor minute to exist. That discarded many otherwise usable observations. The corrected run accepts completed predictor periods with at least two observations and a recent last quote (at most ten minutes old for hourly bars), explicitly includes coverage and quote age, and excludes bars spanning contract changes. NQ's future labels still require complete contiguous bars. The first run is preserved as a comparison.",
        "",
        "Added contracts start in September 2024 and enter only when enough earlier fitting data exists. Missing history is not fabricated. Source coverage and feature ranks are saved per market and model window.",
        "",
        "## Limits and verification",
        "",
        "These model tests have not established a deployable 60% direction predictor or reliable 68% long forecast. All candidates and failures are retained. The consumed final year remains excluded; the quarterly periods have informed repeated development and are not pristine validation. NQ's historical volume-roll mapping is still unverified. Eight neural fits in the first run reached their iteration cap; the corrected run's warning count is recorded in its log, rather than silently treated as convergence.",
        "",
        "Labels, probability bounds, delayed-entry costs and earlier-only policy selection were independently checked from saved forecasts. The reproducible research code and tests are local. See [the protocol and survey references](DIRECTION_PROTOCOL.md).",
        "",
        f"Full output: `{out}`.",
    ]
    (root / "DIRECTION_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(summary.loc[summary.policy.ne("candidate")].to_string(index=False))
    print(pd.DataFrame(leadlag).to_string(index=False))


if __name__ == "__main__":
    main()
