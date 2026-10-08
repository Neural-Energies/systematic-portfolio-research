"""Audit the exact 15-minute forecasts and explain successes and failures."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from fifteen_probability import BASE, measure, wilson_lower


def stability(frame: pd.DataFrame) -> dict:
    trade = frame.probability.ge(0.66)
    if "gate" in frame:
        trade &= frame.gate
    signals = frame.loc[trade]
    months = signals.groupby(signals.index.strftime("%Y-%m")).net_dollars.sum()
    days = frame.groupby("session").apply(
        lambda g: pd.Series(
            {
                "trades": ((g.probability >= 0.66) & g.get("gate", True)).sum(),
                "wins": g.loc[(g.probability >= 0.66) & g.get("gate", True), "target"].sum(),
            }
        ),
        include_groups=False,
    )
    rng = np.random.default_rng(20261003)
    estimates = []
    for _ in range(2000):
        starts = rng.integers(0, len(days), int(np.ceil(len(days) / 10)))
        sample = np.concatenate([(a + np.arange(10)) % len(days) for a in starts])[: len(days)]
        sampled = days.iloc[sample].sum()
        if sampled.trades:
            estimates.append(sampled.wins / sampled.trades)
    lo, hi = np.quantile(estimates, [0.025, 0.975]) if estimates else (np.nan, np.nan)
    return {
        "signal_days": int(signals.session.nunique()),
        "signal_months": len(months),
        "positive_months": int(months.gt(0).sum()),
        "net_without_best_month": float(signals.net_dollars.sum() - months.max())
        if len(months)
        else 0,
        "net_without_best5_trades": float(
            signals.net_dollars.sum() - signals.net_dollars.nlargest(5).sum()
        ),
        "direction_ci_low": float(lo),
        "direction_ci_high": float(hi),
        "direction_iid_wilson95_low": wilson_lower(
            float(signals.target.mean()), len(signals), 1.95996398454
        ),
        "max_drawdown": float(
            (
                signals.net_dollars.cumsum().cummax().clip(lower=0) - signals.net_dollars.cumsum()
            ).max()
        )
        if len(signals)
        else 0,
    }


def run():
    main = sorted((BASE / "fifteen_runs").glob("*/model_summary.csv"))[-1].parent
    ablation = sorted((BASE / "fifteen_ablation_runs").glob("*/summary.csv"))[-1].parent
    data, x, _, _ = joblib.load(BASE / "fifteen_features_v1.joblib")
    forecasts = pd.read_parquet(main / "forecasts.parquet")
    picked = pd.read_parquet(main / "selected.parquet")
    alternatives = pd.read_parquet(ablation / "forecasts.parquet")
    audits = []
    for name, frame in (("gauntlet", forecasts), ("selected", picked), ("ablation", alternatives)):
        expected = data.reindex(frame.index)
        for column in (
            "target",
            "gross_dollars",
            "net_dollars",
            "stress_dollars",
            "delayed_net_dollars",
        ):
            if not np.allclose(frame[column], expected[column]):
                raise AssertionError(f"{name}: {column} replay mismatch")
        if not ((frame.label_end - frame.index) == pd.Timedelta(minutes=15)).all():
            raise AssertionError("Exit is not next bar close")
        if not frame.probability.between(0, 1).all():
            raise AssertionError("Invalid probabilities")
        audits.append(
            {"dataset": name, "predictions": len(frame), "literal_open_close_replay": True}
        )
    tuning = pd.read_csv(main / "monthly_tuning.csv")
    for fold, group in picked.groupby("fold"):
        candidates = tuning.loc[tuning.fold.eq(fold)].set_index("config")
        probability_winner = candidates.tune_brier.idxmin()
        eligible = candidates.loc[candidates.tune_trades.ge(30)].copy()
        eligible["lower"] = [
            wilson_lower(r.tune_direction_win, r.tune_trades) for r in eligible.itertuples()
        ]
        eligible = eligible.loc[eligible.lower.ge(0.55)]
        selective_winner = eligible.lower.idxmax() if len(eligible) else "abstain"
        for policy, expected in (
            ("probability_champion", probability_winner),
            ("qualified_selective", selective_winner),
        ):
            policy_frame = group.loc[group.policy.eq(policy)]
            if not policy_frame.config.eq(expected).all():
                raise AssertionError("Model selection used different criteria from protocol")
            if expected != "abstain":
                candidate = forecasts.loc[forecasts.fold.eq(fold) & forecasts.config.eq(expected)]
                if (
                    not candidate.index.equals(policy_frame.index)
                    or not np.allclose(candidate.probability, policy_frame.probability)
                    or not candidate.gate.equals(policy_frame.gate)
                ):
                    raise AssertionError("Selected predictions differ from candidate")
    for key in ("fit_end", "cal_end", "tune_end"):
        populated = tuning[key].notna()
        if not (
            pd.to_datetime(tuning.loc[populated, key], utc=True)
            < pd.to_datetime(tuning.loc[populated, "test_start"], utc=True)
        ).all():
            raise AssertionError("Training boundary crossed test")
    (main / "audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    policies = pd.read_csv(main / "policy_summary.csv").set_index("policy")
    models = pd.read_csv(main / "model_summary.csv").set_index("config")
    comparisons = pd.read_csv(ablation / "summary.csv").set_index("config")
    policy_stability = pd.DataFrame(
        [{"policy": name, **stability(group)} for name, group in picked.groupby("policy")]
    ).set_index("policy")
    ablation_stability = pd.DataFrame(
        [{"config": name, **stability(group)} for name, group in alternatives.groupby("config")]
    ).set_index("config")
    policy_stability.to_csv(main / "policy_stability.csv")
    ablation_stability.to_csv(main / "ablation_stability.csv")
    reliability = []
    for policy, group in picked.groupby("policy"):
        for lo, hi in (
            (0, 0.45),
            (0.45, 0.50),
            (0.50, 0.55),
            (0.55, 0.60),
            (0.60, 0.66),
            (0.66, 0.70),
            (0.70, 0.80),
            (0.80, 1.000001),
        ):
            bucket = group.loc[group.probability.ge(lo) & group.probability.lt(hi) & group.gate]
            reliability.append(
                {
                    "policy": policy,
                    "bucket": f"{lo:.2f}-{hi:.2f}",
                    "count": len(bucket),
                    "mean_forecast": bucket.probability.mean(),
                    "actual_up": bucket.target.mean(),
                }
            )
    pd.DataFrame(reliability).to_csv(main / "reliability.csv", index=False)
    model_months = []
    for (config, fold), group in forecasts.groupby(["config", "fold"]):
        model_months.append(
            {
                "config": config,
                "fold": fold,
                **measure(group, group.probability.to_numpy(), group.gate.to_numpy()),
            }
        )
    pd.DataFrame(model_months).to_csv(main / "model_months.csv", index=False)
    test_index = picked.loc[picked.policy.eq("probability_champion")].index
    subset = data.loc[test_index]
    correlations = []
    for market in ("ES", "RTY", "ZN"):
        known = x[f"market_{market}_return"].reindex(test_index)
        # The future comparison is diagnostic ONLY; it is never passed to a model.
        future = x[f"market_{market}_return"].reindex(test_index + pd.Timedelta(minutes=15))
        future.index = test_index
        target_ret = np.log(subset.close / subset.open)
        correlations.append(
            {
                "market": market,
                "known_previous_bar_correlation": known.corr(target_ret),
                "future_same_bar_correlation_LOOKAHEAD_NOT_FEATURE": future.corr(target_ret),
            }
        )
    pd.DataFrame(correlations).to_csv(main / "lead_lag_diagnostic.csv", index=False)

    p = policies.loc["probability_champion"]
    s = policy_stability.loc["probability_champion"]
    best = comparisons.loc["180d_nq_es_logistic"]
    bs = ablation_stability.loc["180d_nq_es_logistic"]
    lines = [
        "# Exact 15-minute NQ probability results",
        "",
        "**Rule:** at the next 15-minute open, buy one NQ contract if P(close > open) >=66%; exit at that bar's close. No targets, learned exits, four-hour holding, daily quota or reduced threshold.",
        "",
        f"Tested {len(subset):,} later development bars across {subset.session.nunique()} sessions, with monthly retraining, 325 raw candidate inputs, past-only factor/PCA transformations and top-20/top-100 feature screening. All inputs end at or before entry. Calibration and model choice use earlier periods. {len(tuning)} gauntlet configurations and 13 fixed ablation/control configurations were retained; zero-trade results are included.",
        "",
        "## What failed before",
        "",
        "1. The first 15-minute research labeled a cost-covering trade, not simply an up bar. Later research changed the horizon to one/four hours and used a delayed fill. This run fixes those mismatches.",
        "2. Overall accuracy is different from accuracy on the >=66% signals. A model can be around 52% on every bar yet have a useful small subset. We now report both.",
        "3. A confidence score is not evidence of calibrated probability. Softmax does not create predictive information. Raw scores, sigmoid calibration and smoothed isotonic calibration were compared without raising the threshold artificially.",
        "4. More correlated markets can add noise. Controlled NQ-only/NQ+ES/all-market tests isolate that effect.",
        "5. Feature screening had a schema bug when new markets were absent in the earliest fit slice. The missing-column fix has a regression test.",
        "",
        "## Selected before each test month",
        "",
        "| Policy | All-bar accuracy | Trades >=66% | Actual up | Mean forecast | Net, $25 cost | Net, 1-minute delay |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in policies.iterrows():
        lines.append(
            f"| {name} | {r.accuracy:.1%} | {int(r.trades)} | {r.direction_win:.1%} | {r.forecast_probability:.1%} | ${r.net_dollars:,.0f} | ${r.delayed_net_dollars:,.0f} |"
        )
    lines += [
        "",
        f"Always-up accuracy is {p.up_baseline:.1%}. The probability champion's {int(p.trades)} signals occurred on {int(s.signal_days)} days in {int(s.signal_months)} months. Its session-block 95% direction interval is {s.direction_ci_low:.1%}–{s.direction_ci_high:.1%}; this does not adjust for repeated research. Net without the best month is ${s.net_without_best_month:,.0f}; without the best five trades, ${s.net_without_best5_trades:,.0f}. Trade-close drawdown is ${s.max_drawdown:,.0f}, excluding intrabar adverse excursion.",
        "",
        "The selective policy required >=30 earlier tuning trades and an 80% Wilson lower bound of at least 55%. That qualification was not sufficient to make future confidence reliable. It is an experiment, not a replacement for your entry rule.",
        "",
        "## Does ES help? Fixed comparisons",
        "",
        "| Method | All-bar accuracy | Trades >=66% | Actual up | Net, $25 cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "180d_nq_only_logistic",
        "180d_nq_es_logistic",
        "180d_all_markets_logistic",
        "180d_nq_only_hist_boost",
        "180d_nq_es_hist_boost",
        "180d_all_markets_hist_boost",
        "90d_nq_es_logistic",
        "shuffled_label_control",
    ):
        r = comparisons.loc[name]
        rate = f"{r.direction_win:.1%}" if r.trades else "—"
        lines.append(
            f"| {name} | {r.accuracy:.1%} | {int(r.trades)} | {rate} | ${r.net_dollars:,.0f} |"
        )
    lines += [
        "",
        f"The NQ+ES logistic candidate reached {best.direction_win:.1%} on {int(best.trades)} signals across {int(bs.signal_days)} days and {int(bs.signal_months)} months. Its session-block direction interval is {bs.direction_ci_low:.1%}–{bs.direction_ci_high:.1%}. Net without its best month is ${bs.net_without_best_month:,.0f}; without its best five trades, ${bs.net_without_best5_trades:,.0f}. This is a promising exploratory candidate, identified after comparing variants. It is not a confirmed 73.7% edge, and it does not predict all bars at that accuracy.",
        "",
        "## Techniques actually tested",
        "",
        "Regularized logistic regression, shrinkage LDA, naive Bayes, random forests, extra trees and histogram gradient boosting; a five-quantile return-distribution model; a shrunk time-of-day/momentum/volume probability table; strategy-conditional boosting for adequately supported setups. Fixed setup baselines include noise-area/VWAP breakouts, opening-range breakouts, dip recovery, fast momentum and late-session momentum. The model also uses prior same-clock returns/volume, minute price-volume proxies, a rolling AR(1) forecast and completed 15-minute/hour/four-hour/daily information. Missing-market coverage is disclosed.",
        "",
        "## Research used",
        "",
        "- [Heston, Korajczyk and Sadka](https://arxiv.org/abs/1005.3535): motivates same-clock histories and short reversals. Their cross-sectional equity evidence is not validation of NQ.",
        "- [Gao et al., Market Intraday Momentum](https://www.sciencedirect.com/science/article/pii/S0304405X18301351): motivates completed first-half-hour information for late-session conditions. Our 15-minute exit is an adaptation.",
        "- [Zarattini, Aziz and Barbon](https://concretumgroup.com/wp-content/uploads/2026/02/Beat-the-Market.pdf): motivates a prior-14-session same-clock noise area with gap adjustment and VWAP confirmation. We adapt the entry conditions; we do not reproduce their trailing-stop SPY strategy or claim their returns.",
        "- [Kumbure et al., survey](https://doi.org/10.1016/j.eswa.2022.116659) and [Grinsztajn et al., tabular benchmarks](https://arxiv.org/abs/2207.08815): support comparing feature/model families; neither promises intraday accuracy.",
        "- [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html): confidence calibration differs from classification accuracy.",
        "- [Ernest Chan's ML research](https://predictnow.ai/resources-ai-methodology/): next-trade probability, feature selection and regime adaptation are relevant research directions. This run's conditional models were implemented before the Chan reference; they are not presented as a replication of his proprietary models.",
        "",
        "## What the evidence supports",
        "",
        "Focus next on the NQ+ES conditional opportunity and verify its stability on genuinely new data. Keep >=66%, label direction independently of costs, and measure probability reliability, trade frequency and net expectancy together. Adding every market and selecting on a short profitable window did not reliably improve the forecasts.",
        "",
        "These are reused development data, 2023-08-22 through 2025-08-15. The original sealed year was previously consumed and was not opened here. NQ is a legacy continuous export without a verified dated volume-roll map. Extension markets select the previous completed UTC day's volume leader; this is disclosed rather than claimed equivalent to a two-session volume-crossover roll. Literal open execution assumes instantaneous calculation at the close/open boundary; delay and higher costs are separate sensitivity tests. There is no live order system or verified untouched final year.",
        "",
        f"Audit: all {len(forecasts) + len(picked) + len(alternatives):,} saved forecasts replay their target, literal open/close dollars and delayed-fill sensitivity against source bars. Earlier-only model selections also replay. Causality tests remove future minutes and require unchanged earlier features. Source, costs, monthly tables, probabilities, reliability, stability and feature rankings are saved under `{main.name}` and ablation `{ablation.name}`.",
    ]
    report = BASE / "FIFTEEN_MINUTE_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)
    print(policy_stability.to_string())
    print(ablation_stability.loc[["180d_nq_es_logistic", "180d_all_markets_logistic"]].to_string())
    print(
        models.loc[models.trades.ge(100)]
        .sort_values("direction_win", ascending=False)
        .head(5)
        .to_string()
    )


if __name__ == "__main__":
    run()
