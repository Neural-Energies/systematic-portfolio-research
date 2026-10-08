"""Controlled tests of intermarket information and randomly shuffled training labels."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from direction_expanded import ProbabilityCalibration
from experiment import END, SEED
from fifteen_probability import BASE, catalog, measure, prepare, time_parts
from medium_frequency import FIRST_TEST, raw_scores
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def run():
    out = BASE / "fifteen_ablation_runs" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    data, x, _, _ = prepare()
    limits = list(pd.date_range(FIRST_TEST, periods=12, freq=pd.DateOffset(months=1))) + [END]
    records, forecasts = [], []
    for fold, (start, end) in enumerate(zip(limits[:-1], limits[1:], strict=True), 1):
        parts = time_parts(data, start, end)
        fit, cal, tune, test = parts
        usable = [c for c in x if x.loc[fit.index, c].notna().sum() >= 60]
        groups = {
            "nq_only": [c for c in usable if not c.startswith("market_")],
            "nq_es": [
                c for c in usable if not c.startswith("market_") or c.startswith("market_ES_")
            ],
            "all_markets": usable,
        }
        # This is a controlled ablation: fixed settings; no test-informed feature selection.
        for window in (90, 180):
            fitted = fit.loc[fit.index >= start - pd.Timedelta(days=window)]
            for group, columns in groups.items():
                for family in ("logistic", "hist_boost"):
                    print(f"month {fold}: {window}d {group} {family}", flush=True)
                    model = make_pipeline(
                        SimpleImputer(strategy="median"), StandardScaler(), clone(catalog()[family])
                    ).fit(x.loc[fitted.index, columns], fitted.target)
                    sc, su, st = [raw_scores(model, x.loc[p.index, columns]) for p in parts[1:]]
                    calibration = ProbabilityCalibration("sigmoid").fit(sc, cal.target)
                    pu, pt = calibration.predict(su), calibration.predict(st)
                    config = f"{window}d_{group}_{family}"
                    records.append(
                        {
                            "fold": fold,
                            "config": config,
                            "features": len(columns),
                            "tune_brier": measure(tune, pu)["brier"],
                        }
                    )
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
                    frame["probability"], frame["config"], frame["fold"] = pt, config, fold
                    forecasts.append(frame)
        rng = np.random.default_rng(SEED + fold)
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), clone(catalog()["hist_boost"])
        )
        model.fit(x.loc[fit.index, groups["all_markets"]], rng.permutation(fit.target.to_numpy()))
        scores = [raw_scores(model, x.loc[p.index, groups["all_markets"]]) for p in (cal, test)]
        calibration = ProbabilityCalibration("sigmoid").fit(scores[0], cal.target)
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
        frame["probability"], frame["config"], frame["fold"] = (
            calibration.predict(scores[1]),
            "shuffled_label_control",
            fold,
        )
        forecasts.append(frame)
        pd.DataFrame(records).to_csv(out / "tuning.csv", index=False)
    frame = pd.concat(forecasts)
    frame.to_parquet(out / "forecasts.parquet")
    pd.DataFrame(
        [
            {"config": config, **measure(g, g.probability.to_numpy())}
            for config, g in frame.groupby("config")
        ]
    ).to_csv(out / "summary.csv", index=False)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "selection": "none; fixed ablations compared exploratorily",
                "windows": [90, 180],
                "threshold": 0.66,
                "retraining": "monthly",
                "control": "independently shuffled fit labels; true earlier calibration labels",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"COMPLETE {out}", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
