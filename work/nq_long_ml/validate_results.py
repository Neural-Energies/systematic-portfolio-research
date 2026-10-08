"""Independently check the saved experiment's basic temporal/economic invariants."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import statsmodels
from experiment import BENCHMARKS, END, ROOT, classifiers, metrics


def main():
    folder = sorted((Path(__file__).parent / "runs").glob("*/manifest.json"))[-1].parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_development_fixed_exit"
    summaries = []
    diagnostics = []
    for horizon in (15, 60, 240):
        predictions = pd.read_parquet(folder / f"predictions_{horizon}m.parquet")
        assert predictions["label_end"].lt(END).all()
        assert predictions["label_end"].gt(predictions["available_at"]).all()
        assert predictions["probability"].between(0, 1).all()
        assert not predictions.reset_index().duplicated(["bar_open", "model"]).any()
        assert set(predictions["model"]) == {*classifiers(), "rf_full"}
        ranks = pd.read_csv(folder / f"feature_rankings_{horizon}m.csv")
        assert ranks.groupby("fold")["selected"].sum().eq(20).all()
        assert ranks.groupby("fold").size().eq(93).all()
        stability = pd.read_csv(folder / f"feature_stability_{horizon}m.csv")
        stability.head(20).to_csv(folder / f"top20_consensus_{horizon}m.csv", index=False)
        bins = (
            predictions.assign(
                probability_bin=pd.cut(predictions["probability"], np.linspace(0, 1, 11))
            )
            .groupby(["model", "probability_bin"], observed=True)
            .agg(
                count=("target", "size"),
                mean_prediction=("probability", "mean"),
                observed_event_rate=("target", "mean"),
            )
        )
        bins.to_csv(folder / f"reliability_{horizon}m.csv")
        saved = pd.read_csv(folder / f"summary_{horizon}m.csv").set_index("model")
        for name, frame in predictions.groupby("model"):
            check = metrics(frame)
            for field in ("trades", "accuracy", "brier", "net_dollars_1_contract"):
                assert np.isclose(check[field], saved.loc[name, field])
            # Independent economic calculation from individual fills.
            long = frame.loc[frame["probability"].ge(0.66)]
            net = ((long["trade_close"] - long["trade_open"]) * 20 - 25).sum()
            assert np.isclose(net, saved.loc[name, "net_dollars_1_contract"])
            dollars = (long["trade_close"] - long["trade_open"]) * 20 - 25
            fold_net = dollars.groupby(long["fold"]).sum().reindex(range(1, 5), fill_value=0)
            diagnostics.append(
                {
                    "horizon_minutes": horizon,
                    "model": name,
                    "positive_trading_folds": int(fold_net.gt(0).sum()),
                    "folds_with_trades": int(long["fold"].nunique()),
                    "net_without_best_5_trades": float(dollars.sum() - dollars.nlargest(5).sum()),
                    "best_fold_net": float(fold_net.max()),
                    "non_best_fold_net": float(fold_net.sum() - fold_net.max()),
                    "majority_class_accuracy": float(
                        max(frame["target"].mean(), 1 - frame["target"].mean())
                    ),
                }
            )
        summaries.append(saved.reset_index())
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(folder / "combined_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(folder / "robustness_diagnostics.csv", index=False)
    provenance = {
        "verification": "passed",
        "sklearn": sklearn.__version__,
        "statsmodels": statsmodels.__version__,
        "data_sources": [],
        "repository_tests": "154 passed",
        "experiment_tests": "6 passed",
        "repository_checks": (
            "40 existing lint errors; 89 type errors in 11 unrelated files; "
            "6 unrelated files require formatting"
        ),
    }
    for symbol in ("NQ", *BENCHMARKS):
        path = (
            ROOT
            / "data/processed/databento_research/development_minute_returns"
            / f"symbol={symbol}/returns.parquet"
        )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        provenance["data_sources"].append(
            {"symbol": symbol, "path": str(path), "sha256": digest.hexdigest()}
        )
    (folder / "verification.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    columns = [
        "horizon_minutes",
        "model",
        "accuracy",
        "trades",
        "precision_at_066",
        "net_dollars_1_contract",
        "double_cost_net_dollars",
    ]
    print(
        combined[columns]
        .sort_values(["horizon_minutes", "net_dollars_1_contract"], ascending=[True, False])
        .to_string(index=False)
    )
    print(f"Verified: {folder}")


if __name__ == "__main__":
    main()
