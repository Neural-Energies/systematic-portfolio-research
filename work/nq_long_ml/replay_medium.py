"""Replay frozen forecasts after correcting session bookkeeping and a known closure."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from medium_frequency import evaluate_predictions, prepare, session_keys


def main():
    root = Path(__file__).parent
    parent = root / "medium_runs" / "20261003T024104Z"
    out = root / "medium_runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_replay")
    out.mkdir()
    minutes, _, _ = prepare()
    summaries, folds = [], []
    for name in ("4h", "1d", "3d", "5d"):
        predictions = pd.read_parquet(parent / f"forecasts_{name}.parquet")
        predictions["session"] = session_keys(predictions.index)
        predictions.to_parquet(out / f"forecasts_{name}.parquet")
        summary, fold = evaluate_predictions(predictions, minutes, out, name)
        summaries.extend(summary)
        folds.extend(fold)
    pd.DataFrame(summaries).to_csv(out / "combined_summary.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "fold_results.csv", index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete_development_provisional",
                "parent": str(parent),
                "forecasts_and_tuned_thresholds_unchanged": True,
                "corrections": ["decision-session identity", "published Carter Day 08:00 deadline"],
                "fresh_holdout_available": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out / "source_snapshot.py").write_text(
        (root / "medium_frequency.py").read_text(), encoding="utf-8"
    )
    print(out, flush=True)


if __name__ == "__main__":
    main()
