"""Merge disjoint monthly research workers with strict overlap and coverage checks."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fifteen_probability import BASE
from hour_high import regression_metrics


def combine(chunks: list[Path]):
    out = BASE / "hour_high_runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_combined")
    out.mkdir(parents=True)
    tables = {name: [] for name in ("forecasts", "selected", "quantiles", "tuning", "features")}
    labels = pd.read_parquet(chunks[0] / "labels.parquet")
    seen = set()
    for chunk in chunks:
        pd.testing.assert_frame_equal(labels, pd.read_parquet(chunk / "labels.parquet"))
        tuning = pd.read_csv(chunk / "tuning.csv")
        folds = set(tuning.fold.unique())
        if seen & folds:
            raise ValueError("Overlapping monthly workers")
        seen |= folds
        for fold in folds:
            model = chunk / f"model_month_{fold:02d}.joblib"
            if not model.exists():
                raise ValueError("Unfinished worker: missing monthly model")
            shutil.copyfile(model, out / model.name)
        for name in tables:
            frame = (
                pd.read_csv(chunk / f"{name}.csv")
                if name in ("tuning", "features")
                else pd.read_parquet(chunk / f"{name}.parquet")
            )
            if set(frame.fold.unique()) != folds:
                raise ValueError(f"Unfinished worker table: {name}")
            tables[name].append(frame)
    if seen != set(range(1, 13)):
        raise ValueError("All twelve monthly folds must be present")
    merged = {
        name: pd.concat(frames).sort_values("fold", kind="stable")
        for name, frames in tables.items()
    }
    for name, frame in merged.items():
        if name in ("tuning", "features"):
            frame.to_csv(out / f"{name}.csv", index=False)
        else:
            frame.to_parquet(out / f"{name}.parquet")
    forecasts, selected = merged["forecasts"], merged["selected"]
    for _, frame in forecasts.groupby("config"):
        if frame.index.duplicated().any():
            raise ValueError("Repeated forecast in fixed model")
    all_forecasts = pd.concat([forecasts, selected])
    chosen = all_forecasts.policy.notna()
    all_forecasts.loc[chosen, "config"] = all_forecasts.loc[chosen, "policy"]
    rows = [
        {
            "config": config,
            **regression_metrics(
                frame.actual_high.to_numpy(),
                frame.predicted_high.to_numpy(),
                frame.reference.to_numpy(),
            ),
        }
        for config, frame in all_forecasts.groupby("config")
    ]
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    labels.to_parquet(out / "labels.parquet")
    protocol = json.loads((chunks[0] / "protocol.json").read_text(encoding="utf-8"))
    protocol["monthly_worker_sources"] = [str(c) for c in chunks]
    protocol["source_papers"] = [
        "https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2014/04/0210.pdf"
        if link == "https://www.bis.org/cgfs/Diebold-et-al.pdf"
        else link
        for link in protocol.get("source_papers", [])
    ]
    protocol["execution"] = (
        "Independent monthly folds computed locally in isolated output directories"
    )
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", nargs="+", type=Path)
    args = parser.parse_args()
    combine(args.chunks)
