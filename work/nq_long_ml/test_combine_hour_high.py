"""Independent monthly result storage must preserve coverage and reject overlaps."""

import json

import combine_hour_high as merger
import joblib
import numpy as np
import pandas as pd
import pytest


def make_chunk(root, folds):
    root.mkdir(parents=True)
    index = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")
    labels = pd.DataFrame({"actual_high": 1010 + np.arange(24)}, index=index)
    labels.to_parquet(root / "labels.parquet")
    tables, selected, quantiles, tuning, features = [], [], [], [], []
    for fold in folds:
        clock = index[(fold - 1) * 2 : fold * 2]
        frame = pd.DataFrame(
            {
                "actual_high": [1010 + fold, 1012 + fold],
                "reference": 1000.0,
                "predicted_high": [1011 + fold, 1013 + fold],
                "fold": fold,
                "config": "fixed_model",
            },
            index=clock,
        )
        tables.append(frame)
        for policy in ("earlier_selected", "earlier_selected_within20"):
            selected.append(frame.assign(policy=policy))
        quantiles.append(frame)
        tuning.append({"fold": fold, "config": "fixed_model", "tune_mae": 1})
        features.append({"fold": fold, "group": "nq_only", "feature": "past_return"})
        joblib.dump({"fold": fold}, root / f"model_month_{fold:02d}.joblib")
    pd.concat(tables).to_parquet(root / "forecasts.parquet")
    pd.concat(selected).to_parquet(root / "selected.parquet")
    pd.concat(quantiles).to_parquet(root / "quantiles.parquet")
    pd.DataFrame(tuning).to_csv(root / "tuning.csv", index=False)
    pd.DataFrame(features).to_csv(root / "features.csv", index=False)
    (root / "protocol.json").write_text(json.dumps({"target": "next_hour_high"}), encoding="utf-8")
    return root


def test_disjoint_months_merge_with_independently_known_error(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "BASE", tmp_path / "output")
    chunks = [make_chunk(tmp_path / "a", range(1, 7)), make_chunk(tmp_path / "b", range(7, 13))]
    merger.combine(chunks)
    out = next((tmp_path / "output/hour_high_runs").iterdir())
    summary = pd.read_csv(out / "summary.csv")
    assert summary.observations.eq(24).all()
    assert summary.mae_points.eq(1).all()
    assert summary.mse_points_squared.eq(1).all()
    assert summary.within_20_points.eq(1).all()
    assert len(list(out.glob("model_month_*.joblib"))) == 12


def test_overlapping_worker_months_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "BASE", tmp_path / "output")
    chunk = make_chunk(tmp_path / "a", [1])
    with pytest.raises(ValueError, match="Overlapping"):
        merger.combine([chunk, chunk])


def test_incomplete_year_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(merger, "BASE", tmp_path / "output")
    chunk = make_chunk(tmp_path / "a", range(1, 12))
    with pytest.raises(ValueError, match="twelve"):
        merger.combine([chunk])
