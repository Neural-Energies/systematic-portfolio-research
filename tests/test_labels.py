from __future__ import annotations

import numpy as np
import pandas as pd

from systematic_research.labels import build_forward_labels, build_walk_forward_folds


def test_forward_labels_are_separate_and_timestamped_at_realization() -> None:
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    panel = pd.DataFrame(
        {
            "symbol": "NQU6",
            "trading_date": dates,
            "session_close_utc": dates.tz_localize("UTC") + pd.Timedelta(hours=21),
            "close": 100.0 * np.power(1.01, np.arange(25)),
        }
    )
    labels = build_forward_labels(panel)
    value = labels["forward_log_return_5s"].iloc[0]
    assert isinstance(value, (float, np.floating))
    assert np.isclose(value, 5.0 * np.log(1.01))
    assert labels.loc[0, "label_available_at_5s_utc"] == panel.loc[5, "session_close_utc"]
    assert labels["forward_log_return_20s"].tail(20).isna().all()


def test_walk_forward_folds_have_embargo_and_no_overlap() -> None:
    dates = pd.Series(pd.date_range("2022-01-01", periods=735, freq="B"))
    folds = build_walk_forward_folds(dates)
    assert len(folds) == 4
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["embargo_start"])).all()
    assert (pd.to_datetime(folds["embargo_end"]) < pd.to_datetime(folds["validation_start"])).all()
