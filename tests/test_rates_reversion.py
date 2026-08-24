from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.rates_reversion import (
    RATE_LEADS,
    classification_diagnostics,
    fit_probability_models,
    simulate_hourly_weights,
)
from systematic_research.regression_displacement import REGRESSION_FEATURES


def test_rates_leads_are_exact_and_keep_cl_rejected() -> None:
    assert {(lead.symbol, lead.horizon_hours) for lead in RATE_LEADS} == {
        ("ZTU6", 23),
        ("ZNU6", 115),
        ("ZTU6", 115),
    }
    assert all(lead.symbol != "CLU6" for lead in RATE_LEADS)


def test_probability_models_emit_outer_validation_rows_only() -> None:
    dates = pd.date_range("2020-01-01", periods=70, freq="D", tz="UTC")
    rows: list[dict[str, object]] = []
    for symbol_offset, symbol in enumerate(("ZNU6", "ZTU6")):
        for index, timestamp in enumerate(dates):
            row: dict[str, object] = {
                "symbol": symbol,
                "trading_date": timestamp.tz_localize(None).normalize(),
                "hour_start_utc": timestamp,
                "feature_available_at_utc": timestamp + pd.Timedelta(minutes=59),
                "regression_residual_zscore_24h": 3.5 if index % 2 else -3.5,
                "prediction_interval_position_24h": 3.2 if index % 2 else -3.2,
                "realized_volatility_24h": 0.10,
                "target_before_stop_23h": float((index + symbol_offset) % 2),
                "target_before_stop_115h": float((index + symbol_offset) % 2),
                "label_available_at_23h_utc": timestamp + pd.Timedelta(hours=23),
                "label_available_at_115h_utc": timestamp + pd.Timedelta(hours=115),
            }
            row.update({feature: float(index + 1) for feature in REGRESSION_FEATURES})
            row["regression_residual_zscore_24h"] = 3.5 if index % 2 else -3.5
            row["prediction_interval_position_24h"] = 3.2 if index % 2 else -3.2
            rows.append(row)
    folds = pd.DataFrame(
        [
            {
                "fold": 1,
                "train_end": "2020-02-09",
                "validation_start": "2020-02-15",
                "validation_end": "2020-02-20",
            }
        ]
    )
    predictions, coefficients = fit_probability_models(pd.DataFrame(rows), folds)
    assert predictions["trading_date"].min() == pd.Timestamp("2020-02-15")
    assert predictions["trading_date"].max() == pd.Timestamp("2020-02-20")
    assert set(predictions["model_variant"]) == {"displacement_only", "full_state"}
    assert set(zip(predictions["symbol"], predictions["horizon_hours"], strict=True)) == {
        ("ZTU6", 23),
        ("ZNU6", 115),
        ("ZTU6", 115),
    }
    assert predictions["reversion_probability"].between(0.0, 1.0).all()
    assert not coefficients.empty

    predictions.loc[predictions.index[-1], "target_before_stop"] = np.nan
    diagnostics = classification_diagnostics(predictions)
    assert diagnostics["observations"].sum() == len(predictions) - 1


def test_hourly_simulation_delays_weights_and_charges_costs() -> None:
    index = pd.date_range("2024-01-02", periods=4, freq="h", tz="UTC")
    returns = pd.DataFrame({"ZTU6": [0.10, 0.10, 0.00, 0.00]}, index=index)
    target = pd.DataFrame({"ZTU6": [1.0, 0.0, 0.0, 0.0]}, index=index)

    free = simulate_hourly_weights(returns, target, cost_bps=0.0)
    costed = simulate_hourly_weights(returns, target, cost_bps=3.5)

    assert free.loc[index[0], "gross_return"] == 0.0
    assert np.isclose(float(cast(Any, free.loc[index[1], "gross_return"])), 0.10)
    assert costed["cost"].sum() > 0.0
    assert costed["net_return"].sum() < free["net_return"].sum()
