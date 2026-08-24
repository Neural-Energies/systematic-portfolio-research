from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from systematic_research.regimes import (
    REGIMES,
    build_observable_regime_probabilities,
    combine_regime_probabilities,
)

OBSERVABLE_COLUMNS = [
    "standardized_return_20s",
    "trend_efficiency_20s",
    "trend_efficiency_60s",
    "ema_gap_5_20s",
    "ema_gap_10_60s",
    "realized_volatility_20s",
    "atr_14s_fraction",
    "return_volatility_20s",
    "volatility_ratio_5_20s",
    "range_zscore_20s",
    "range_percentile_60s",
    "true_range_fraction",
    "return_autocorrelation_20s",
    "standardized_return_5s",
    "volume_zscore_20s",
    "volume_per_observed_minute",
]


def _inputs(days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-01", periods=days, freq="B")
    feature_rows = []
    panel_rows = []
    for symbol_index, symbol in enumerate(("NQU6", "CLU6")):
        for day, date in enumerate(dates):
            wave = np.sin(day / 20.0 + symbol_index)
            row: dict[str, object] = {
                "symbol": symbol,
                "trading_date": date,
                "feature_available_at_utc": date.tz_localize("UTC"),
                "model_row_complete": day >= 120,
            }
            row.update(
                {
                    column: wave + column_index * 0.01
                    for column_index, column in enumerate(OBSERVABLE_COLUMNS)
                }
            )
            feature_rows.append(row)
            panel_rows.append(
                {
                    "symbol": symbol,
                    "trading_date": date,
                    "close_to_close_return": 0.01 * np.sin(day / 7.0 + symbol_index),
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(panel_rows)


def test_observable_regime_probabilities_sum_to_one() -> None:
    features, panel = _inputs(160)
    probabilities = build_observable_regime_probabilities(features, panel)
    mature = probabilities[list(REGIMES)].dropna()
    np.testing.assert_allclose(mature.sum(axis=1), 1.0)
    unavailable = probabilities[list(REGIMES)].isna().all(axis=1)
    assert probabilities.loc[unavailable, "observable_confidence"].isna().all()
    assert probabilities["observable_confidence"].dropna().between(0.0, 1.0).all()


def test_observable_regimes_are_prefix_invariant() -> None:
    short_features, short_panel = _inputs(150)
    long_features, long_panel = _inputs(170)
    short = build_observable_regime_probabilities(short_features, short_panel)
    long = build_observable_regime_probabilities(long_features, long_panel).iloc[:150]
    assert_frame_equal(short.reset_index(drop=True), long.reset_index(drop=True))


def test_markov_component_cannot_dominate_ensemble() -> None:
    date = pd.Timestamp("2025-01-01")
    observable = pd.DataFrame(
        {
            "trading_date": [date],
            **{regime: [1.0 if index == 0 else 0.0] for index, regime in enumerate(REGIMES)},
            "observable_confidence": [1.0],
            "correlation_concentration": [0.5],
        }
    )
    markov = pd.DataFrame(
        {
            "trading_date": [date],
            **{regime: [1.0 if index == 3 else 0.0] for index, regime in enumerate(REGIMES)},
            "markov_confidence": [1.0],
            "fold": [1],
        }
    )
    combined = combine_regime_probabilities(observable, markov, markov_weight=0.25)
    assert combined.loc[0, "ensemble_low_volatility_trend"] == 0.75
    assert combined.loc[0, "ensemble_stress_transition"] == 0.25
