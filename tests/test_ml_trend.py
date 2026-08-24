import pandas as pd

from systematic_research.ml_trend import (
    DROPPED_UNIVERSE,
    ML_TREND_FEATURES,
    SCREENED_UNIVERSE,
    fit_walk_forward,
)


def test_feature_count_is_within_requested_range() -> None:
    assert 10 <= len(ML_TREND_FEATURES) <= 30
    assert len(set(ML_TREND_FEATURES)) == len(ML_TREND_FEATURES)


def test_screened_and_dropped_universes_do_not_overlap() -> None:
    assert set(SCREENED_UNIVERSE).isdisjoint(DROPPED_UNIVERSE)
    assert set(SCREENED_UNIVERSE) == {"CLU6"}


def test_walk_forward_predictions_are_validation_only() -> None:
    dates = pd.date_range("2020-01-01", periods=40)
    rows = []
    for symbol_offset, symbol in enumerate(("A", "B", "C")):
        for index, date in enumerate(dates):
            row = {
                "symbol": symbol,
                "trading_date": date,
                "forward_log_return_5s": 0.001 * (index + symbol_offset),
                "label_available_at_5s_utc": date + pd.Timedelta(days=5),
                "return_volatility_20s": 0.01,
            }
            row.update({feature: float(index + symbol_offset) for feature in ML_TREND_FEATURES})
            rows.append(row)
    folds = pd.DataFrame(
        [
            {
                "fold": 1,
                "train_end": "2020-01-20",
                "validation_start": "2020-01-26",
                "validation_end": "2020-01-30",
            }
        ]
    )
    forecasts, coefficients = fit_walk_forward(pd.DataFrame(rows), folds)
    assert len(forecasts) == 15
    assert forecasts["trading_date"].min() == pd.Timestamp("2020-01-26")
    assert forecasts["trading_date"].max() == pd.Timestamp("2020-01-30")
    assert not coefficients.empty
