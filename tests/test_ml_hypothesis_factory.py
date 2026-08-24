import pandas as pd

from systematic_research.ml_hypothesis_factory import strategy_returns


def test_strategy_returns_uses_fixed_hurdle_and_top_four() -> None:
    date = pd.Timestamp("2025-01-02")
    predictions = pd.DataFrame(
        {
            "trading_date": [date] * 6,
            "symbol": ["A", "B", "C", "D", "E", "F"],
            "target_return": [0.001] * 6,
            "ensemble_prediction": [0.0010, 0.0009, 0.0008, 0.0007, 0.0006, 0.0001],
        }
    )
    portfolio, records = strategy_returns(
        predictions,
        prediction_column="ensemble_prediction",
        entry_hurdle_one_way_bps=1.0,
        one_way_cost_bps=3.5,
    )
    assert int(records["position"].abs().sum()) == 4
    assert records.loc[records["symbol"].eq("F"), "position"].item() == 0.0
    assert abs(portfolio.iloc[0] - 0.0003) < 1e-12
