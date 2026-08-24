import numpy as np
import pandas as pd

from systematic_research.pca_residual_reversal import (
    _balanced_tail_weights,
    _causal_residual_scores,
    chained_selection,
)


def test_balanced_tail_weights_are_dollar_neutral() -> None:
    scores = pd.DataFrame([[2.0, 1.5, -1.3, -2.1, 0.1]], columns=list("ABCDE"))
    weights = _balanced_tail_weights(scores, threshold=1.0)
    assert abs(weights.iloc[0].sum()) < 1e-12
    assert abs(weights.iloc[0].abs().sum() - 1.0) < 1e-12
    assert weights.loc[0, "A"] < 0.0
    assert weights.loc[0, "E"] == 0.0


def test_causal_pca_scores_do_not_change_when_future_returns_change() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=25)
    index = pd.date_range("2024-01-02", periods=25 * 4, freq="15min")
    trading_dates = pd.Series(np.repeat(dates.to_numpy(), 4), index=index)
    returns = pd.DataFrame(rng.normal(0.0, 0.001, (len(index), 4)), index=index)
    baseline = _causal_residual_scores(returns, trading_dates, training_sessions=20)
    changed = returns.copy()
    changed.loc[trading_dates.eq(dates[-1])] = 100.0
    replay = _causal_residual_scores(changed, trading_dates, training_sessions=20)
    comparison_date = dates[-2]
    mask = trading_dates.eq(comparison_date)
    pd.testing.assert_frame_equal(baseline.loc[mask], replay.loc[mask])


def test_chained_selector_does_not_overfilter_negative_training_scores() -> None:
    dates = pd.bdate_range("2024-01-02", periods=80)
    returns = pd.DataFrame(
        {
            "weak": np.r_[np.full(20, -0.001), np.full(60, 0.001)],
            "weaker": np.full(80, -0.002),
        },
        index=dates,
    )
    sides = pd.DataFrame(2.0, index=dates, columns=returns.columns)
    portfolio, selections = chained_selection(returns, sides)
    assert not portfolio.empty
    assert "__NO_SELECTION__" not in selections["strategy"].tolist()
