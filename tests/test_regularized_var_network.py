import numpy as np
import pandas as pd

from systematic_research.regularized_var_network import _causal_var_forecasts


def test_var_forecasts_do_not_change_when_future_training_data_change() -> None:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-02", periods=25)
    index = pd.date_range("2024-01-02", periods=25 * 8, freq="15min")
    trading_dates = pd.Series(np.repeat(dates.to_numpy(), 8), index=index)
    features = pd.DataFrame(rng.normal(size=(len(index), 3)), index=index)
    targets = features.shift(1).mul(0.1) + pd.DataFrame(
        rng.normal(scale=0.1, size=(len(index), 3)), index=index
    )
    baseline = _causal_var_forecasts(
        features,
        targets,
        trading_dates,
        training_sessions=20,
        ridge_penalty=10.0,
    )
    changed_features = features.copy()
    changed_targets = targets.copy()
    future_mask = trading_dates.eq(dates[-1])
    changed_features.loc[future_mask] = 100.0
    changed_targets.loc[future_mask] = -100.0
    replay = _causal_var_forecasts(
        changed_features,
        changed_targets,
        trading_dates,
        training_sessions=20,
        ridge_penalty=10.0,
    )
    comparison_mask = trading_dates.eq(dates[-2])
    pd.testing.assert_frame_equal(baseline.loc[comparison_mask], replay.loc[comparison_mask])
