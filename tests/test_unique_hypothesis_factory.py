import numpy as np
import pandas as pd

from systematic_research.unique_hypothesis_factory import (
    ECONOMIC_LEADERS,
    UniqueStrategySpec,
    _binary_entropy,
    _intraday_trade_sides,
    _rolling_series_zscore,
    extended_snapshot,
    walk_forward_select,
)


def test_economic_leaders_are_declared_and_never_self_referential() -> None:
    assert ECONOMIC_LEADERS
    assert all(target != leader for target, leader in ECONOMIC_LEADERS.items())


def test_extended_snapshot_reports_weekly_hit_rate() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    snapshot = extended_snapshot(pd.Series(0.001, index=dates))
    assert snapshot["win_weeks"] == 1.0
    assert snapshot["positive_weeks"] == snapshot["weeks"]


def test_rolling_series_zscore_accepts_unnamed_series() -> None:
    values = pd.Series(np.arange(80, dtype=float))
    scores = _rolling_series_zscore(values, history=40)
    assert scores.name is None
    assert scores.iloc[-1] > 0.0


def test_intraday_trade_sides_count_transitions_and_forced_session_close() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 2))
    position = pd.Series([1.0, 1.0, -1.0, 1.0, 1.0])
    sides = _intraday_trade_sides(position, dates)
    assert sides.tolist() == [1.0, 0.0, 3.0, 1.0, 1.0]
    assert sides.groupby(dates).sum().tolist() == [4.0, 2.0]


def test_binary_entropy_has_expected_edges_and_maximum() -> None:
    entropy = _binary_entropy(pd.Series([0.0, 0.5, 1.0]))
    assert entropy.tolist() == [0.0, 1.0, 0.0]


def test_walk_forward_selection_never_uses_current_fold() -> None:
    dates = pd.bdate_range("2024-01-01", periods=80)
    early_winner = np.r_[np.full(20, 0.002), np.full(60, -0.001)]
    future_winner = np.r_[np.full(20, -0.002), np.full(60, 0.003)]
    returns = pd.DataFrame({"early": early_winner, "future": future_winner}, index=dates)
    sides = pd.DataFrame(2.0, index=dates, columns=returns.columns)
    specs = [
        UniqueStrategySpec("early", "H100", "CL", 15, 1, 0.0, 1),
        UniqueStrategySpec("future", "H100", "NG", 15, 1, 0.0, 1),
        UniqueStrategySpec("other", "H101", "ES", 15, 1, 0.0, 1),
    ]
    returns["other"] = 0.001
    sides["other"] = 2.0
    _, _, selections = walk_forward_select(specs, returns, sides, folds=4, per_hypothesis=1)
    first_evaluation = selections.loc[
        selections["evaluation_fold"].eq(2) & selections["hypothesis_id"].eq("H100")
    ]
    assert "early" in first_evaluation["strategy"].tolist()
    assert "future" not in first_evaluation["strategy"].tolist()
