"""Unit tests for H024 dependency-network relative value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from systematic_research.graph_laplacian_rv import (
    balanced_positions,
    build_population,
    network_weights,
    outlier_removal_table,
    residual_scores,
    session_panel,
)


def _session_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(next(iter(values.values()))), freq="B")
    return pd.DataFrame(values, index=index)


def test_network_weights_rows_are_stochastic_and_sparse() -> None:
    correlations = pd.DataFrame(
        [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = network_weights(correlations, max_neighbors=1)
    assert np.allclose(weights.sum(axis=1).to_numpy(dtype=float), 1.0)
    assert float(weights.to_numpy()[0, 0]) == 0.0
    top = weights.loc["A"].idxmax()
    assert top == "B"


def test_network_weights_isolated_node_falls_back_to_uniform() -> None:
    correlations = pd.DataFrame(
        [
            [1.0, -0.8, -0.7],
            [-0.8, 1.0, 0.5],
            [-0.7, 0.5, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    weights = network_weights(correlations, max_neighbors=4)
    assert np.allclose(weights.loc["A"].to_numpy(dtype=float), 1.0 / 3.0)


def test_residual_scores_do_not_use_future_information() -> None:
    rng = np.random.default_rng(11)
    frame = _session_frame(
        {symbol: rng.normal(0.0, 0.01, 120).tolist() for symbol in ("A", "B", "C", "D", "E", "F")}
    )
    extended = frame.copy()
    extended.iloc[100:] = extended.iloc[100:] + 0.05
    prefix_scores = residual_scores(frame, 40, 5, 3)
    extended_scores = residual_scores(extended, 40, 5, 3)
    unaffected = prefix_scores.iloc[:100].dropna(how="all")
    assert not unaffected.empty
    np.testing.assert_allclose(
        unaffected.to_numpy(dtype=float),
        extended_scores.loc[unaffected.index].to_numpy(dtype=float),
    )


def test_residual_scores_warmup_is_nan() -> None:
    rng = np.random.default_rng(5)
    frame = _session_frame(
        {symbol: rng.normal(0.0, 0.01, 30).tolist() for symbol in ("A", "B", "C")}
    )
    scores = residual_scores(frame, 20, 10, 2)
    assert scores.iloc[:10].isna().all().all()


def test_balanced_positions_are_dollar_neutral_and_causal() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="B")
    scores = pd.DataFrame(
        {
            "A": [-2.0, np.nan, -2.0, -2.0, -2.0, -2.0],
            "B": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "C": [-2.0, np.nan, -2.0, -2.0, -2.0, -2.0],
            "D": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            "E": [-2.5, np.nan, -2.5, -2.5, -2.5, -2.5],
            "F": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    positions = balanced_positions(scores, threshold=1.5, holding_sessions=1)
    active = positions.loc[index[1]]
    assert abs(float(active.sum())) < 1e-12
    assert float(active["A"]) < 0.0 and float(active["B"]) > 0.0
    assert abs(float(positions.loc[index[0]].sum())) < 1e-12
    perturbed = scores.copy()
    perturbed.iloc[0, 0] = 99.0
    after = balanced_positions(perturbed, threshold=1.5, holding_sessions=1)
    np.testing.assert_allclose(
        positions.iloc[2:].to_numpy(dtype=float), after.iloc[2:].to_numpy(dtype=float)
    )


def test_cost_accounting_matches_sides_times_cost() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    positions = pd.DataFrame({"A": [0.5, 0.5, -0.25]}, index=index)
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03]}, index=index)
    gross = positions.mul(returns).sum(axis=1)
    previous = positions.shift()
    previous.iloc[0] = 0.0
    sides = positions.sub(previous).abs().sum(axis=1)
    net = gross - sides * 1.0 / 10_000.0
    expected_gross = [0.005, -0.01, -0.0075]
    np.testing.assert_allclose(gross.to_numpy(), expected_gross)
    np.testing.assert_allclose(
        net.to_numpy(),
        [
            expected_gross[0] - 0.00005,
            expected_gross[1],
            expected_gross[2] - 0.000075,
        ],
    )


def test_outlier_removal_table_is_monotone_in_removed_share() -> None:
    values = [0.001] * 100 + [-0.0002] * 100
    values[0] = 0.05
    values[1] = 0.04
    values[2] = 0.03
    portfolio = pd.Series(values)
    table = outlier_removal_table(portfolio)
    shares = table["share_of_net_sum_removed"].to_numpy(dtype=float)
    assert shares[1] <= shares[2] <= shares[3] <= shares[4]
    assert table["best_days_removed"].tolist() == [0, 1, 3, 5, 10]
    assert float(shares[-1]) > 0.5
    assert float(table["total_return"].iloc[-1]) < float(table["total_return"].iloc[0])


def test_session_panel_shapes_from_minute_data() -> None:
    stamps = pd.date_range("2024-01-02 09:30", periods=4, freq="min", tz="UTC")
    minute_data = pd.DataFrame(
        {
            "symbol": ["ES"] * 4,
            "trading_date": [pd.Timestamp("2024-01-02").date()] * 4,
            "timestamp_utc": stamps,
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.1, 1.2, 1.3, 1.4],
            "low": [1.0, 1.05, 1.15, 1.25],
            "close": [1.1, 1.2, 1.3, 1.4],
            "volume": [10] * 4,
            "return_1m": [0.1] * 4,
            "gap_before_minutes": [0] * 4,
        }
    )
    returns, closes = session_panel(minute_data)
    assert returns.shape == (1, 1)
    first_return = float(returns.to_numpy(dtype=float)[0, 0])
    first_close = float(closes.to_numpy(dtype=float)[0, 0])
    assert pytest.approx(first_return, abs=1e-12) == 1.4 / 1.0 - 1.0
    assert first_close == 1.4


def test_build_population_is_deterministic_and_lagged() -> None:
    rng = np.random.default_rng(23)
    symbols = ("A", "B", "C", "D", "E", "F")
    common = rng.normal(0.0, 0.01, 260)
    frame = _session_frame(
        {symbol: (common * 0.6 + rng.normal(0.0, 0.008, 260)).tolist() for symbol in symbols}
    )
    variants, gross, sides = build_population(frame)
    assert len(variants) == 16
    assert list(gross.columns) == [variant.name for variant in variants]
    repeated_variants, repeated_gross, _ = build_population(frame)
    np.testing.assert_allclose(gross.to_numpy(), repeated_gross.to_numpy())
    active_days = gross.index[(sides.sum(axis=1) > 0.0)]
    if len(active_days):
        first_active_location = int(frame.index.get_indexer(pd.Index([active_days[0]]))[0])
        assert first_active_location >= 6
