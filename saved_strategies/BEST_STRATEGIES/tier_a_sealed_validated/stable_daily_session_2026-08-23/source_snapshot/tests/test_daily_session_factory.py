from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from systematic_research.daily_session_factory import (
    DailySessionSpec,
    build_parameter_ensembles,
    capped_inverse_volatility_weights,
    select_complementary,
    session_panels,
    simulate_population,
)
from systematic_research.daily_session_sealed import assert_sealed_evaluation_unused, sha256_file


def test_session_return_excludes_overnight_and_roll_gap() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["ES", "ES"],
            "trading_date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open": [100.0, 200.0],
            "high": [102.0, 204.0],
            "low": [99.0, 198.0],
            "close": [101.0, 202.0],
            "volume": [10, 20],
        }
    )
    returns = session_panels(frame)["session_return"]
    assert np.allclose(returns["ES"], [0.01, 0.01])


def test_signal_is_lagged_and_round_trip_cost_is_charged() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.DataFrame({"ES": [0.01, 0.02, -0.01]}, index=dates)
    score = pd.DataFrame({"ES": [2.0, -2.0, 2.0]}, index=dates)
    spec = DailySessionSpec("s", "ES", "session_momentum", 1, 1.0)
    net, gross, entries = simulate_population(
        [spec], returns, {("session_momentum", 1): score}, one_way_cost_bps=1.0
    )
    assert gross["s"].tolist() == [0.0, 0.02, 0.01]
    assert entries["s"].tolist() == [0.0, 1.0, 1.0]
    assert np.allclose(net["s"], [0.0, 0.0198, 0.0098])


def test_selection_enforces_symbol_and_correlation_limits() -> None:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    returns = pd.DataFrame(
        {
            "a": np.linspace(-0.01, 0.01, 30),
            "b": np.linspace(-0.01, 0.01, 30),
            "c": np.tile([-0.01, 0.01], 15),
            "d": np.sin(np.arange(30)) * 0.01,
        },
        index=dates,
    )
    summary = pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "symbol": ["ES", "CL", "ZN", "GC"],
            "build_annualized_return": 0.1,
            "selection_annualized_return": 0.1,
            "build_trades": 30,
            "selection_trades": 15,
            "robust_score": [4.0, 3.0, 2.0, 1.0],
        }
    )
    selected = select_complementary(
        summary,
        returns,
        np.ones(len(dates), dtype=bool),
        target=3,
        maximum_correlation=0.65,
    )
    assert selected == ["a", "c", "d"]


def test_parameter_ensembles_average_variants_within_symbol_family() -> None:
    returns = pd.DataFrame({"a": [0.01, 0.03], "b": [0.03, 0.01], "c": [0.02, 0.02]})
    summary = pd.DataFrame(
        {
            "name": ["a", "b", "c"],
            "symbol": ["ES", "ES", "CL"],
            "family": ["session_momentum", "session_momentum", "session_reversal"],
            "build_annualized_return": 0.1,
            "selection_annualized_return": 0.1,
            "build_trades": 30,
            "selection_trades": 10,
        }
    )
    ensembles, members = build_parameter_ensembles(summary, returns, minimum_sleeves=1)
    assert members == {"ES_session_momentum": ["a", "b"]}
    assert np.allclose(ensembles["ES_session_momentum"], [0.02, 0.02])


def test_inverse_volatility_weights_respect_concentration_cap() -> None:
    returns = pd.DataFrame(
        {
            "low_vol": np.tile([-0.001, 0.001], 20),
            "medium_vol": np.tile([-0.01, 0.01], 20),
            "high_vol": np.tile([-0.02, 0.02], 20),
        }
    )
    weights = capped_inverse_volatility_weights(returns, maximum_weight=0.50)
    assert np.isclose(weights.sum(), 1.0)
    assert weights.max() <= 0.50 + 1e-12


def test_sealed_evaluation_lock_refuses_a_second_attempt(tmp_path: Path) -> None:
    assert_sealed_evaluation_unused(tmp_path)
    (tmp_path / "SEALED_EVALUATION_LOCK.json").write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="already been attempted"):
        assert_sealed_evaluation_unused(tmp_path)


def test_file_hash_changes_with_content(tmp_path: Path) -> None:
    target = tmp_path / "input.bin"
    target.write_bytes(b"first")
    original = sha256_file(target)
    target.write_bytes(b"second")
    assert sha256_file(target) != original
