import numpy as np
import pandas as pd

from systematic_research.strategy_factory import (
    ProductStrategy,
    generate_population,
    make_research_windows,
    portfolio_candidates,
    select_one_hundred,
    simulate_product_strategy,
)


def test_population_is_balanced_unique_and_reproducible() -> None:
    first = generate_population(["B", "A"], per_symbol=20, seed=7)
    second = generate_population(["B", "A"], per_symbol=20, seed=7)
    assert first == second
    assert len(first) == 40
    assert sum(spec.symbol == "A" for spec in first) == 20
    assert len({spec.name for spec in first}) == 40


def test_research_windows_reserve_portfolio_test() -> None:
    dates = pd.bdate_range("2023-01-02", "2025-08-15")
    windows = make_research_windows(dates)
    assert windows.selection_end < windows.portfolio_test_start
    assert windows.portfolio_test_start == "2024-10-01"


def test_single_product_simulation_delays_entry() -> None:
    index = pd.date_range("2024-01-02 14:00Z", periods=4, freq="30min")
    score = pd.Series([2.0, 1.0, 0.0, 0.0], index=index)
    returns = pd.Series([0.01, 0.02, -0.01, 0.0], index=index)
    eligible = pd.Series(True, index=index)
    dates = pd.Series(pd.Timestamp("2024-01-02"), index=index)
    spec = ProductStrategy("x", "A", 30, "momentum", 2, 1, 1.5, 0.1, 2, "all", "all", 0.5, 0.5)
    gross, turnover, entries = simulate_product_strategy(score, returns, eligible, dates, spec)
    assert np.isclose(gross.iloc[0], 0.01)
    assert np.isclose(turnover.iloc[0], 2.0)
    assert entries.iloc[0] == 1.0


def test_selection_takes_best_one_hundred_positive_strategies() -> None:
    rows = []
    for symbol in ("A", "B", "C", "D", "E"):
        for index in range(25):
            rows.append(
                {
                    "strategy": f"{symbol}_{index}",
                    "symbol": symbol,
                    "build_annualized_return": 0.1,
                    "selection_annualized_return": 0.1,
                    "build_sharpe": 1.0 + index / 100,
                    "selection_sharpe": 1.0 + index / 100,
                    "build_trades": 20,
                    "selection_trades": 10,
                }
            )
    selected = select_one_hundred(pd.DataFrame(rows))
    assert len(selected) == 100
    assert len(set(selected)) == 100
    assert "A_24" in selected


def test_portfolio_is_fixed_capital_not_volatility_balanced() -> None:
    returns = pd.DataFrame({f"s{i}": [0.01, -0.01] for i in range(100)})
    summary = pd.DataFrame(
        {
            "strategy": returns.columns,
            "selection_annualized_return": np.arange(1, 101),
            "selection_sharpe": np.arange(1, 101),
        }
    )
    portfolios, weights = portfolio_candidates(returns, summary)
    assert np.isclose(weights["all_100_equal_capital"].sum(), 1.0)
    assert np.isclose(weights["top_25_equal_capital"].sum(), 1.0)
    assert "all_100_expectancy_weighted" in portfolios
