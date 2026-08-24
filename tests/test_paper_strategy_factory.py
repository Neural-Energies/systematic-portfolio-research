import pandas as pd

from systematic_research.paper_strategy_factory import (
    PAPER_HYPOTHESES,
    SECOND_WAVE_FAMILIES,
    generate_paper_population,
    select_development_candidates,
)


def test_paper_population_is_deterministic_and_uses_declared_hypotheses() -> None:
    first = generate_paper_population(["ES", "ZN"], per_symbol=16, seed=9)
    second = generate_paper_population(["ES", "ZN"], per_symbol=16, seed=9)
    assert first == second
    assert len(first) == 32
    assert {spec.signal_family for spec in first}.issubset(PAPER_HYPOTHESES)
    assert all(spec.direction == 1 for spec in first)


def test_balanced_population_includes_session_close_hypothesis() -> None:
    population = generate_paper_population(["ES"], per_symbol=len(PAPER_HYPOTHESES), seed=3)
    close_specs = [spec for spec in population if spec.signal_family == "intraday_close_momentum"]
    assert len(close_specs) == 1
    assert close_specs[0].time_rule == "preclose"


def test_second_wave_population_covers_every_new_family_and_timeframe() -> None:
    population = generate_paper_population(
        ["ES"],
        per_symbol=28,
        seed=4,
        families_override=SECOND_WAVE_FAMILIES,
    )
    assert {spec.signal_family for spec in population} == set(SECOND_WAVE_FAMILIES)
    assert {spec.timeframe_minutes for spec in population} == {15, 30, 60, 240}


def test_selection_never_needs_portfolio_test_columns() -> None:
    summary = pd.DataFrame(
        {
            "strategy": ["a", "b", "c"],
            "build_annualized_return": [0.1, -0.1, 0.2],
            "selection_annualized_return": [0.1, 0.2, -0.1],
            "build_sharpe": [1.0, -1.0, 0.5],
            "selection_sharpe": [1.0, 0.2, -0.5],
            "build_trades": [4, 4, 4],
            "selection_trades": [4, 4, 4],
        }
    )
    assert select_development_candidates(summary, target=2) == ["a", "c"]
