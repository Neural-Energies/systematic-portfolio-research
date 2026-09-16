import pandas as pd

from systematic_research.instrument_mean_reversion import SleeveSpec, sleeve_returns, specifications


def test_grid_assigns_same_parameter_space_to_each_instrument() -> None:
    specs = specifications(["ES", "CL"])
    assert len(specs) == 144
    assert len([spec for spec in specs if spec.symbol == "ES"]) == 72


def test_position_is_lagged_before_return_is_earned() -> None:
    returns = pd.Series([0.0, 0.0, 0.10, 0.0, 0.0, -0.10])
    result = sleeve_returns(SleeveSpec("ES", "return_reversal", 2, 0.5, 1), returns, 0.0)
    assert result.iloc[2] == 0.0
