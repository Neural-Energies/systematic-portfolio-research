from typing import Any, cast

import numpy as np
import pandas as pd

from systematic_research.mean_reversion import (
    StrategySpec,
    build_signal,
    strategy_specs,
)


def test_registry_contains_fifty_unique_prespecified_rules() -> None:
    specs = strategy_specs()
    assert len(specs) == 50
    assert len({spec.name for spec in specs}) == 50
    assert len({spec.family for spec in specs}) == 5


def test_signal_is_contrarian_and_bounded() -> None:
    frame = pd.DataFrame(
        {
            "trading_date": pd.date_range("2020-01-01", periods=70),
            "symbol": "A",
            "momentum_5s": np.arange(1.0, 71.0),
        }
    )
    signal = build_signal(frame, StrategySpec("x", "x", "momentum_5s"))
    assert float(cast(Any, signal.iloc[-1, 0])) < 0.0
    assert signal.abs().max().max() <= 1.0
