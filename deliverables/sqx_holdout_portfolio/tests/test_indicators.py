import numpy as np
import pandas as pd

from systematic_futures.indicators import keltner, tema


def test_keltner_definition() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    bars = pd.DataFrame(
        {"high": [3, 4, 5, 6], "low": [1, 2, 3, 4], "close": [2, 3, 4, 5]}, index=idx
    )
    upper, lower = keltner(bars, 2, 1.5)
    assert upper.iloc[-1] == 7.5
    assert lower.iloc[-1] == 1.5


def test_tema_constant_series_is_constant_after_warmup() -> None:
    values = pd.Series(np.ones(100))
    output = tema(values, 10).dropna()
    assert np.allclose(output, 1.0)
