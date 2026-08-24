from __future__ import annotations

import pandas as pd
import pytest

from systematic_research.data import validate_market_data


def test_market_data_is_normalized_to_utc_and_sorted() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2025-01-02T10:00:00-05:00", "2025-01-01T10:00:00-05:00"],
            "symbol": ["ES", "ES"],
            "close": [101.0, 100.0],
        }
    )
    result = validate_market_data(frame)
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result["close"].tolist() == [100.0, 101.0]


def test_market_data_rejects_naive_timestamps() -> None:
    frame = pd.DataFrame({"timestamp": ["2025-01-01"], "symbol": ["ES"], "close": [100.0]})
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_market_data(frame)
