from __future__ import annotations

import math

import pandas as pd

from systematic_research.research_panel import build_minute_returns, build_session_panel


def _canonical_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["ESU6"] * 4,
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-08-20T21:59:00Z",
                    "2026-08-20T22:00:00Z",
                    "2026-08-20T22:01:00Z",
                    "2026-08-20T22:03:00Z",
                ],
                utc=True,
            ),
            "open": [100.0, 101.0, 102.0, 104.0],
            "high": [101.0, 102.0, 103.0, 105.0],
            "low": [99.0, 100.0, 101.0, 103.0],
            "close": [100.0, 101.0, 102.0, 104.0],
            "volume": [10, 20, 30, 40],
        }
    )


def test_returns_do_not_cross_sessions_or_data_gaps() -> None:
    result = build_minute_returns(_canonical_frame(), "America/New_York", 18)
    assert result["trading_date"].astype(str).tolist() == [
        "2026-08-20",
        "2026-08-21",
        "2026-08-21",
        "2026-08-21",
    ]
    assert math.isnan(result["return_1m"].iloc[0])
    assert math.isnan(result["return_1m"].iloc[1])
    assert math.isclose(result["return_1m"].iloc[2], 102.0 / 101.0 - 1.0)
    assert math.isnan(result["return_1m"].iloc[3])
    assert result["gap_before_minutes"].tolist() == [0, 0, 0, 1]


def test_session_panel_uses_observed_minutes_only() -> None:
    minute_returns = build_minute_returns(_canonical_frame(), "America/New_York", 18)
    sessions = build_session_panel(minute_returns)
    second_session = sessions.iloc[1]
    assert second_session["minute_observations"] == 3
    assert second_session["missing_minutes_inside_session"] == 1
    assert second_session["volume"] == 90
