import numpy as np
import pandas as pd

from systematic_research.eia_release_drift import (
    eligible_standard_release_dates,
    event_returns,
)


def test_release_calendar_excludes_holiday_disrupted_week() -> None:
    dates = pd.bdate_range("2024-01-08", "2024-01-19")
    cl_dates = eligible_standard_release_dates(dates, "CL")
    assert pd.Timestamp("2024-01-10") in cl_dates
    assert pd.Timestamp("2024-01-17") not in cl_dates


def test_response_and_target_windows_do_not_overlap() -> None:
    date = pd.Timestamp("2024-01-10")
    local_index = pd.date_range("2024-01-10 10:30", periods=45, freq="min", tz="America/New_York")
    minute_data = pd.DataFrame(
        {
            "symbol": "CL",
            "trading_date": date,
            "timestamp_utc": local_index.tz_convert("UTC"),
            "open": np.arange(100.0, 145.0),
            "close": np.arange(100.5, 145.5),
        }
    )
    _, _, evidence = event_returns(
        minute_data,
        symbol="CL",
        response_minutes=15,
        hold_minutes=30,
        threshold=0.5,
        daily_index=pd.DatetimeIndex([date]),
    )
    assert abs(evidence.loc[date, "response_return"] - (114.5 / 100.0 - 1.0)) < 1e-12
    assert abs(evidence.loc[date, "target_return"] - (144.5 / 115.0 - 1.0)) < 1e-12
