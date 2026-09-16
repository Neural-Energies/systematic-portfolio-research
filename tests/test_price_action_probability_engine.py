import pandas as pd

from systematic_research.price_action_probability_engine import rth_session


def test_evening_bar_belongs_to_next_rth_session() -> None:
    stamps = pd.Series(pd.to_datetime(["2025-01-06 23:00:00+00:00", "2025-01-07 15:00:00+00:00"]))
    sessions = rth_session(stamps)
    assert sessions.iloc[0] == sessions.iloc[1]
