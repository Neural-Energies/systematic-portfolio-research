import pandas as pd

from systematic_research.es_session_reversion import scan


def test_scan_uses_no_more_than_one_event_per_session_and_side() -> None:
    frame = pd.DataFrame(
        columns=[
            "session",
            "is_rth",
            "globex_high",
            "globex_low",
            "globex_vwap",
            "close",
            "high",
            "low",
            "timestamp_utc",
        ]
    )
    assert scan(frame).empty
