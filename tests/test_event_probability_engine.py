import pandas as pd

from systematic_research.event_probability_engine import rolling_breakouts, vwap_stretches


def test_event_builders_return_named_directional_events() -> None:
    bars = pd.DataFrame(
        {
            "high": range(30),
            "low": range(30),
            "close": range(30),
            "session_vwap": range(30),
            "range_atr": [1.0] * 30,
        }
    )
    events = rolling_breakouts(bars) + vwap_stretches(bars)
    assert {event.direction for event in events} == {-1, 1}
    assert len(events) == 4
