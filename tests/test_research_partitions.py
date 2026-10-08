import pandas as pd
import pytest

from systematic_research.research_partitions import purged_training_rows, require_before_lockout


def test_five_session_embargo_and_crossing_labels() -> None:
    sessions = pd.date_range(
        "2025-08-04", periods=14, freq="B", tz="America/New_York"
    ) + pd.Timedelta(hours=9, minutes=30)
    starts = sessions
    ends = starts + pd.Timedelta(hours=4)
    mask = purged_training_rows(
        starts, ends, sessions, pd.Timestamp("2025-08-21", tz="America/New_York")
    )
    assert starts[mask][-1].strftime("%Y-%m-%d") == "2025-08-13"
    crossing = ends.copy().to_numpy()
    crossing[0] = pd.Timestamp("2025-08-22", tz="America/New_York")
    assert not purged_training_rows(
        starts,
        pd.DatetimeIndex(crossing),
        sessions,
        pd.Timestamp("2025-08-21", tz="America/New_York"),
    )[0]


def test_exact_lockout_boundary_is_rejected() -> None:
    cutoff = pd.Timestamp("2025-08-21", tz="America/New_York")
    require_before_lockout(pd.DatetimeIndex([cutoff - pd.Timedelta(minutes=1)]), cutoff)
    with pytest.raises(ValueError, match="Reserved lockout"):
        require_before_lockout(pd.DatetimeIndex([cutoff]), cutoff)
