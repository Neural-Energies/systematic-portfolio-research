import numpy as np
import pandas as pd

from systematic_research.clock_excursion import excursion_summary, same_clock_history


def test_iqr_only_trims_reference_distribution() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
    summary = excursion_summary(x, minimum=5)
    assert summary["removed_outliers"] == 1
    assert summary["mean"] == 4.5
    assert summary["median"] == 4.5
    assert summary["mean70"] == 3.15
    assert np.isclose(summary["q70"], 5.9)


def test_previous_twenty_sessions_not_twenty_qualified_bars() -> None:
    days = pd.date_range("2024-01-01", periods=25, freq="B", tz="UTC")
    index = days + pd.Timedelta(hours=15)
    frame = pd.DataFrame(
        {
            "anchor": days,
            "slot": 600,
            "excursion": np.arange(25, dtype=float),
            "opening_up": np.arange(25) % 2 == 0,
            "scorable": True,
        },
        index=index,
    )
    stats = same_clock_history(frame, days)
    expected = np.arange(4, 24, dtype=float)[np.arange(4, 24) % 2 == 0]
    assert stats.loc[index[24], "mean"] == expected.mean()
    assert stats.loc[index[24], "qualified_samples"] == 10
    assert stats.iloc[:20]["mean"].isna().all()
    frame.loc[index[24], "excursion"] = 99999
    changed = same_clock_history(frame, days)
    pd.testing.assert_frame_equal(stats, changed)


def test_missing_sessions_are_not_backfilled_with_older_samples() -> None:
    days = pd.date_range("2024-01-01", periods=25, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {"anchor": days, "slot": 600, "excursion": 10.0, "opening_up": True, "scorable": True},
        index=days + pd.Timedelta(hours=15),
    ).drop(days[10] + pd.Timedelta(hours=15))
    stats = same_clock_history(frame, days)
    assert stats.iloc[-1].qualified_samples == 19
