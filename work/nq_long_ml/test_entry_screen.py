"""Frozen rule catalog, completed-data causality and session accounting."""

import numpy as np
import pandas as pd
from entry_screen import build_signals, fdr, session_metadata
from experiment import aggregate


def fixture():
    index = pd.date_range("2024-08-25 22:00", periods=5 * 1440, freq="min", tz="UTC")
    rng = np.random.default_rng(12)
    price = 10000 + np.cumsum(rng.normal(0, 0.5, len(index)))
    return pd.DataFrame(
        {
            "open": price,
            "close": price + 0.1,
            "high": price + 1,
            "low": price - 1,
            "volume": rng.integers(1, 100, len(index)),
        },
        index=index,
    )


def test_catalog_is_100_rules_and_signals_are_causal():
    minute = fixture()
    bars = aggregate(minute, 5)
    full, _, registry = build_signals(bars, bars)
    assert len(registry) == 100
    assert registry.family.nunique() == 20
    assert registry.signal.is_unique
    for date in ("2024-08-28 13:40", "2024-08-28 14:00", "2024-08-28 20:00"):
        cutoff = pd.Timestamp(date, tz="UTC")
        prefix = aggregate(minute.loc[minute.index < cutoff], 5)
        truncated, _, _ = build_signals(prefix, prefix)
        pd.testing.assert_frame_equal(full.loc[truncated.index], truncated)


def test_session_stops_and_breaks_follow_user_clock():
    clock = pd.DatetimeIndex(
        [
            "2024-08-26 13:30",
            "2024-08-26 18:00",
            "2024-08-26 20:00",
            "2024-08-26 22:00",
            "2024-08-27 13:25",
        ],
        tz="UTC",
    )
    meta = session_metadata(clock)
    assert list(meta.kind) == ["RTH", "RTH", "break", "Globex", "Globex"]
    assert meta.iloc[1].session_end == pd.Timestamp("2024-08-26 20:00", tz="UTC")
    assert meta.iloc[3].session_end == pd.Timestamp("2024-08-27 13:30", tz="UTC")
    assert meta.iloc[2].anchor != meta.iloc[3].anchor


def test_fdr_has_known_independent_reference_values():
    result = fdr([0.01, 0.04, 0.03, 0.2])
    assert np.allclose(result, [0.04, 0.053333333333, 0.053333333333, 0.2])
