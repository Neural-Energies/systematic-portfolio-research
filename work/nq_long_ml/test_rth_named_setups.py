import numpy as np
import pandas as pd
from rth_named_setups import build_setups


def test_named_rules_and_developing_profiles_do_not_see_future():
    clocks = [
        pd.date_range(
            f"2024-03-0{day} 09:30", periods=390, freq="min", tz="America/New_York"
        ).tz_convert("UTC")
        for day in (4, 5, 6)
    ]
    clock = clocks[0].append(clocks[1:])
    rng = np.random.default_rng(71)
    price = 18000 + np.cumsum(rng.normal(0, 0.7, len(clock)))
    minute = pd.DataFrame(
        {
            "open": price,
            "close": price + rng.normal(0, 0.2, len(clock)),
            "high": price + 1,
            "low": price - 1,
            "volume": rng.integers(10, 100, len(clock)),
        },
        index=clock,
    )
    full, metadata, registry, _ = build_setups(minute)
    cutoff = pd.Timestamp("2024-03-06 10:20", tz="America/New_York").tz_convert("UTC")
    prefix, _, _, _ = build_setups(minute.loc[minute.index < cutoff])
    pd.testing.assert_frame_equal(full.loc[prefix.index], prefix)
    assert len(registry) == 61
    assert full.columns.is_unique
    assert metadata.kind.eq("RTH").all()


def test_value_acceptance_requires_two_prior_bars_in_current_session():
    clocks = [
        pd.date_range(
            f"2024-03-0{day} 09:30", periods=390, freq="min", tz="America/New_York"
        ).tz_convert("UTC")
        for day in (4, 5, 6)
    ]
    frames = []
    for k, clock in enumerate(clocks):
        price = np.full(390, 110.0 if k == 0 else 140.0)
        volume = np.full(390, 100)
        high = price + (10 if k == 0 else 2)
        low = price - (10 if k == 0 else 2)
        if k == 1:
            price[-15:-10] = 99
            price[-10:] = 101
            volume[-15:] = 1
            high[-15:] = price[-15:] + 1
            low[-15:] = price[-15:] - 1
        frames.append(
            pd.DataFrame(
                {"open": price - 1, "close": price, "high": high, "low": low, "volume": volume},
                index=clock,
            )
        )
    signals, _, _, _ = build_setups(pd.concat(frames))
    first = pd.Timestamp("2024-03-06 09:35", tz="America/New_York").tz_convert("UTC")
    assert not signals.loc[first, signals.columns.str.startswith("vp_value_acceptance")].any()


def test_closed_rth_quote_does_not_replace_prior_session_profile():
    a = pd.date_range(
        "2025-01-08 09:30", periods=390, freq="min", tz="America/New_York"
    ).tz_convert("UTC")
    b = pd.DatetimeIndex(
        [pd.Timestamp("2025-01-09 09:30", tz="America/New_York").tz_convert("UTC")]
    )
    c = pd.date_range(
        "2025-01-10 09:30", periods=390, freq="min", tz="America/New_York"
    ).tz_convert("UTC")
    clock = a.append([b, c])
    price = np.r_[np.full(len(a), 100.0), [10000.0], np.full(len(c), 101.0)]
    minute = pd.DataFrame(
        {"open": price, "close": price, "high": price + 1, "low": price - 1, "volume": 100},
        index=clock,
    )
    _, _, _, levels = build_setups(minute)
    selected = levels.loc[
        (levels.index.date == pd.Timestamp("2025-01-10").date()) & levels.name.eq("poc_1pt_close"),
        "level",
    ]
    assert len(selected) > 0
    assert selected.eq(100.5).all()
