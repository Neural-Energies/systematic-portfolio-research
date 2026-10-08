"""Explicit calendar boundaries for chronological development-only modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def purged_training_rows(
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    sessions: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    embargo_sessions: int = 5,
) -> NDArray[np.bool_]:
    """All earlier rows whose full labels finish before a cash-session embargo."""
    if starts.tz is None or ends.tz is None or sessions.tz is None or cutoff.tz is None:
        raise ValueError("Explicit timezone required")
    if len(starts) != len(ends) or embargo_sessions < 0:
        raise ValueError("Invalid label clock or embargo")
    prior = sessions[sessions < cutoff]
    if embargo_sessions and len(prior) < embargo_sessions:
        return np.zeros(len(starts), dtype=bool)
    boundary = prior[-embargo_sessions] if embargo_sessions else cutoff
    return np.asarray((starts < boundary) & (ends <= boundary), dtype=bool)


def require_before_lockout(index: pd.DatetimeIndex, lockout_start: pd.Timestamp) -> None:
    """Refuse a fit/evaluation input containing even one reserved timestamp."""
    if index.tz is None or lockout_start.tz is None:
        raise ValueError("Explicit timezone required")
    if len(index) and index.max() >= lockout_start:
        raise ValueError("Reserved lockout rows reached the component research input")
