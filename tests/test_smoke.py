from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from systematic_research.smoke import mean_return


def test_research_stack_smoke() -> None:
    expected = (0.01 + (100.5 / 101.0 - 1.0) + (102.0 / 100.5 - 1.0)) / 3.0
    assert math.isclose(mean_return(), expected, rel_tol=1e-12)


@given(st.floats(allow_nan=False, allow_infinity=False))
def test_identity_is_stable(value: float) -> None:
    assert value == value
