from datetime import UTC, datetime

from systematic_research.free_market_data import parse_utc


def test_parse_utc_assumes_utc_for_naive_timestamp() -> None:
    assert parse_utc("2026-01-01 12:00:00") == datetime(2026, 1, 1, 12, tzinfo=UTC)


def test_parse_utc_preserves_offset_as_utc() -> None:
    assert parse_utc("2026-01-01T07:00:00-05:00") == datetime(2026, 1, 1, 12, tzinfo=UTC)
