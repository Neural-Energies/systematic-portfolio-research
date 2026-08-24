from systematic_research.archive_daily_session_candidate import strategy_folder_name


def test_strategy_folder_name_is_readable_and_deterministic() -> None:
    spec = {
        "name": "daily_0184",
        "symbol": "CL",
        "family": "session_reversal",
        "lookback_sessions": 20,
        "entry_threshold": 1.5,
    }
    assert strategy_folder_name(1, spec) == "01_CL_daily_0184_session_reversal_lb20_threshold1.5"
