from systematic_research.pattern_discovery import bh_adjust, enumerate_candidates


def test_declared_grid_is_exhaustively_enumerated() -> None:
    settings = {
        "universe": ["A", "B"],
        "families": {
            "mean_reversion_zscore": {"windows": [8], "thresholds": [1.0, 2.0]},
            "trend_momentum": {"windows": [8], "thresholds": [0.0]},
            "breakout": {"windows": [8]},
            "lead_lag": {"lags": [1], "thresholds": [0.0, 1.0]},
            "cointegration": {"lookbacks": [20], "entry_zscores": [2.0]},
        },
    }
    assert len(enumerate_candidates(settings)) == 11


def test_fdr_uses_total_trial_count() -> None:
    adjusted = bh_adjust(__import__("pandas").Series([0.01, 0.02]), trials=100)
    assert adjusted.iloc[0] == 1.0
