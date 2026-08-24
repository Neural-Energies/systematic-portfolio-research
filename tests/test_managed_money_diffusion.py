import pandas as pd

from systematic_research.managed_money_diffusion import build_position_variants


def _reports() -> pd.DataFrame:
    values = {"CL": -0.4, "GC": -0.2, "HG": 0.0, "NG": 0.2, "ZC": 0.4}
    return pd.DataFrame(
        {
            "symbol": list(values),
            "report_date": pd.Timestamp("2024-01-02"),
            "release_date": pd.Timestamp("2024-01-05"),
            "managed_money_net_share": list(values.values()),
        }
    )


def test_managed_money_signal_waits_then_follows_rank_for_five_sessions() -> None:
    dates = pd.bdate_range("2024-01-02", periods=10)
    panel = build_position_variants(_reports(), dates)[(5, 1.0)]
    assert panel.loc["2024-01-05"].abs().sum() == 0.0
    active = panel.loc["2024-01-08":"2024-01-12"]
    assert len(active) == 5
    assert (active["ZC"] == 1.0).all()
    assert (active["CL"] == -1.0).all()
    assert active[["GC", "HG", "NG"]].abs().sum().sum() == 0.0
