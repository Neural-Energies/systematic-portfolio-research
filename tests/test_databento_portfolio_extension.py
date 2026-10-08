from pathlib import Path

import pandas as pd

from systematic_research.databento_portfolio_extension import (
    build_causal_daily_returns,
    cluster_balanced_inverse_vol_weights,
    filter_outrights,
    parse_outright_symbol,
    verify_manifest,
)


def test_parse_outright_symbol_rejects_spreads_and_user_defined() -> None:
    assert parse_outright_symbol("ZBZ5") == ("ZB", "Z", "5")
    assert parse_outright_symbol("SR3H26") == ("SR3", "H", "26")
    assert parse_outright_symbol("ZBZ5-ZBH6") is None
    assert parse_outright_symbol("UD:ZB: TL 123") is None
    assert parse_outright_symbol("MESZ5") is None


def test_filter_outrights_maps_supported_roots_only() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["ZBZ5", "ZBZ5-ZBH6", "RTYH6", "MESZ5"],
            "close": [1.0, 2.0, 3.0, 4.0],
        },
        index=pd.date_range("2026-01-01", periods=4, freq="min", tz="UTC", name="ts_event"),
    )
    result = filter_outrights(frame)
    assert result["symbol"].tolist() == ["ZBZ5", "RTYH6"]
    assert result["root_symbol"].tolist() == ["ZB", "RTY"]


def test_causal_contract_selection_uses_prior_day_volume_leader() -> None:
    days = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    frame = pd.DataFrame(
        {
            "root_symbol": ["ZB"] * 6,
            "symbol": ["ZBH6", "ZBM6"] * 3,
            "trading_date": days.repeat(2),
            "close": [100.0, 110.0, 101.0, 112.2, 102.01, 111.078],
            "volume": [1000, 500, 600, 1500, 500, 1600],
            "contract_return": [pd.NA, pd.NA, 0.01, 0.02, 0.01, -0.01],
            "minute_observations": [10] * 6,
        }
    )
    result = build_causal_daily_returns(frame)
    assert pd.isna(result.iloc[0]["selected_contract"])
    assert result.iloc[1]["selected_contract"] == "ZBH6"
    assert result.iloc[1]["contract_return"] == 0.01
    assert result.iloc[2]["selected_contract"] == "ZBM6"
    assert result.iloc[2]["contract_return"] == -0.01


def test_verify_manifest_detects_hash_match(tmp_path: Path) -> None:
    payload = tmp_path / "sample.bin"
    payload.write_bytes(b"institutional-data")
    import hashlib
    import json

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "filename": payload.name,
                        "hash": f"sha256:{digest}",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert verify_manifest(tmp_path)[0].valid


def test_cluster_balanced_weights_equalize_cluster_budgets() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    calibration = pd.DataFrame(
        {
            "ES": [0.01, -0.01, 0.02, -0.02, 0.01],
            "NQ": [0.02, -0.02, 0.04, -0.04, 0.02],
            "ZN": [0.001, -0.001, 0.002, -0.002, 0.001],
        },
        index=index,
    )
    weights = cluster_balanced_inverse_vol_weights(calibration)
    assert abs(weights[["ES", "NQ"]].sum() - 0.5) < 1e-12
    assert abs(weights["ZN"] - 0.5) < 1e-12
    assert weights["ES"] > weights["NQ"]
