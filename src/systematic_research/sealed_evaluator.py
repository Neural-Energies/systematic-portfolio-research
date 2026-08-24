"""One-shot sequential evaluation of the frozen development candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from systematic_research.broad_strategy_search import build_signal
from systematic_research.features import build_session_features
from systematic_research.intraday_strategy_search import (
    IntradaySpec,
    _panels,
    build_score,
    combine_with_saved_baseline,
    complementary_portfolios,
    resample_bars,
    simulate,
)
from systematic_research.saved_strategy_runner import _spec_from_component, performance_snapshot
from systematic_research.trend import construct_target_weights, simulate_with_drawdown_controls


def assert_holdout_unused(candidate: dict[str, Any], output: Path) -> None:
    """Refuse evaluation unless the candidate is frozen and has never seen holdout."""
    policy = candidate["holdout_policy"]
    if not bool(policy["frozen"]):
        raise PermissionError("candidate must be frozen before sealed evaluation")
    if int(policy["holdout_run_count"]) != 0 or bool(candidate["candidate"]["holdout_accessed"]):
        raise PermissionError("sealed holdout has already been evaluated")
    if output.exists():
        raise PermissionError("sealed evaluation artifact already exists")


def _session_sleeve(
    development_panel: pd.DataFrame,
    holdout_panel: pd.DataFrame,
    baseline_candidate: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    combined_panel = pd.concat([development_panel, holdout_panel], ignore_index=True).sort_values(
        ["symbol", "trading_date"], ignore_index=True
    )
    features = build_session_features(combined_panel)
    market_returns = combined_panel.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    market_returns.index = pd.to_datetime(market_returns.index)
    market_returns = market_returns.sort_index()
    holdout_start = pd.Timestamp(holdout_panel["trading_date"].min())
    component_returns: dict[str, pd.Series] = {}
    for raw in baseline_candidate["components"]:
        spec = _spec_from_component(cast(dict[str, Any], raw))
        signal = build_signal(features, spec)
        stacked = cast(
            pd.Series,
            signal.rename_axis(index="trading_date", columns="symbol").stack(future_stack=True),
        )
        forecast = stacked.rename("forecast").reset_index()
        targets = construct_target_weights(forecast, market_returns, minimum_assets=3)
        development_net, _ = simulate_with_drawdown_controls(
            market_returns.loc[market_returns.index < holdout_start],
            targets.loc[targets.index < holdout_start],
            cost_bps=3.5,
        )
        holdout_net, _ = simulate_with_drawdown_controls(
            market_returns.loc[market_returns.index >= holdout_start],
            targets.loc[targets.index >= holdout_start],
            cost_bps=3.5,
        )
        component_returns[spec.name] = pd.concat([development_net, holdout_net])
    portfolio = pd.DataFrame(component_returns).mean(axis=1)
    return portfolio.loc[portfolio.index < holdout_start], portfolio.loc[
        portfolio.index >= holdout_start
    ]


def _load_minute_partitions(root: Path) -> pd.DataFrame:
    pieces = [pd.read_parquet(path) for path in sorted(root.glob("symbol=*/returns.parquet"))]
    pieces = [piece for piece in pieces if not piece.empty]
    if not pieces:
        raise ValueError(f"no minute data in {root}")
    return pd.concat(pieces, ignore_index=True)


def _intraday_sleeve(
    development_minutes: pd.DataFrame,
    holdout_minutes: pd.DataFrame,
    raw_specs: list[dict[str, Any]],
) -> tuple[pd.Series, pd.Series]:
    holdout_start = pd.Timestamp(holdout_minutes["trading_date"].min())
    combined = pd.concat([development_minutes, holdout_minutes], ignore_index=True)
    bars = resample_bars(combined, 240)
    close, returns, dates = _panels(bars)
    gross_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    for raw in raw_specs:
        spec = IntradaySpec(
            name=str(raw["name"]),
            theme="lead_lag",
            timeframe_minutes=240,
            lookback_bars=int(raw["lookback_bars"]),
            entry_threshold=float(raw["entry_threshold"]),
            exit_threshold=float(raw["exit_threshold"]),
            maximum_hold_bars=int(raw["maximum_hold_bars"]),
            volatility_filter=str(raw["volatility_filter"]),
            time_rule=str(raw["time_rule"]),
            stop_loss=None,
        )
        score = build_score(close, returns, spec.theme, spec.lookback_bars)
        development_rows = dates.lt(holdout_start)
        holdout_rows = ~development_rows
        development_gross, development_turnover = simulate(
            score.loc[development_rows],
            returns.loc[development_rows],
            dates.loc[development_rows],
            spec,
        )
        holdout_gross, holdout_turnover = simulate(
            score.loc[holdout_rows], returns.loc[holdout_rows], dates.loc[holdout_rows], spec
        )
        gross_map[spec.name] = pd.concat([development_gross, holdout_gross])
        turnover_map[spec.name] = pd.concat([development_turnover, holdout_turnover])
    gross = pd.DataFrame(gross_map).sort_index()
    turnover = pd.DataFrame(turnover_map).reindex(gross.index)
    portfolio_gross, portfolio_turnover = complementary_portfolios(gross, turnover)
    net = (
        portfolio_gross["causal_correlation_adjusted"]
        - portfolio_turnover["causal_correlation_adjusted"] * 3.5 / 10_000.0
    )
    return net.loc[net.index < holdout_start], net.loc[net.index >= holdout_start]


def run_once(project_root: Path) -> Path:
    """Open the sealed data once, evaluate, and permanently record the result."""
    candidate_path = project_root / "config/candidates/combined_session_intraday_2026-08-23.yaml"
    candidate = cast(dict[str, Any], yaml.safe_load(candidate_path.read_text(encoding="utf-8")))
    output = (
        project_root / "saved_strategies/combined_session_intraday_2026-08-23/sealed_result.json"
    )
    assert_holdout_unused(candidate, output)

    baseline_candidate = cast(
        dict[str, Any],
        yaml.safe_load(
            (project_root / candidate["session_sleeve"]["definition"]).read_text(encoding="utf-8")
        ),
    )
    session_root = project_root / "data/processed/databento_research"
    minute_root = project_root / "data/processed/research"
    development_panel = pd.read_parquet(session_root / "development_session_panel.parquet")
    holdout_panel = pd.read_parquet(session_root / "sealed_holdout_session_panel.parquet")
    development_minutes = _load_minute_partitions(minute_root / "development_minute_returns")
    holdout_minutes = _load_minute_partitions(minute_root / "sealed_holdout_minute_returns")

    session_development, session_holdout = _session_sleeve(
        development_panel, holdout_panel, baseline_candidate
    )
    intraday_development, intraday_holdout = _intraday_sleeve(
        development_minutes,
        holdout_minutes,
        cast(list[dict[str, Any]], candidate["intraday_sleeve"]["strategies"]),
    )
    development_combined, _ = combine_with_saved_baseline(
        session_development, intraday_development, lookbacks=(120,)
    )
    all_session = pd.concat([session_development, session_holdout])
    all_intraday = pd.concat([intraday_development, intraday_holdout])
    all_combined, all_weights = combine_with_saved_baseline(
        all_session, all_intraday, lookbacks=(120,)
    )
    frozen_name = "combined_causal_inverse_vol_120d"
    holdout_start = pd.Timestamp(holdout_panel["trading_date"].min())
    holdout_return = all_combined.loc[all_combined.index >= holdout_start, frozen_name]
    snapshot, periods = performance_snapshot(holdout_return)
    output.parent.mkdir(parents=True, exist_ok=False)
    all_combined.loc[all_combined.index >= holdout_start].to_parquet(
        output.parent / "sealed_portfolio_returns.parquet"
    )
    all_weights.loc[all_weights.index >= holdout_start].to_parquet(
        output.parent / "sealed_sleeve_weights.parquet"
    )
    periods.to_csv(output.parent / "sealed_calendar_period_returns.csv", index=False)
    record = {
        "candidate_id": candidate["candidate"]["id"],
        "evaluation_type": "single_sequential_sealed_holdout",
        "holdout_accessed": True,
        "holdout_run_count": 1,
        "frozen_strategy": frozen_name,
        "development_recalculation_observations": len(development_combined),
        "sealed_result": snapshot,
    }
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    candidate["candidate"]["status"] = "sealed_holdout_evaluated_once"
    candidate["candidate"]["holdout_accessed"] = True
    candidate["holdout_policy"]["holdout_run_count"] = 1
    candidate["holdout_policy"]["next_action"] = "no_further_holdout_selection_or_reruns"
    candidate["sealed_result_artifact"] = str(output.relative_to(project_root))
    candidate_path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    print(json.dumps(record, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen sealed evaluation exactly once")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run_once(arguments.project_root.resolve())


if __name__ == "__main__":
    main()
