"""Run a saved fixed strategy definition without reselecting its components."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from systematic_research.broad_strategy_search import StrategySpec, build_signal
from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.trend import construct_target_weights, simulate_with_drawdown_controls


def _compound(returns: pd.Series) -> float:
    return float(cast(float, (1.0 + returns.astype(float)).prod()) - 1.0)


def performance_snapshot(
    returns: pd.Series,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Calculate comparable daily and calendar-period performance statistics."""
    sample = returns.dropna().astype(float).sort_index()
    if sample.empty:
        raise ValueError("portfolio return series is empty")
    sample.index = pd.to_datetime(sample.index)
    monthly = cast(pd.Series, sample.resample("ME").apply(_compound))[
        sample.resample("ME").count() > 0
    ]
    quarterly = cast(pd.Series, sample.resample("QE").apply(_compound))[
        sample.resample("QE").count() > 0
    ]
    yearly = cast(pd.Series, sample.resample("YE").apply(_compound))[
        sample.resample("YE").count() > 0
    ]
    wealth = (1.0 + sample).cumprod()
    snapshot: dict[str, float | int | str] = {
        "observations": len(sample),
        "start": sample.index.min().date().isoformat(),
        "end": sample.index.max().date().isoformat(),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "annualized_return": float(sample.mean() * 252.0),
        "annualized_volatility": annualized_volatility(sample),
        "sharpe": sharpe_ratio(sample),
        "maximum_drawdown": maximum_drawdown(sample),
        "win_days": float((sample > 0.0).mean()),
        "win_months": float((monthly > 0.0).mean()),
        "win_quarters": float((quarterly > 0.0).mean()),
        "win_years": float((yearly > 0.0).mean()),
        "positive_days": int((sample > 0.0).sum()),
        "positive_months": int((monthly > 0.0).sum()),
        "positive_quarters": int((quarterly > 0.0).sum()),
        "positive_years": int((yearly > 0.0).sum()),
        "months": len(monthly),
        "quarters": len(quarterly),
        "years": len(yearly),
    }
    periods: pd.DataFrame = (
        pd.concat(
            [
                monthly.rename("return").to_frame().assign(period_type="month"),
                quarterly.rename("return").to_frame().assign(period_type="quarter"),
                yearly.rename("return").to_frame().assign(period_type="year"),
            ]
        )
        .rename_axis("period_end")
        .reset_index()
    )
    return snapshot, periods


def _spec_from_component(component: dict[str, Any]) -> StrategySpec:
    raw_direction = component.get("direction_value", component.get("direction", 1.0))
    direction = float(raw_direction) if not isinstance(raw_direction, str) else 1.0
    if raw_direction in {"contrarian", "cross_sectional_contrarian"}:
        direction = -1.0
    return StrategySpec(
        name=str(component["strategy"]),
        theme=str(component["theme"]),
        feature=str(component["feature"]),
        direction=direction,
        hold_sessions=int(component["hold_sessions"]),
        cross_sectional=bool(component.get("cross_sectional", False)),
        secondary_feature=component.get("secondary_feature"),
        secondary_mode=component.get("secondary_mode"),
    )


def run_saved_strategy(
    candidate_path: Path,
    features_path: Path,
    panel_path: Path,
    output_root: Path,
    *,
    folds_path: Path | None = None,
) -> Path:
    """Execute the exact saved components and write a dated comparison run."""
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(features_path)
    panel = pd.read_parquet(panel_path)
    market_returns = panel.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    )
    market_returns.index = pd.to_datetime(market_returns.index)
    market_returns = market_returns.sort_index()
    cost_bps = float(candidate["execution"]["one_way_cost_bps"])

    component_returns: dict[str, pd.Series] = {}
    component_specs: list[StrategySpec] = []
    for raw_component in candidate["components"]:
        component = dict(raw_component)
        spec = _spec_from_component(component)
        component_specs.append(spec)
        signal = build_signal(features, spec)
        stacked = signal.rename_axis(index="trading_date", columns="symbol").stack(
            future_stack=True
        )
        forecast = pd.Series(stacked, name="forecast").reset_index()
        target = construct_target_weights(forecast, market_returns, minimum_assets=3)
        net, _ = simulate_with_drawdown_controls(market_returns, target, cost_bps=cost_bps)
        component_returns[spec.name] = net
    components = pd.DataFrame(component_returns, index=market_returns.index)
    portfolio = components.mean(axis=1).rename("net_return")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = output_root / "runs" / run_id
    destination.mkdir(parents=True, exist_ok=False)
    components.to_parquet(destination / "component_returns.parquet")
    portfolio.to_frame().to_parquet(destination / "portfolio_returns.parquet")
    full_snapshot, full_periods = performance_snapshot(portfolio)
    full_periods.to_csv(destination / "calendar_period_returns.csv", index=False)

    validation_snapshot: dict[str, float | int | str] | None = None
    if folds_path is not None:
        folds = pd.read_csv(folds_path)
        mask = pd.Series(False, index=portfolio.index)
        for fold in folds.to_dict("records"):
            mask.loc[
                pd.Timestamp(fold["validation_start"]) : pd.Timestamp(fold["validation_end"])
            ] = True
        validation_snapshot, validation_periods = performance_snapshot(portfolio.loc[mask])
        validation_periods.to_csv(
            destination / "validation_calendar_period_returns.csv", index=False
        )

    run_record = {
        "candidate_id": candidate["candidate"]["id"],
        "run_id": run_id,
        "candidate_file": str(candidate_path),
        "features_file": str(features_path),
        "panel_file": str(panel_path),
        "holdout_accessed": False,
        "components": [asdict(spec) for spec in component_specs],
        "execution": candidate["execution"],
        "full_available_history": full_snapshot,
        "saved_validation_windows": validation_snapshot,
    }
    (destination / "metrics.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed saved portfolio strategy.")
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("config/candidates/theme_champions_baseline_2026-08-23.yaml"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/databento_features/development_session_features.parquet"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/databento_research/development_session_panel.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("saved_strategies/theme_champions_baseline_2026-08-23"),
    )
    parser.add_argument(
        "--folds", type=Path, default=Path("outputs/development_walk_forward_folds.csv")
    )
    arguments = parser.parse_args()
    destination = run_saved_strategy(
        arguments.candidate,
        arguments.features,
        arguments.panel,
        arguments.output,
        folds_path=arguments.folds,
    )
    print(f"Saved strategy run: {destination}")


if __name__ == "__main__":
    main()
