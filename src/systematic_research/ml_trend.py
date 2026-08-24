"""Pooled walk-forward linear-regression trend system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

from systematic_research.trend import (
    construct_target_weights,
    simulate_with_drawdown_controls,
    summarize_folds,
)

ML_TREND_FEATURES = (
    "momentum_2s",
    "momentum_5s",
    "momentum_10s",
    "momentum_20s",
    "momentum_60s",
    "momentum_120s",
    "standardized_return_20s",
    "standardized_return_60s",
    "price_to_sma_20s",
    "price_to_sma_60s",
    "price_to_sma_120s",
    "ema_gap_5_20s",
    "ema_gap_10_60s",
    "trend_slope_20s",
    "trend_slope_60s",
    "trend_slope_120s",
    "trend_r_squared_20s",
    "trend_r_squared_60s",
    "trend_efficiency_20s",
    "trend_efficiency_60s",
)

SCREENED_UNIVERSE = ("CLU6",)
DROPPED_UNIVERSE = ("HGU6", "NQU6", "ZNU6", "ZTU6")


def make_model(alpha: float = 10.0) -> Pipeline:
    """Create a regularized linear model with fold-local preprocessing."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(ML_TREND_FEATURES)),
            (
                "symbol",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["symbol"],
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("regression", Ridge(alpha=alpha))])


def fit_walk_forward(
    dataset: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit expanding pooled models and return validation-only forecasts and coefficients."""
    required = {
        "symbol",
        "trading_date",
        "forward_log_return_5s",
        "label_available_at_5s_utc",
        "return_volatility_20s",
        *ML_TREND_FEATURES,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"missing model columns: {missing}")
    frame = dataset.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    label_date = pd.to_datetime(frame["label_available_at_5s_utc"], utc=True).dt.tz_localize(None)
    target_scale = frame["return_volatility_20s"].abs() * np.sqrt(5.0)
    frame["target"] = (frame["forward_log_return_5s"] / target_scale.replace(0.0, np.nan)).clip(
        -5.0, 5.0
    )
    forecasts: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    model_columns = [*ML_TREND_FEATURES, "symbol"]
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        train_end = pd.Timestamp(fold["train_end"])
        validation_start = pd.Timestamp(fold["validation_start"])
        validation_end = pd.Timestamp(fold["validation_end"])
        training = frame.loc[
            frame["trading_date"].le(train_end)
            & label_date.le(train_end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
            & frame["target"].notna()
        ]
        validation = frame.loc[frame["trading_date"].between(validation_start, validation_end)]
        if training.empty or validation.empty:
            raise ValueError(f"fold {fold['fold']} has empty training or validation data")
        model = make_model()
        model.fit(training[model_columns], training["target"])
        fold_forecast = validation[["trading_date", "symbol"]].copy()
        fold_forecast["forecast"] = model.predict(validation[model_columns])
        fold_forecast["fold"] = int(fold["fold"])
        forecasts.append(fold_forecast)
        names = model.named_steps["preprocessing"].get_feature_names_out()
        values = cast(Ridge, model.named_steps["regression"]).coef_
        coefficients.append(
            pd.DataFrame({"fold": int(fold["fold"]), "feature": names, "coefficient": values})
        )
    return pd.concat(forecasts, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def prediction_diagnostics(forecasts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Measure validation forecast direction and rank association by fold and instrument."""
    joined = forecasts.merge(
        labels[["symbol", "trading_date", "forward_log_return_5s"]],
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (fold, symbol), group in joined.groupby(["fold", "symbol"], observed=True):
        rows.append(
            {
                "fold": int(cast(Any, fold)),
                "symbol": str(symbol),
                "observations": len(group),
                "pearson_ic": float(group["forecast"].corr(group["forward_log_return_5s"])),
                "spearman_ic": float(
                    group["forecast"].corr(group["forward_log_return_5s"], method="spearman")
                ),
                "directional_accuracy": float(
                    (np.sign(group["forecast"]) == np.sign(group["forward_log_return_5s"])).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def run(project_root: Path) -> None:
    """Train and backtest the full development-only systematic trend portfolio."""
    features = pd.read_parquet(
        project_root / "data/processed/features/development_session_features.parquet"
    )
    labels = pd.read_parquet(
        project_root / "data/processed/labels/development_forward_labels.parquet"
    )
    panel = pd.read_parquet(
        project_root / "data/processed/research/development_session_panel.parquet"
    )
    features = features.loc[features["symbol"].isin(SCREENED_UNIVERSE)].copy()
    labels = labels.loc[labels["symbol"].isin(SCREENED_UNIVERSE)].copy()
    panel = panel.loc[panel["symbol"].isin(SCREENED_UNIVERSE)].copy()
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    dataset = features.merge(
        labels,
        on=["symbol", "trading_date"],
        how="inner",
        validate="one_to_one",
    )
    forecasts, coefficients = fit_walk_forward(dataset, folds)
    returns = panel.pivot(index="trading_date", columns="symbol", values="close_to_close_return")
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    weights = construct_target_weights(forecasts, returns, minimum_assets=1)
    net, implemented = simulate_with_drawdown_controls(returns, weights, cost_bps=3.5)
    summary = summarize_folds(net, implemented, folds)
    diagnostics = prediction_diagnostics(forecasts, labels)
    output = project_root / "data/processed/portfolios"
    output.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(output / "ml_trend_screened_validation_forecasts.parquet", index=False)
    coefficients.to_csv(output / "ml_trend_screened_fold_coefficients.csv", index=False)
    weights.to_parquet(output / "ml_trend_screened_target_weights.parquet")
    pd.DataFrame({"net_return": net}).to_parquet(
        output / "ml_trend_screened_portfolio_returns.parquet"
    )
    summary.to_csv(output / "ml_trend_screened_walk_forward_summary.csv", index=False)
    diagnostics.to_csv(output / "ml_trend_screened_prediction_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "symbol": [*SCREENED_UNIVERSE, *DROPPED_UNIVERSE],
            "decision": ["retain"] * len(SCREENED_UNIVERSE) + ["drop"] * len(DROPPED_UNIVERSE),
            "basis": [
                "positive association in all four validation folds and positive contribution",
                "forecast association did not survive portfolio contribution and fold stability",
                "unstable rank association and net contribution",
                "unstable directional accuracy and net contribution",
                "negative average rank association and net contribution",
            ],
        }
    ).to_csv(output / "ml_trend_universe_screen.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
