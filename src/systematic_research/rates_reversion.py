"""Post-selection development retest for the observed ZN/ZT reversion leads."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio
from systematic_research.regression_displacement import REGRESSION_FEATURES

EXTREME_DISPLACEMENT = 3.0
ENTRY_PROBABILITY = 0.55
TARGET_VOLATILITY = 0.10
MAXIMUM_ASSET_WEIGHT = 0.25
IMPLEMENTATION_COST_BPS = 3.5

BASELINE_FEATURES = (
    "absolute_regression_zscore_24h",
    "absolute_prediction_interval_position_24h",
)
FULL_STATE_FEATURES = (*REGRESSION_FEATURES, *BASELINE_FEATURES)


@dataclass(frozen=True)
class RateLead:
    """One observed rates lead retained for a controlled development retest."""

    symbol: str
    horizon_hours: int

    @property
    def name(self) -> str:
        return f"{self.symbol}_{self.horizon_hours}h"


RATE_LEADS = (
    RateLead("ZTU6", 23),
    RateLead("ZNU6", 115),
    RateLead("ZTU6", 115),
)
REJECTED_MEAN_REVERSION_INSTRUMENTS = ("CLU6",)


def _add_displacement_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["absolute_regression_zscore_24h"] = frame["regression_residual_zscore_24h"].abs()
    frame["absolute_prediction_interval_position_24h"] = frame[
        "prediction_interval_position_24h"
    ].abs()
    return frame


def make_probability_model(feature_columns: tuple[str, ...]) -> Pipeline:
    """Create a fixed regularized logistic model with fold-local preprocessing."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(feature_columns)),
            (
                "symbol",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["symbol"],
            ),
        ]
    )
    classifier = LogisticRegression(C=1.0, max_iter=1_000, random_state=42)
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def fit_probability_models(
    dataset: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit training-only baseline/full models and return outer validation probabilities."""
    required = {
        "symbol",
        "trading_date",
        "hour_start_utc",
        "feature_available_at_utc",
        "regression_displacement_24h",
        "regression_residual_zscore_24h",
        "prediction_interval_position_24h",
        "realized_volatility_24h",
        *REGRESSION_FEATURES,
    }
    for lead in RATE_LEADS:
        required.update(
            {
                f"target_before_stop_{lead.horizon_hours}h",
                f"label_available_at_{lead.horizon_hours}h_utc",
            }
        )
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"missing rates-reversion columns: {missing}")
    frame = _add_displacement_features(dataset)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    probabilities: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    variants = {
        "displacement_only": BASELINE_FEATURES,
        "full_state": FULL_STATE_FEATURES,
    }
    for lead in RATE_LEADS:
        lead_frame = frame.loc[frame["symbol"].eq(lead.symbol)].copy()
        target_column = f"target_before_stop_{lead.horizon_hours}h"
        availability_column = f"label_available_at_{lead.horizon_hours}h_utc"
        target = pd.to_numeric(lead_frame[target_column], errors="coerce")
        label_available = pd.to_datetime(lead_frame[availability_column], utc=True)
        extreme = lead_frame["absolute_regression_zscore_24h"].ge(EXTREME_DISPLACEMENT)
        for raw_fold in folds.to_dict("records"):
            fold = cast(dict[str, Any], raw_fold)
            train_end = pd.Timestamp(str(fold["train_end"]))
            cutoff = (train_end + pd.Timedelta(days=1)).tz_localize("UTC")
            validation_start = pd.Timestamp(str(fold["validation_start"]))
            validation_end = pd.Timestamp(str(fold["validation_end"]))
            training_mask = (
                lead_frame["trading_date"].le(train_end)
                & label_available.lt(cutoff)
                & target.notna()
                & extreme
            )
            validation_mask = (
                lead_frame["trading_date"].between(validation_start, validation_end) & extreme
            )
            training = lead_frame.loc[training_mask]
            validation = lead_frame.loc[validation_mask]
            training_target = target.loc[training_mask].astype(int)
            if training.empty or validation.empty:
                raise ValueError(f"{lead.name} fold {fold['fold']} has no extreme observations")
            if training_target.nunique() < 2:
                raise ValueError(f"{lead.name} fold {fold['fold']} training target has one class")
            for variant, feature_columns in variants.items():
                model = make_probability_model(feature_columns)
                model_columns = [*feature_columns, "symbol"]
                model.fit(training[model_columns], training_target)
                prediction = validation[
                    [
                        "symbol",
                        "trading_date",
                        "hour_start_utc",
                        "feature_available_at_utc",
                        "regression_displacement_24h",
                        "regression_residual_zscore_24h",
                        "realized_volatility_24h",
                    ]
                ].copy()
                prediction["reversion_probability"] = model.predict_proba(
                    validation[model_columns]
                )[:, 1]
                prediction["target_before_stop"] = target.loc[validation_mask].to_numpy()
                prediction["fold"] = int(str(fold["fold"]))
                prediction["horizon_hours"] = lead.horizon_hours
                prediction["lead"] = lead.name
                prediction["model_variant"] = variant
                probabilities.append(prediction)
                names = model.named_steps["preprocessing"].get_feature_names_out()
                values = cast(LogisticRegression, model.named_steps["classifier"]).coef_[0]
                coefficients.append(
                    pd.DataFrame(
                        {
                            "fold": int(str(fold["fold"])),
                            "lead": lead.name,
                            "model_variant": variant,
                            "feature": names,
                            "coefficient": values,
                        }
                    )
                )
    return pd.concat(probabilities, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def classification_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare probability accuracy and calibration by lead, fold, and model variant."""
    rows: list[dict[str, Any]] = []
    grouping = predictions.groupby(["lead", "fold", "model_variant"], observed=True)
    for (lead, fold, variant), group in grouping:
        group = group.loc[group["target_before_stop"].notna()]
        if group.empty:
            continue
        actual = group["target_before_stop"].astype(int)
        probability = group["reversion_probability"].clip(1e-6, 1.0 - 1e-6)
        area = float(roc_auc_score(actual, probability)) if actual.nunique() > 1 else np.nan
        rows.append(
            {
                "lead": str(lead),
                "fold": int(str(fold)),
                "model_variant": str(variant),
                "observations": len(group),
                "base_rate": float(actual.mean()),
                "mean_probability": float(probability.mean()),
                "brier_score": float(brier_score_loss(actual, probability)),
                "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
                "roc_auc": area,
                "accuracy_at_55pct": float((probability.ge(ENTRY_PROBABILITY) == actual).mean()),
            }
        )
    return pd.DataFrame(rows)


def simulate_hourly_weights(
    returns: pd.DataFrame, target_weights: pd.DataFrame, *, cost_bps: float
) -> pd.DataFrame:
    """Apply target weights on the next observed hourly bar and charge turnover costs."""
    aligned_returns = returns.sort_index().astype(float)
    target = target_weights.reindex(
        index=aligned_returns.index, columns=aligned_returns.columns, fill_value=0.0
    ).fillna(0.0)
    implemented = target.shift(1).fillna(0.0)
    turnover = implemented.diff().abs().div(2.0)
    turnover.iloc[0] = implemented.iloc[0].abs().div(2.0)
    gross = (implemented * aligned_returns.fillna(0.0)).sum(axis=1)
    hourly_turnover = turnover.sum(axis=1)
    cost = hourly_turnover * cost_bps / 10_000.0
    if len(cost):
        liquidation = implemented.iloc[-1].abs().sum() / 2.0
        cost.iloc[-1] += liquidation * cost_bps / 10_000.0
        hourly_turnover.iloc[-1] += liquidation
    return pd.DataFrame(
        {
            "gross_return": gross,
            "cost": cost,
            "net_return": gross - cost,
            "gross_exposure": implemented.abs().sum(axis=1),
            "turnover": hourly_turnover,
        },
        index=aligned_returns.index,
    )


def _scope_predictions(predictions: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "combined":
        return predictions
    return predictions.loc[predictions["lead"].eq(scope)]


def build_costed_sleeve(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    cost_bps: float = IMPLEMENTATION_COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate individual leads and their combined rates sleeve in each outer fold."""
    frame = features.loc[features["symbol"].isin({"ZNU6", "ZTU6"})].copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame["hour_start_utc"] = pd.to_datetime(frame["hour_start_utc"], utc=True)
    frame["simple_return_1h"] = np.expm1(frame["log_return_1h"].astype(float))
    scopes = ("combined", *(lead.name for lead in RATE_LEADS))
    hourly_results: list[pd.DataFrame] = []
    daily_results: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        fold_number = int(str(fold["fold"]))
        validation_start = pd.Timestamp(str(fold["validation_start"]))
        validation_end = pd.Timestamp(str(fold["validation_end"]))
        validation = frame.loc[
            frame["trading_date"].between(validation_start, validation_end)
        ].copy()
        timestamp_to_date = validation.drop_duplicates("hour_start_utc").set_index(
            "hour_start_utc"
        )["trading_date"]
        date_lookup = cast(dict[Any, Any], timestamp_to_date.to_dict())
        returns = validation.pivot(
            index="hour_start_utc", columns="symbol", values="simple_return_1h"
        ).sort_index()
        for variant in ("displacement_only", "full_state"):
            fold_predictions = predictions.loc[
                predictions["fold"].eq(fold_number) & predictions["model_variant"].eq(variant)
            ].copy()
            for scope in scopes:
                selected = _scope_predictions(fold_predictions, scope).copy()
                selected["confidence"] = (
                    (selected["reversion_probability"] - ENTRY_PROBABILITY)
                    / (1.0 - ENTRY_PROBABILITY)
                ).clip(0.0, 1.0)
                selected["direction"] = -np.sign(selected["regression_displacement_24h"])
                per_instrument_risk = TARGET_VOLATILITY / np.sqrt(2.0)
                selected["target_weight"] = (
                    selected["direction"]
                    * selected["confidence"]
                    * per_instrument_risk
                    / selected["realized_volatility_24h"].replace(0.0, np.nan)
                ).clip(-MAXIMUM_ASSET_WEIGHT, MAXIMUM_ASSET_WEIGHT)
                target_long = (
                    selected.groupby(["hour_start_utc", "symbol"], observed=True)["target_weight"]
                    .mean()
                    .reset_index()
                )
                if target_long.empty:
                    target = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
                else:
                    target = target_long.pivot(
                        index="hour_start_utc", columns="symbol", values="target_weight"
                    ).reindex(index=returns.index, columns=returns.columns, fill_value=0.0)
                simulated = simulate_hourly_weights(returns, target, cost_bps=cost_bps)
                simulated["trading_date"] = [
                    date_lookup.get(timestamp) for timestamp in simulated.index
                ]
                simulated["fold"] = fold_number
                simulated["model_variant"] = variant
                simulated["scope"] = scope
                hourly_results.append(simulated.reset_index(names="hour_start_utc"))
                daily = simulated.groupby("trading_date", observed=True).agg(
                    gross_return=("gross_return", "sum"),
                    cost=("cost", "sum"),
                    net_return=("net_return", "sum"),
                    average_gross_exposure=("gross_exposure", "mean"),
                    turnover=("turnover", "sum"),
                )
                daily["fold"] = fold_number
                daily["model_variant"] = variant
                daily["scope"] = scope
                daily_results.append(daily.reset_index())
                active_hours = int(target.abs().sum(axis=1).gt(0.0).sum())
                prior_target = target.shift(1).fillna(0.0)
                entries = int(
                    (
                        (target.ne(0.0))
                        & ((prior_target.eq(0.0)) | (np.sign(target) != np.sign(prior_target)))
                    )
                    .sum()
                    .sum()
                )
                summary_rows.append(
                    {
                        "fold": fold_number,
                        "model_variant": variant,
                        "scope": scope,
                        "sessions": len(daily),
                        "annualized_return": float(daily["net_return"].mean() * 252.0),
                        "annualized_volatility": annualized_volatility(daily["net_return"]),
                        "sharpe": sharpe_ratio(daily["net_return"]),
                        "maximum_drawdown": maximum_drawdown(daily["net_return"]),
                        "total_cost": float(daily["cost"].sum()),
                        "average_daily_turnover": float(daily["turnover"].mean()),
                        "active_hours": active_hours,
                        "entries": entries,
                    }
                )
    return (
        pd.concat(hourly_results, ignore_index=True),
        pd.concat(daily_results, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def run(project_root: Path) -> None:
    """Run the rates-only probability comparison and costed development retest."""
    source = project_root / "data/processed/regression_displacement"
    features = pd.read_parquet(source / "development_hourly_features.parquet")
    targets = pd.read_parquet(source / "development_hourly_targets.parquet")
    dataset = features.merge(
        targets,
        on=["symbol", "trading_date", "hour_start_utc"],
        how="inner",
        validate="one_to_one",
    )
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    predictions, coefficients = fit_probability_models(dataset, folds)
    diagnostics = classification_diagnostics(predictions)
    hourly, daily, summary = build_costed_sleeve(features, predictions, folds)

    output = project_root / "data/processed/rates_reversion"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "validation_probabilities.parquet", index=False)
    coefficients.to_csv(output / "fold_coefficients.csv", index=False)
    diagnostics.to_csv(output / "probability_diagnostics.csv", index=False)
    hourly.to_parquet(output / "costed_hourly_returns.parquet", index=False)
    daily.to_parquet(output / "costed_daily_returns.parquet", index=False)
    summary.to_csv(output / "costed_walk_forward_summary.csv", index=False)
    combined = summary.loc[summary["scope"].eq("combined")]
    combined_gross = (
        daily.loc[daily["scope"].eq("combined")]
        .groupby(["model_variant", "fold"], observed=True)["gross_return"]
        .sum()
    )
    verdict = {
        "verdict": "REJECT",
        "hypothesis": "Trade >=3 sigma rates displacement toward the 24-hour regression line.",
        "costed_positive_model_folds": int((combined["annualized_return"] > 0.0).sum()),
        "costed_model_folds": int(len(combined)),
        "gross_positive_model_folds": int((combined_gross > 0.0).sum()),
        "gross_model_folds": int(len(combined_gross)),
        "reason": (
            "The combined sleeve was negative after costs in every fold for both probability "
            "models and gross-positive in only two of eight model-fold evaluations. A projected "
            "regression line can catch up to price without a sufficiently profitable contrarian "
            "price move. The first-passage target therefore does not support this trading rule."
        ),
        "repair_policy": (
            "Do not tune probability, displacement, cost, or sizing thresholds on these folds. "
            "Retain the failed experiment and require a distinct target or independent history "
            "before revisiting it."
        ),
        "cl_status": "REJECTED and not modeled in this branch",
    }
    (output / "research_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    registry = {
        "status": "rejected_trading_rule",
        "leads": [lead.name for lead in RATE_LEADS],
        "rejected_instruments": list(REJECTED_MEAN_REVERSION_INSTRUMENTS),
        "models": {
            "displacement_only": list(BASELINE_FEATURES),
            "full_state": list(FULL_STATE_FEATURES),
        },
        "target": "probability that half-reversion occurs before 1.5x adverse displacement",
        "entry": {
            "minimum_absolute_regression_zscore": EXTREME_DISPLACEMENT,
            "minimum_probability": ENTRY_PROBABILITY,
            "direction": "opposite the current 24-hour regression displacement",
        },
        "execution": {
            "delay": "next observed hourly bar",
            "cost_bps": IMPLEMENTATION_COST_BPS,
            "maximum_asset_weight": MAXIMUM_ASSET_WEIGHT,
            "target_annualized_volatility": TARGET_VOLATILITY,
        },
        "validation": (
            "Same outer development folds used to observe these leads; post-selection retest, "
            "not fresh out-of-sample evidence; sealed holdout not accessed."
        ),
    }
    (output / "research_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rates-only reversion development retest.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run(arguments.project_root.resolve())


if __name__ == "__main__":
    main()
