"""Development-only probabilistic regime ensemble."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore[import-untyped]
from statsmodels.tsa.regime_switching.markov_regression import (  # type: ignore[import-untyped]
    MarkovRegression,
)

REGIMES = (
    "low_volatility_trend",
    "high_volatility_trend",
    "mean_reverting_choppy",
    "stress_transition",
)


def _row_mean(frame: pd.DataFrame, columns: list[str], *, absolute: bool = False) -> pd.Series:
    available = [column for column in columns if column in frame]
    values = frame[available]
    if absolute:
        values = values.abs()
    return values.mean(axis=1)


def _rolling_correlation_concentration(panel: pd.DataFrame, window: int = 60) -> pd.Series:
    returns = panel.pivot(
        index="trading_date", columns="symbol", values="close_to_close_return"
    ).sort_index()
    rows: list[float] = []
    for end in range(len(returns)):
        start = max(0, end - window + 1)
        correlation = returns.iloc[start : end + 1].corr(min_periods=min(20, window))
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
        rows.append(float(upper.abs().mean()) if not upper.empty else np.nan)
    concentration = pd.Series(rows, index=returns.index, name="correlation_concentration")
    minimum = concentration.expanding(min_periods=120).min().shift(1)
    maximum = concentration.expanding(min_periods=120).max().shift(1)
    return concentration.sub(minimum).div((maximum - minimum).where(maximum.ne(minimum)))


def build_observable_regime_probabilities(
    model_features: pd.DataFrame, panel: pd.DataFrame
) -> pd.DataFrame:
    """Map causal observable measures to soft regime probabilities."""
    numeric = model_features.drop(
        columns=["symbol", "trading_date", "feature_available_at_utc", "model_row_complete"]
    )
    aggregate = numeric.groupby(pd.to_datetime(model_features["trading_date"])).median()
    aggregate.index.name = "trading_date"
    trend_strength = _row_mean(
        aggregate,
        [
            "standardized_return_20s",
            "trend_efficiency_20s",
            "trend_efficiency_60s",
            "ema_gap_5_20s",
            "ema_gap_10_60s",
        ],
        absolute=True,
    )
    volatility = _row_mean(
        aggregate,
        ["realized_volatility_20s", "atr_14s_fraction", "return_volatility_20s"],
    )
    acceleration = _row_mean(aggregate, ["volatility_ratio_5_20s", "range_zscore_20s"])
    range_expansion = _row_mean(aggregate, ["range_percentile_60s", "true_range_fraction"])
    mean_reversion = -_row_mean(aggregate, ["return_autocorrelation_20s", "standardized_return_5s"])
    liquidity_stress = -_row_mean(aggregate, ["volume_zscore_20s", "volume_per_observed_minute"])
    correlation = _rolling_correlation_concentration(panel).reindex(aggregate.index)

    scores = pd.DataFrame(index=aggregate.index)
    scores["low_volatility_trend"] = trend_strength - 0.75 * volatility - 0.25 * acceleration
    scores["high_volatility_trend"] = trend_strength + 0.50 * volatility + 0.50 * acceleration
    scores["mean_reverting_choppy"] = (
        mean_reversion - 0.50 * trend_strength - 0.25 * range_expansion
    )
    scores["stress_transition"] = (
        volatility
        + acceleration
        + range_expansion
        + correlation.fillna(0.0)
        + 0.50 * liquidity_stress
    )
    valid = scores.notna().all(axis=1)
    probabilities = pd.DataFrame(np.nan, index=scores.index, columns=list(REGIMES))
    probabilities.loc[valid] = softmax(scores.loc[valid].to_numpy(dtype=float), axis=1)
    entropy = -(probabilities * np.log(probabilities.clip(lower=1e-12))).sum(
        axis=1, min_count=len(REGIMES)
    )
    probabilities["observable_confidence"] = 1.0 - entropy / np.log(len(REGIMES))
    probabilities["correlation_concentration"] = correlation
    return probabilities.reset_index()


def _markov_state_mapping(parameters: pd.Series) -> dict[int, str]:
    means = {state: float(parameters[f"const[{state}]"]) for state in range(4)}
    variances = {state: float(parameters[f"sigma2[{state}]"]) for state in range(4)}
    stress = max(variances, key=lambda state: variances[state])
    remaining = [state for state in range(4) if state != stress]
    low_volatility = min(remaining, key=lambda state: variances[state])
    remaining.remove(low_volatility)
    high_volatility_trend = max(
        remaining,
        key=lambda state: abs(means[state]) / np.sqrt(max(variances[state], 1e-12)),
    )
    remaining.remove(high_volatility_trend)
    return {
        low_volatility: "low_volatility_trend",
        high_volatility_trend: "high_volatility_trend",
        remaining[0]: "mean_reverting_choppy",
        stress: "stress_transition",
    }


def build_markov_walk_forward_probabilities(
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    maximum_iterations: int = 500,
    search_repetitions: int = 5,
) -> pd.DataFrame:
    """Fit on each fold's training set and filter validation observations causally."""
    market_return = (
        panel.pivot(index="trading_date", columns="symbol", values="close_to_close_return")
        .mean(axis=1)
        .dropna()
        .mul(100.0)
    )
    market_return.index = pd.to_datetime(market_return.index)
    rows: list[pd.DataFrame] = []
    for fold in folds.to_dict(orient="records"):
        train_end = pd.Timestamp(str(fold["train_end"]))
        validation_start = pd.Timestamp(str(fold["validation_start"]))
        validation_end = pd.Timestamp(str(fold["validation_end"]))
        train = market_return.loc[:train_end]
        through_validation = market_return.loc[:validation_end]
        model = MarkovRegression(
            train.reset_index(drop=True), k_regimes=4, trend="c", switching_variance=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = model.fit(
                maxiter=maximum_iterations,
                disp=False,
                search_reps=search_repetitions,
                em_iter=10,
            )
        if not bool(result.mle_retvals.get("converged", False)):
            continue
        filter_model = MarkovRegression(
            through_validation.reset_index(drop=True),
            k_regimes=4,
            trend="c",
            switching_variance=True,
        )
        filtered = filter_model.filter(result.params).filtered_marginal_probabilities
        filtered.index = through_validation.index
        filtered = filtered.loc[validation_start:validation_end]
        mapping = _markov_state_mapping(result.params)
        named = pd.DataFrame(index=filtered.index)
        for state, regime in mapping.items():
            named[regime] = filtered[state]
        named = named[list(REGIMES)]
        entropy = -(named * np.log(named.clip(lower=1e-12))).sum(axis=1)
        named["markov_confidence"] = 1.0 - entropy / np.log(len(REGIMES))
        named["fold"] = int(str(fold["fold"]))
        rows.append(named.reset_index(names="trading_date"))
    if not rows:
        return pd.DataFrame(columns=["trading_date", *REGIMES, "markov_confidence", "fold"])
    return pd.concat(rows, ignore_index=True)


def combine_regime_probabilities(
    observable: pd.DataFrame, markov: pd.DataFrame, *, markov_weight: float = 0.25
) -> pd.DataFrame:
    """Blend Markov and observable probabilities; observables dominate by design."""
    if not 0.0 <= markov_weight <= 0.5:
        raise ValueError("markov_weight must be between 0 and 0.5")
    markov_columns = {regime: f"markov_{regime}" for regime in REGIMES}
    renamed = markov.rename(columns=markov_columns)
    result = observable.merge(renamed, on="trading_date", how="left")
    available = result["markov_confidence"].notna()
    for regime in REGIMES:
        observed = result[regime]
        markov_probability = result[f"markov_{regime}"]
        result[f"ensemble_{regime}"] = observed.where(
            ~available, (1.0 - markov_weight) * observed + markov_weight * markov_probability
        )
    ensemble_columns = [f"ensemble_{regime}" for regime in REGIMES]
    entropy = -(result[ensemble_columns] * np.log(result[ensemble_columns].clip(lower=1e-12))).sum(
        axis=1, min_count=len(REGIMES)
    )
    result["ensemble_confidence"] = 1.0 - entropy / np.log(len(REGIMES))
    result["markov_available"] = available
    return result


def write_regime_probabilities(
    model_feature_path: Path = Path(
        "data/processed/model_inputs/development_model_features.parquet"
    ),
    panel_path: Path = Path("data/processed/research/development_session_panel.parquet"),
    fold_path: Path = Path("data/catalog/development_walk_forward_folds.csv"),
    output_path: Path = Path("data/processed/regimes/development_regime_probabilities.parquet"),
) -> pd.DataFrame:
    model_features = pd.read_parquet(model_feature_path)
    panel = pd.read_parquet(panel_path)
    folds = pd.read_csv(fold_path)
    observable = build_observable_regime_probabilities(model_features, panel)
    markov = build_markov_walk_forward_probabilities(panel, folds)
    ensemble = combine_regime_probabilities(observable, markov)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensemble.to_parquet(output_path, index=False, compression="zstd")
    return ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Build development-only regime probabilities.")
    parser.parse_args()
    regimes = write_regime_probabilities()
    print(
        f"Built {len(regimes):,} regime rows; "
        f"Markov probabilities available for {int(regimes['markov_available'].sum()):,}."
    )


if __name__ == "__main__":
    main()
