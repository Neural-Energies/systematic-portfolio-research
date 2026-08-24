"""Development-only hourly regression-displacement research."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # type: ignore[import-untyped]

REGRESSION_FEATURES = (
    "regression_displacement_24h",
    "regression_residual_zscore_24h",
    "prediction_interval_position_24h",
    "regression_slope_24h",
    "regression_slope_change_4h",
    "regression_r_squared_24h",
    "residual_volatility_24h",
    "residual_to_asset_volatility_24h",
    "residual_autocorrelation_120h",
    "residual_half_life_120h",
    "outside_two_sigma_run",
    "displacement_speed_1h",
    "displacement_acceleration_1h",
    "regression_residual_zscore_120h",
    "regression_slope_120h",
    "regression_r_squared_120h",
    "realized_volatility_24h",
    "volatility_ratio_24h_120h",
    "range_expansion_24h",
    "volume_zscore_24h",
    "trend_efficiency_24h",
    "peer_return_1h",
    "factor_residual_zscore_24h",
    "cross_asset_correlation_120h",
    "prior_trend_probability",
    "prior_stress_probability",
)

TARGET_HORIZONS = (1, 2, 4, 8, 23, 46, 69, 115)
MODEL_HORIZONS = (8, 23, 115)


def aggregate_hourly_bars(minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed minute bars without bridging hours or using later observations."""
    required = {
        "symbol",
        "trading_date",
        "timestamp_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = sorted(required - set(minute_bars.columns))
    if missing:
        raise ValueError(f"missing minute-bar columns: {missing}")
    frame = minute_bars.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    frame = frame.sort_values(["symbol", "timestamp_utc"], ignore_index=True)
    frame["hour_start_utc"] = frame["timestamp_utc"].dt.floor("h")
    grouped = frame.groupby(["symbol", "trading_date", "hour_start_utc"], sort=True, observed=True)
    hourly = grouped.agg(
        feature_available_at_utc=("timestamp_utc", "max"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        minute_observations=("timestamp_utc", "size"),
    )
    return hourly.reset_index().sort_values(["symbol", "hour_start_utc"], ignore_index=True)


def _rolling_linear_state(log_price: pd.Series, window: int) -> pd.DataFrame:
    """Fit causal rolling price-versus-time OLS and return endpoint state."""
    x = np.arange(window, dtype=float)
    x_mean = float(x.mean())
    sxx = float(np.square(x - x_mean).sum())
    rolling = log_price.rolling(window, min_periods=window)
    sum_y = rolling.sum()
    sum_y2 = log_price.pow(2).rolling(window, min_periods=window).sum()
    weighted_sum = rolling.apply(lambda values: float(np.dot(values, x)), raw=True)
    slope = (weighted_sum - x_mean * sum_y) / sxx
    mean = sum_y / window
    fitted = mean + slope * (window - 1 - x_mean)
    total_sum_squares = (sum_y2 - sum_y.pow(2) / window).clip(lower=0.0)
    regression_sum_squares = slope.pow(2) * sxx
    error_sum_squares = (total_sum_squares - regression_sum_squares).clip(lower=0.0)
    r_squared = regression_sum_squares.div(total_sum_squares.replace(0.0, np.nan)).clip(0.0, 1.0)
    residual_standard_error = np.sqrt(error_sum_squares / max(window - 2, 1))
    endpoint_leverage = np.sqrt(1.0 + 1.0 / window + (window - 1 - x_mean) ** 2 / sxx)
    return pd.DataFrame(
        {
            "fitted": fitted,
            "slope": slope,
            "r_squared": r_squared,
            "residual_standard_error": residual_standard_error,
            "prediction_standard_error": residual_standard_error * endpoint_leverage,
        },
        index=log_price.index,
    )


def _outside_run(values: pd.Series, threshold: float = 2.0) -> pd.Series:
    run = 0
    result = np.zeros(len(values), dtype=float)
    for index, value in enumerate(values.to_numpy(dtype=float)):
        run = run + 1 if np.isfinite(value) and abs(value) >= threshold else 0
        result[index] = run
    return pd.Series(result, index=values.index)


def _single_asset_features(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.sort_values("hour_start_utc").copy()
    log_price = pd.Series(
        np.log(frame["close"].to_numpy(dtype=float)), index=frame.index, name="log_price"
    )
    frame["log_price"] = log_price
    frame["log_return_1h"] = log_price.diff()

    short = _rolling_linear_state(log_price, 24)
    long = _rolling_linear_state(log_price, 120)
    frame["fitted_log_price_24h"] = short["fitted"]
    frame["regression_displacement_24h"] = log_price - short["fitted"]
    short_residual_scale = frame["regression_displacement_24h"].rolling(120, min_periods=48).std()
    frame["regression_residual_zscore_24h"] = frame["regression_displacement_24h"].div(
        short_residual_scale.replace(0.0, np.nan)
    )
    frame["prediction_interval_position_24h"] = frame["regression_displacement_24h"].div(
        short["prediction_standard_error"].replace(0.0, np.nan)
    )
    frame["regression_slope_24h"] = short["slope"]
    frame["regression_slope_change_4h"] = short["slope"].diff(4)
    frame["regression_r_squared_24h"] = short["r_squared"]
    frame["residual_volatility_24h"] = (
        frame["regression_displacement_24h"].rolling(24, min_periods=24).std()
    )
    hourly_volatility_24h = frame["log_return_1h"].rolling(24, min_periods=24).std()
    frame["residual_to_asset_volatility_24h"] = frame["residual_volatility_24h"].div(
        hourly_volatility_24h.replace(0.0, np.nan)
    )
    frame["residual_autocorrelation_120h"] = (
        frame["regression_displacement_24h"]
        .rolling(120, min_periods=48)
        .corr(frame["regression_displacement_24h"].shift(1))
    )
    autocorrelation = frame["residual_autocorrelation_120h"].where(
        frame["residual_autocorrelation_120h"].between(0.0, 0.999999)
    )
    frame["residual_half_life_120h"] = (-np.log(2.0) / np.log(autocorrelation)).clip(1.0, 240.0)
    frame["outside_two_sigma_run"] = _outside_run(frame["regression_residual_zscore_24h"])
    frame["displacement_speed_1h"] = frame["regression_residual_zscore_24h"].diff()
    frame["displacement_acceleration_1h"] = frame["displacement_speed_1h"].diff()

    frame["fitted_log_price_120h"] = long["fitted"]
    frame["regression_displacement_120h"] = log_price - long["fitted"]
    long_residual_scale = frame["regression_displacement_120h"].rolling(240, min_periods=120).std()
    frame["regression_residual_zscore_120h"] = frame["regression_displacement_120h"].div(
        long_residual_scale.replace(0.0, np.nan)
    )
    frame["regression_slope_120h"] = long["slope"]
    frame["regression_r_squared_120h"] = long["r_squared"]

    annualization = np.sqrt(23.0 * 252.0)
    frame["realized_volatility_24h"] = hourly_volatility_24h * annualization
    volatility_120h = frame["log_return_1h"].rolling(120, min_periods=48).std()
    frame["volatility_ratio_24h_120h"] = hourly_volatility_24h.div(
        volatility_120h.replace(0.0, np.nan)
    )
    hourly_range = (frame["high"] - frame["low"]).div(frame["close"])
    frame["range_expansion_24h"] = hourly_range.div(
        hourly_range.rolling(120, min_periods=48).median().replace(0.0, np.nan)
    )
    volume_mean = frame["volume"].rolling(120, min_periods=48).mean()
    volume_std = frame["volume"].rolling(120, min_periods=48).std()
    frame["volume_zscore_24h"] = (frame["volume"] - volume_mean).div(
        volume_std.replace(0.0, np.nan)
    )
    path = frame["log_return_1h"].abs().rolling(24, min_periods=24).sum()
    frame["trend_efficiency_24h"] = log_price.diff(24).abs().div(path.replace(0.0, np.nan))
    return frame


def build_regression_features(
    hourly_bars: pd.DataFrame, regimes: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build 26 causal regression, risk, cross-asset, and prior-regime features."""
    pieces = [
        _single_asset_features(group)
        for _, group in hourly_bars.groupby("symbol", sort=True, observed=True)
    ]
    result = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "hour_start_utc"], ignore_index=True
    )
    returns = result.pivot(
        index="hour_start_utc", columns="symbol", values="log_return_1h"
    ).sort_index()
    count = returns.notna().sum(axis=1)
    peer_returns = returns.rsub(returns.sum(axis=1), axis=0).div(
        count.sub(1).replace(0.0, np.nan), axis=0
    )
    cross_asset: list[pd.DataFrame] = []
    for symbol in returns:
        asset = returns[symbol]
        peer = peer_returns[symbol]
        peer_variance = peer.rolling(120, min_periods=48).var()
        beta = asset.rolling(120, min_periods=48).cov(peer).div(peer_variance.replace(0.0, np.nan))
        factor_residual = asset - beta * peer
        factor_displacement = factor_residual.rolling(24, min_periods=24).sum()
        factor_scale = factor_residual.rolling(120, min_periods=48).std() * np.sqrt(24.0)
        cross_asset.append(
            pd.DataFrame(
                {
                    "hour_start_utc": returns.index,
                    "symbol": str(symbol),
                    "peer_return_1h": peer,
                    "factor_residual_zscore_24h": factor_displacement.div(
                        factor_scale.replace(0.0, np.nan)
                    ),
                    "cross_asset_correlation_120h": asset.rolling(120, min_periods=48).corr(peer),
                }
            )
        )
    result = result.merge(
        pd.concat(cross_asset, ignore_index=True),
        on=["symbol", "hour_start_utc"],
        how="left",
        validate="one_to_one",
    )
    if regimes is not None and not regimes.empty:
        daily = regimes.copy()
        daily["trading_date"] = pd.to_datetime(daily["trading_date"])
        daily = daily.sort_values("trading_date")
        daily["prior_trend_probability"] = (
            daily["ensemble_low_volatility_trend"] + daily["ensemble_high_volatility_trend"]
        ).shift(1)
        daily["prior_stress_probability"] = daily["ensemble_stress_transition"].shift(1)
        result = result.merge(
            daily[["trading_date", "prior_trend_probability", "prior_stress_probability"]],
            on="trading_date",
            how="left",
            validate="many_to_one",
        )
    else:
        result["prior_trend_probability"] = np.nan
        result["prior_stress_probability"] = np.nan
    result["regression_feature_set_complete"] = (
        result[list(REGRESSION_FEATURES)].notna().all(axis=1)
    )
    return result.sort_values(["symbol", "hour_start_utc"], ignore_index=True)


def _first_passage_labels(group: pd.DataFrame, horizon: int) -> pd.DataFrame:
    log_price = group["log_price"].to_numpy(dtype=float)
    fitted = group["fitted_log_price_24h"].to_numpy(dtype=float)
    slope = group["regression_slope_24h"].to_numpy(dtype=float)
    initial = group["regression_displacement_24h"].to_numpy(dtype=float)
    target_before_stop = np.full(len(group), np.nan)
    time_to_half = np.full(len(group), np.nan)
    for index in range(len(group) - horizon):
        if not all(np.isfinite([fitted[index], slope[index], initial[index]])):
            continue
        if abs(initial[index]) < 1e-12:
            continue
        steps = np.arange(1, horizon + 1, dtype=float)
        path = log_price[index + 1 : index + horizon + 1] - (fitted[index] + slope[index] * steps)
        target_hits = np.flatnonzero(np.abs(path) <= 0.5 * abs(initial[index]))
        stop_hits = np.flatnonzero(np.abs(path) >= 1.5 * abs(initial[index]))
        target_step = int(target_hits[0] + 1) if len(target_hits) else None
        stop_step = int(stop_hits[0] + 1) if len(stop_hits) else None
        target_before_stop[index] = float(
            target_step is not None and (stop_step is None or target_step < stop_step)
        )
        if target_step is not None:
            time_to_half[index] = float(target_step)
    return pd.DataFrame(
        {
            f"target_before_stop_{horizon}h": target_before_stop,
            f"time_to_half_reversion_{horizon}h": time_to_half,
        },
        index=group.index,
    )


def build_regression_targets(
    features: pd.DataFrame, horizons: tuple[int, ...] = TARGET_HORIZONS
) -> pd.DataFrame:
    """Build future-return, displacement, and first-passage targets separately from features."""
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("target horizons must be positive")
    output: list[pd.DataFrame] = []
    for _, raw_group in features.groupby("symbol", sort=True, observed=True):
        group = raw_group.sort_values("hour_start_utc").copy()
        targets = group[["symbol", "trading_date", "hour_start_utc"]].copy()
        for horizon in horizons:
            future_log_price = group["log_price"].shift(-horizon)
            targets[f"future_log_return_{horizon}h"] = future_log_price - group["log_price"]
            targets[f"label_available_at_{horizon}h_utc"] = group["feature_available_at_utc"].shift(
                -horizon
            )
            projected_equilibrium = (
                group["fitted_log_price_24h"] + group["regression_slope_24h"] * horizon
            )
            future_displacement = future_log_price - projected_equilibrium
            targets[f"future_displacement_change_{horizon}h"] = (
                future_displacement - group["regression_displacement_24h"]
            )
            targets[f"partial_reversion_at_{horizon}h"] = (
                future_displacement.abs() <= 0.5 * group["regression_displacement_24h"].abs()
            ).where(future_log_price.notna())
            targets[f"complete_reversion_at_{horizon}h"] = (
                future_displacement * group["regression_displacement_24h"] <= 0.0
            ).where(future_log_price.notna())
            targets = pd.concat([targets, _first_passage_labels(group, horizon)], axis=1)
        output.append(targets)
    return pd.concat(output, ignore_index=True).sort_values(
        ["symbol", "hour_start_utc"], ignore_index=True
    )


def make_model(alpha: float = 10.0) -> Pipeline:
    """Create a fixed Ridge baseline with fold-local preprocessing."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
        ]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric, list(REGRESSION_FEATURES)),
            (
                "symbol",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["symbol"],
            ),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("regression", Ridge(alpha=alpha))])


def fit_walk_forward_models(
    dataset: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = MODEL_HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit fixed pooled Ridge baselines and emit outer-fold validation forecasts only."""
    required = {
        "symbol",
        "trading_date",
        "hour_start_utc",
        "feature_available_at_utc",
        "realized_volatility_24h",
        *REGRESSION_FEATURES,
    }
    for horizon in horizons:
        required.update({f"future_log_return_{horizon}h", f"label_available_at_{horizon}h_utc"})
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"missing regression model columns: {missing}")
    frame = dataset.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    model_columns = [*REGRESSION_FEATURES, "symbol"]
    forecasts: list[pd.DataFrame] = []
    coefficients: list[pd.DataFrame] = []
    for horizon in horizons:
        label_column = f"future_log_return_{horizon}h"
        availability_column = f"label_available_at_{horizon}h_utc"
        label_available = pd.to_datetime(frame[availability_column], utc=True)
        target_scale = frame["realized_volatility_24h"] * np.sqrt(horizon / (23.0 * 252.0))
        target = frame[label_column].div(target_scale.replace(0.0, np.nan)).clip(-5.0, 5.0)
        for raw_fold in folds.to_dict("records"):
            fold = cast(dict[str, Any], raw_fold)
            train_end = pd.Timestamp(str(fold["train_end"]))
            cutoff = (train_end + pd.Timedelta(days=1)).tz_localize("UTC")
            validation_start = pd.Timestamp(str(fold["validation_start"]))
            validation_end = pd.Timestamp(str(fold["validation_end"]))
            training_mask = (
                frame["trading_date"].le(train_end) & label_available.lt(cutoff) & target.notna()
            )
            validation_mask = frame["trading_date"].between(validation_start, validation_end)
            training = frame.loc[training_mask]
            validation = frame.loc[validation_mask]
            if training.empty or validation.empty:
                raise ValueError(
                    f"fold {fold['fold']} horizon {horizon} has empty training or validation"
                )
            model = make_model()
            model.fit(training[model_columns], target.loc[training_mask])
            prediction = validation[["trading_date", "hour_start_utc", "symbol"]].copy()
            prediction["forecast"] = model.predict(validation[model_columns])
            prediction["fold"] = int(str(fold["fold"]))
            prediction["horizon_hours"] = horizon
            forecasts.append(prediction)
            names = model.named_steps["preprocessing"].get_feature_names_out()
            values = cast(Ridge, model.named_steps["regression"]).coef_
            coefficients.append(
                pd.DataFrame(
                    {
                        "fold": int(str(fold["fold"])),
                        "horizon_hours": horizon,
                        "feature": names,
                        "coefficient": values,
                    }
                )
            )
    return pd.concat(forecasts, ignore_index=True), pd.concat(coefficients, ignore_index=True)


def prediction_diagnostics(forecasts: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Measure outer-fold association without converting forecasts into a trading rule."""
    rows: list[dict[str, Any]] = []
    for horizon, horizon_forecasts in forecasts.groupby("horizon_hours", sort=True):
        horizon_hours = int(str(horizon))
        label = f"future_log_return_{horizon_hours}h"
        joined = horizon_forecasts.merge(
            targets[["symbol", "hour_start_utc", label]],
            on=["symbol", "hour_start_utc"],
            how="left",
            validate="one_to_one",
        )
        for (fold, symbol), group in joined.groupby(["fold", "symbol"], sort=True, observed=True):
            valid = group[["forecast", label]].dropna()
            rows.append(
                {
                    "fold": int(str(fold)),
                    "symbol": str(symbol),
                    "horizon_hours": horizon_hours,
                    "observations": len(valid),
                    "pearson_ic": float(valid["forecast"].corr(valid[label])),
                    "spearman_ic": float(valid["forecast"].corr(valid[label], method="spearman")),
                    "directional_accuracy": float(
                        (np.sign(valid["forecast"]) == np.sign(valid[label])).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def conditional_reversion_diagnostics(
    features: pd.DataFrame, targets: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    """Summarize first-passage behavior by predeclared displacement bands."""
    joined = features[
        ["symbol", "trading_date", "hour_start_utc", "regression_residual_zscore_24h"]
    ].merge(targets, on=["symbol", "trading_date", "hour_start_utc"], validate="one_to_one")
    joined["displacement_band"] = pd.cut(
        joined["regression_residual_zscore_24h"].abs(),
        bins=[0.0, 1.0, 2.0, 3.0, np.inf],
        labels=["under_1", "1_to_2", "2_to_3", "over_3"],
        include_lowest=True,
    )
    pieces: list[pd.DataFrame] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        validation = joined.loc[
            joined["trading_date"].between(
                pd.Timestamp(str(fold["validation_start"])),
                pd.Timestamp(str(fold["validation_end"])),
            )
        ]
        for horizon in TARGET_HORIZONS:
            grouped = validation.groupby(["symbol", "displacement_band"], observed=True).agg(
                observations=(f"target_before_stop_{horizon}h", "count"),
                target_before_stop_rate=(f"target_before_stop_{horizon}h", "mean"),
                partial_reversion_rate=(f"partial_reversion_at_{horizon}h", "mean"),
                complete_reversion_rate=(f"complete_reversion_at_{horizon}h", "mean"),
                median_time_to_half_reversion=(
                    f"time_to_half_reversion_{horizon}h",
                    "median",
                ),
            )
            grouped["fold"] = int(str(fold["fold"]))
            grouped["horizon_hours"] = horizon
            pieces.append(grouped.reset_index())
    return pd.concat(pieces, ignore_index=True)


def run(project_root: Path) -> None:
    """Execute the predeclared development-only regression-displacement study."""
    minute_root = project_root / "data/processed/research/development_minute_returns"
    hourly_parts: list[pd.DataFrame] = []
    for path in sorted(minute_root.glob("symbol=*/returns.parquet")):
        minute = pd.read_parquet(path)
        if not minute.empty:
            hourly_parts.append(aggregate_hourly_bars(minute))
    if not hourly_parts:
        raise FileNotFoundError("No non-empty development minute-return partitions were found")
    hourly = pd.concat(hourly_parts, ignore_index=True)
    regimes = pd.read_parquet(
        project_root / "data/processed/regimes/development_regime_probabilities.parquet"
    )
    features = build_regression_features(hourly, regimes)
    targets = build_regression_targets(features)
    dataset = features.merge(
        targets,
        on=["symbol", "trading_date", "hour_start_utc"],
        how="inner",
        validate="one_to_one",
    )
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    forecasts, coefficients = fit_walk_forward_models(dataset, folds)
    diagnostics = prediction_diagnostics(forecasts, targets)
    reversion = conditional_reversion_diagnostics(features, targets, folds)

    output = project_root / "data/processed/regression_displacement"
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "development_hourly_features.parquet", index=False)
    targets.to_parquet(output / "development_hourly_targets.parquet", index=False)
    forecasts.to_parquet(output / "validation_forecasts.parquet", index=False)
    coefficients.to_csv(output / "fold_coefficients.csv", index=False)
    diagnostics.to_csv(output / "prediction_diagnostics.csv", index=False)
    reversion.to_csv(output / "conditional_reversion_diagnostics.csv", index=False)
    registry = {
        "hypothesis": (
            "Point-in-time displacement from a rolling price-time regression may condition "
            "future returns and first-passage reversion probabilities."
        ),
        "status": "research_further",
        "regression_definition": "Rolling OLS of log price on time over 24 and 120 hourly bars.",
        "displacement_definition": (
            "Endpoint log-price residual normalized by prior residual dispersion and by the "
            "rolling prediction standard error."
        ),
        "features": list(REGRESSION_FEATURES),
        "target_horizons_hours": list(TARGET_HORIZONS),
        "model_horizons_hours": list(MODEL_HORIZONS),
        "model": "Fixed Ridge(alpha=10) pooled across instruments with fold-local preprocessing.",
        "validation": (
            "Expanding outer walk-forward folds; five-session embargo; labels must be realized "
            "before each training cutoff; final holdout not accessed."
        ),
        "trading_rule": None,
    }
    (output / "hypothesis_registry.json").write_text(
        json.dumps(registry, indent=2), encoding="utf-8"
    )
    print(diagnostics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run development-only hourly regression-displacement research."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run(arguments.project_root.resolve())


if __name__ == "__main__":
    main()
