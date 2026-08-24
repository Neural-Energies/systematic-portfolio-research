"""Point-in-time intraday mean-reversion portfolio sleeve."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.special import ndtr

from systematic_research.metrics import annualized_volatility, maximum_drawdown, sharpe_ratio

MINUTES_PER_SESSION = 23 * 60


def _same_time_zscore(values: pd.Series, minute: pd.Series, minimum_history: int = 60) -> pd.Series:
    """Standardize against prior sessions at the same minute, excluding the current observation."""
    grouped = values.groupby(minute, sort=False)
    prior_mean = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=minimum_history).mean()
    )
    prior_std = grouped.transform(
        lambda series: series.shift(1).expanding(min_periods=minimum_history).std()
    )
    return values.sub(prior_mean).div(prior_std.replace(0.0, np.nan))


def build_intraday_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build running-midpoint, time-of-day and exhaustion features with no future bars."""
    data = frame.sort_values("timestamp_utc", ignore_index=True).copy()
    data["trading_date"] = pd.to_datetime(data["trading_date"])
    timestamp = pd.to_datetime(data["timestamp_utc"], utc=True)
    local = timestamp.dt.tz_convert(ZoneInfo("America/New_York"))
    minute_of_day = local.dt.hour * 60 + local.dt.minute
    data["session_minute"] = (minute_of_day - 18 * 60) % (24 * 60)
    data["time_quartile"] = (
        np.floor(data["session_minute"] / (MINUTES_PER_SESSION / 4)).clip(0, 3) + 1
    ).astype(int)

    session = data.groupby("trading_date", sort=False)
    running_high = session["high"].cummax()
    running_low = session["low"].cummin()
    running_range = (running_high - running_low).replace(0.0, np.nan)
    data["running_midpoint"] = (running_high + running_low) / 2.0
    data["range_location"] = (data["close"] - running_low) / running_range
    data["range_quartile"] = np.ceil(data["range_location"].clip(0.0, 1.0) * 4.0).clip(1, 4)
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    dollar_volume = typical * data["volume"]
    cumulative_volume = session["volume"].cumsum()
    # Assigning a temporary named column keeps cumulative VWAP aligned within sessions.
    data["_dollar_volume"] = dollar_volume
    data["running_vwap"] = data.groupby("trading_date", sort=False)[
        "_dollar_volume"
    ].cumsum() / cumulative_volume.replace(0, np.nan)
    session_open = session["open"].transform("first")
    data["midpoint_displacement"] = (data["close"] - data["running_midpoint"]) / running_range
    data["vwap_deviation"] = data["close"] / data["running_vwap"] - 1.0
    data["open_deviation"] = data["close"] / session_open - 1.0
    data["return_5m"] = session["close"].pct_change(5, fill_method=None)
    data["running_range_fraction"] = running_range / session_open
    data["cumulative_volume"] = cumulative_volume

    minute = data["session_minute"]
    for column in (
        "vwap_deviation",
        "open_deviation",
        "return_5m",
        "running_range_fraction",
        "cumulative_volume",
    ):
        data[f"{column}_same_time_z"] = _same_time_zscore(data[column], minute)
    data["range_percentile"] = ndtr(data["running_range_fraction_same_time_z"])
    data["volume_pace_percentile"] = ndtr(data["cumulative_volume_same_time_z"])
    return data.drop(columns=["_dollar_volume"])


def build_combined_score(features: pd.DataFrame) -> pd.DataFrame:
    """Combine related overextension measures into one economically defined sleeve score."""
    frame = features.copy()
    midpoint = (-2.0 * frame["midpoint_displacement"]).clip(-1.0, 1.0)
    vwap = -np.tanh(frame["vwap_deviation_same_time_z"].clip(-5.0, 5.0) / 2.0)
    open_move = -np.tanh(frame["open_deviation_same_time_z"].clip(-5.0, 5.0) / 2.0)
    short_return = -np.tanh(frame["return_5m_same_time_z"].clip(-5.0, 5.0) / 2.0)
    frame["midpoint_score"] = midpoint
    frame["vwap_score"] = vwap
    frame["open_move_score"] = open_move
    frame["short_return_score"] = short_return
    activity = (0.50 + 0.50 * frame["range_percentile"]) * (
        0.75 + 0.25 * frame["volume_pace_percentile"]
    )
    last_return = frame["return_1m"].fillna(0.0)
    variants = {
        "balanced": (0.40, 0.25, 0.20, 0.15),
        "midpoint_dominant": (0.65, 0.15, 0.10, 0.10),
        "fast_reversal_dominant": (0.20, 0.10, 0.10, 0.60),
        "vwap_dominant": (0.20, 0.60, 0.10, 0.10),
    }
    components = (midpoint, vwap, open_move, short_return)
    for name, component_weights in variants.items():
        raw = sum(
            weight * component
            for weight, component in zip(component_weights, components, strict=True)
        )
        confirmation = np.where(np.sign(last_return) == np.sign(raw), 1.15, 0.85)
        frame[f"score_{name}"] = (raw * activity * confirmation).clip(-1.0, 1.0)
    frame["combined_score"] = frame["score_balanced"]
    ready = (
        frame[
            [
                "vwap_deviation_same_time_z",
                "open_deviation_same_time_z",
                "return_5m_same_time_z",
                "range_percentile",
                "volume_pace_percentile",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    score_columns = [column for column in frame if str(column).startswith("score_")]
    frame.loc[~ready, score_columns] = 0.0
    frame.loc[~ready, "combined_score"] = 0.0
    return frame


def hysteresis_positions(
    score: pd.Series,
    trading_date: pd.Series,
    session_minute: pd.Series,
    *,
    entry: float = 0.55,
    exit: float = 0.15,
    maximum_minutes: int = 60,
) -> pd.Series:
    """Turn a score into persistent positions with entry/exit bands and a holding cap."""
    positions = np.zeros(len(score), dtype=float)
    current = 0.0
    age = 0
    previous_date: Any = None
    observations = zip(score, trading_date, session_minute, strict=True)
    for index, (value, date, minute) in enumerate(observations):
        if date != previous_date or int(minute) >= MINUTES_PER_SESSION - 5:
            current, age = 0.0, 0
        if current == 0.0 and abs(float(value)) >= entry:
            current, age = float(np.sign(value)), 0
        elif current != 0.0:
            age += 1
            if (
                abs(float(value)) <= exit
                or np.sign(value) != np.sign(current)
                or age >= maximum_minutes
            ):
                current, age = 0.0, 0
        positions[index] = current
        previous_date = date
    return pd.Series(positions, index=score.index, name="position_signal")


def build_portfolio_weights(signals: pd.DataFrame, rebalance_minutes: int = 15) -> pd.DataFrame:
    """Combine absolute signals while constraining portfolio net exposure to +/-25%."""
    frame = signals.copy()
    frame["minute_volatility"] = frame.groupby("symbol", sort=False)["return_1m"].transform(
        lambda series: series.rolling(240, min_periods=120).std()
    )
    frame["risk_score"] = frame["position_signal"] / frame["minute_volatility"].replace(0.0, np.nan)
    group = frame.groupby("timestamp_utc", sort=False)
    frame["long_score"] = frame["risk_score"].clip(lower=0.0)
    frame["short_score"] = -frame["risk_score"].clip(upper=0.0)
    initial_long = group["long_score"].transform("sum")
    initial_short = group["short_score"].transform("sum")
    one_sided = initial_long.eq(0.0) | initial_short.eq(0.0)
    centered = frame["risk_score"] - group["risk_score"].transform("median")
    frame.loc[one_sided, "long_score"] = centered[one_sided].clip(lower=0.0)
    frame.loc[one_sided, "short_score"] = -centered[one_sided].clip(upper=0.0)
    long_total = group["long_score"].transform("sum").replace(0.0, np.nan)
    short_total = group["short_score"].transform("sum").replace(0.0, np.nan)
    desired_net = ((long_total - short_total) / (long_total + short_total)).clip(-0.25, 0.25)
    long_budget = (1.0 + desired_net) / 2.0
    short_budget = (1.0 - desired_net) / 2.0
    frame["target_weight"] = (
        long_budget * frame["long_score"] / long_total
        - short_budget * frame["short_score"] / short_total
    ).fillna(0.0)
    rebalance = frame["session_minute"].mod(rebalance_minutes).eq(0) | frame["session_minute"].ge(
        MINUTES_PER_SESSION - 5
    )
    frame["target_weight"] = frame["target_weight"].where(rebalance)
    frame["target_weight"] = (
        frame.groupby("symbol", sort=False)["target_weight"].ffill().fillna(0.0)
    )
    frame["implemented_weight"] = frame.groupby("symbol", sort=False)["target_weight"].shift(1)
    consecutive = frame.groupby("symbol", sort=False)["timestamp_utc"].diff().dt.total_seconds().eq(
        60.0
    ) & frame.groupby("symbol", sort=False)["trading_date"].diff().eq(pd.Timedelta(0))
    frame["implemented_weight"] = frame["implemented_weight"].where(consecutive, 0.0)
    return frame


def portfolio_returns(weights: pd.DataFrame, cost_bps: float = 3.5) -> pd.DataFrame:
    """Aggregate minute P&L and implementation costs to daily portfolio returns."""
    frame = weights.sort_values(["symbol", "timestamp_utc"], ignore_index=True).copy()
    base_gross = frame["implemented_weight"] * frame["return_1m"].fillna(0.0)
    base_daily = base_gross.groupby(frame["trading_date"]).sum()
    prior_vol = base_daily.rolling(20, min_periods=20).std().shift(1) * np.sqrt(252)
    risk_scale = (0.10 / prior_vol).clip(0.25, 2.0).fillna(1.0)
    frame["risk_scale"] = frame["trading_date"].map(risk_scale).fillna(1.0)
    frame["scaled_weight"] = frame["implemented_weight"] * frame["risk_scale"]

    daily_rows: list[dict[str, Any]] = []
    wealth = peak = 1.0
    prior_by_symbol: dict[str, float] = {}
    for date, day in frame.groupby("trading_date", sort=True):
        drawdown = 1.0 - wealth / peak
        drawdown_scale = (
            0.15
            if drawdown >= 0.15
            else 0.50
            if drawdown >= 0.10
            else 0.75
            if drawdown >= 0.075
            else 1.0
        )
        day = day.sort_values("timestamp_utc").copy()
        day["actual_weight"] = day["scaled_weight"] * drawdown_scale
        previous = day.groupby("symbol", sort=False)["actual_weight"].shift(1)
        previous = previous.fillna(day["symbol"].map(prior_by_symbol).fillna(0.0))
        day["turnover"] = (day["actual_weight"] - previous).abs() / 2.0
        gross = float((day["actual_weight"] * day["return_1m"].fillna(0.0)).sum())
        costs = float(day["turnover"].sum() * cost_bps / 10_000.0)
        net = gross - costs
        wealth *= 1.0 + net
        peak = max(peak, wealth)
        prior_by_symbol = cast(
            dict[str, float], day.groupby("symbol")["actual_weight"].last().to_dict()
        )
        daily_rows.append(
            {
                "trading_date": pd.Timestamp(cast(Any, date)),
                "gross_return": gross,
                "cost": costs,
                "net_return": net,
                "gross_exposure": float(
                    day.groupby("timestamp_utc")["actual_weight"]
                    .apply(lambda x: x.abs().sum())
                    .mean()
                ),
                "turnover": float(day["turnover"].sum()),
                "drawdown_scale": drawdown_scale,
            }
        )
    return pd.DataFrame(daily_rows).set_index("trading_date")


def summarize_walk_forward(daily: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw_fold in folds.to_dict("records"):
        fold = cast(dict[str, Any], raw_fold)
        sample = daily.loc[
            pd.Timestamp(fold["validation_start"]) : pd.Timestamp(fold["validation_end"])
        ]
        rows.append(
            {
                "fold": int(fold["fold"]),
                "sessions": len(sample),
                "annualized_return": float(sample["net_return"].mean() * 252),
                "annualized_volatility": annualized_volatility(sample["net_return"]),
                "sharpe": sharpe_ratio(sample["net_return"]),
                "maximum_drawdown": maximum_drawdown(sample["net_return"]),
                "average_daily_cost": float(sample["cost"].mean()),
                "average_daily_turnover": float(sample["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


def run(project_root: Path) -> None:
    """Build the combined intraday sleeve from development minute partitions only."""
    minute_root = project_root / "data/processed/research/development_minute_returns"
    pieces: list[pd.DataFrame] = []
    for path in sorted(minute_root.glob("symbol=*/returns.parquet")):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        featured = build_combined_score(build_intraday_features(frame))
        # Development hypothesis: intraday extremes continue; reversion is allowed only after
        # price rotates back into the two interior running-range quartiles.
        score_columns = [column for column in featured if str(column).startswith("score_")]
        featured.loc[~featured["range_quartile"].isin([2.0, 3.0]), score_columns] = 0.0
        pieces.append(featured)
    combined = pd.concat(pieces, ignore_index=True).sort_values(
        ["symbol", "timestamp_utc"], ignore_index=True
    )
    folds = pd.read_csv(project_root / "outputs/development_walk_forward_folds.csv")
    output = project_root / "data/processed/portfolios"
    output.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    daily_returns: dict[str, pd.Series] = {}
    weights: pd.DataFrame | None = None
    for score_column in (
        "score_balanced",
        "score_midpoint_dominant",
        "score_fast_reversal_dominant",
        "score_vwap_dominant",
    ):
        trial = combined.copy()
        trial["combined_score"] = trial[score_column]
        trial["position_signal"] = 0.0
        for _, indices in trial.groupby("symbol", sort=False).groups.items():
            trial.loc[indices, "position_signal"] = hysteresis_positions(
                trial.loc[indices, "combined_score"],
                trial.loc[indices, "trading_date"],
                trial.loc[indices, "session_minute"],
                entry=0.30,
                exit=0.10,
                maximum_minutes=60,
            )
        trial_weights = build_portfolio_weights(trial, rebalance_minutes=15)
        daily = portfolio_returns(trial_weights)
        name = score_column.removeprefix("score_")
        summaries.append(summarize_walk_forward(daily, folds).assign(score_variant=name))
        daily_returns[name] = daily["net_return"]
        if name == "balanced":
            weights = trial_weights
            daily.to_parquet(output / "intraday_mean_reversion_portfolio_daily.parquet")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(output / "intraday_mean_reversion_walk_forward.csv", index=False)
    pd.DataFrame(daily_returns).to_parquet(
        output / "intraday_mean_reversion_score_variations_daily.parquet"
    )
    if weights is None:
        raise RuntimeError("balanced portfolio weights were not built")
    weights["gross_contribution"] = weights["implemented_weight"] * weights["return_1m"].fillna(0.0)
    attribution = weights.groupby(["symbol", "trading_date"], observed=True).agg(
        average_score=("combined_score", "mean"),
        active_minutes=("position_signal", lambda x: int(x.ne(0.0).sum())),
        average_absolute_weight=("implemented_weight", lambda x: float(x.abs().mean())),
        gross_contribution=("gross_contribution", "sum"),
    )
    attribution.to_parquet(output / "intraday_mean_reversion_attribution.parquet")
    quartile = weights.groupby(["time_quartile", "range_quartile"], observed=True).agg(
        observations=("return_1m", "size"),
        active_fraction=("position_signal", lambda x: float(x.ne(0.0).mean())),
        average_absolute_score=("combined_score", lambda x: float(x.abs().mean())),
        gross_contribution=("gross_contribution", "sum"),
    )
    quartile.to_csv(output / "intraday_mean_reversion_quartile_attribution.csv")
    next_return = weights.groupby("symbol", sort=False)["return_1m"].shift(-1)
    component_rows = []
    for component in (
        "midpoint_score",
        "vwap_score",
        "open_move_score",
        "short_return_score",
        "combined_score",
    ):
        component_rows.append(
            {
                "component": component,
                "next_minute_information_coefficient": float(weights[component].corr(next_return)),
            }
        )
    pd.DataFrame(component_rows).to_csv(
        output / "intraday_mean_reversion_component_diagnostics.csv", index=False
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
