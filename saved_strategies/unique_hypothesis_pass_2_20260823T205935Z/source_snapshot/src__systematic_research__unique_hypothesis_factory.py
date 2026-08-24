"""Walk-forward research engine for distinct, registered futures hypotheses."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from systematic_research.daily_session_factory import (
    block_bootstrap_summary,
    capped_inverse_volatility_weights,
)
from systematic_research.intraday_strategy_search import resample_bars
from systematic_research.metrics import sharpe_ratio
from systematic_research.saved_strategy_runner import performance_snapshot
from systematic_research.vectorbt_strategy_factory import load_development_minutes

OUTPUT_ROOT = Path("data/processed/unique_hypothesis_factory")
TIMEFRAMES = (15, 30, 60, 240)
ECONOMIC_LEADERS = {
    "6E": "6J",
    "6J": "6E",
    "CL": "NG",
    "ES": "ZN",
    "GC": "HG",
    "HG": "GC",
    "NG": "CL",
    "NKD": "ES",
    "ZC": "CL",
    "ZN": "ES",
}


@dataclass(frozen=True)
class UniqueStrategySpec:
    """One parameter variant belonging to a registered hypothesis."""

    name: str
    hypothesis_id: str
    symbol: str
    timeframe_minutes: int
    lookback: int
    threshold: float
    direction: int
    leader: str | None = None


def _rolling_zscore(values: pd.DataFrame, history: int = 160) -> pd.DataFrame:
    mean = values.rolling(history, min_periods=40).mean().shift(1)
    scale = values.rolling(history, min_periods=40).std().shift(1).replace(0.0, np.nan)
    return values.sub(mean).div(scale).clip(-6.0, 6.0)


def _rolling_series_zscore(values: pd.Series, history: int = 160) -> pd.Series:
    """Compute the causal rolling z-score of a Series, including unnamed inputs."""
    column = "__value__"
    frame = values.to_frame(name=column)
    result = _rolling_zscore(frame, history=history)[column]
    result.name = values.name
    return result


def _daily_sum(values: pd.Series, dates: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    output = values.groupby(dates).sum().reindex(daily_index).fillna(0.0)
    output.index = pd.to_datetime(output.index)
    return output.astype(float)


def _intraday_trade_sides(position: pd.Series, dates: pd.Series) -> pd.Series:
    """Count actual position transitions and a forced close at each session end."""
    same_session_as_previous = dates.eq(dates.shift(1))
    previous_position = position.shift(1).where(same_session_as_previous, 0.0).fillna(0.0)
    transitions = position.sub(previous_position).abs()
    session_end = ~dates.eq(dates.shift(-1))
    return transitions.add(position.abs().where(session_end, 0.0))


def _append_variant(
    specs: list[UniqueStrategySpec],
    gross: dict[str, pd.Series],
    sides: dict[str, pd.Series],
    *,
    hypothesis_id: str,
    symbol: str,
    timeframe: int,
    lookback: int,
    threshold: float,
    direction: int,
    gross_return: pd.Series,
    trade_sides: pd.Series,
    leader: str | None = None,
) -> None:
    name = f"unique_{len(specs) + 1:05d}"
    specs.append(
        UniqueStrategySpec(
            name=name,
            hypothesis_id=hypothesis_id,
            symbol=symbol,
            timeframe_minutes=timeframe,
            lookback=lookback,
            threshold=threshold,
            direction=direction,
            leader=leader,
        )
    )
    gross[name] = gross_return
    sides[name] = trade_sides


def build_unique_population(
    minute_data: pd.DataFrame,
) -> tuple[list[UniqueStrategySpec], pd.DataFrame, pd.DataFrame]:
    """Build five distinct hypothesis families without accessing sealed observations."""
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    symbols = sorted(minute_data["symbol"].astype(str).unique())
    specs: list[UniqueStrategySpec] = []
    gross: dict[str, pd.Series] = {}
    sides: dict[str, pd.Series] = {}

    for timeframe in TIMEFRAMES:
        bars = resample_bars(minute_data, timeframe)
        bars = bars.sort_values(["symbol", "trading_date", "timestamp_utc"], ignore_index=True)
        open_panel = bars.pivot(index="timestamp_utc", columns="symbol", values="open").sort_index()
        close_panel = bars.pivot(
            index="timestamp_utc", columns="symbol", values="close"
        ).reindex_like(open_panel)
        volume_panel = bars.pivot(
            index="timestamp_utc", columns="symbol", values="volume"
        ).reindex_like(open_panel)
        date_panel = bars.pivot(
            index="timestamp_utc", columns="symbol", values="trading_date"
        ).reindex_like(open_panel)
        execution_return = close_panel.div(open_panel).sub(1.0)
        close_return = close_panel.pct_change(fill_method=None)
        same_session = date_panel.eq(date_panel.shift(1))

        # H005: declared economic leader transmits information to the target's next bar.
        for lookback in (1, 2, 4):
            leader_impulse = close_return.rolling(lookback, min_periods=lookback).sum()
            leader_score = _rolling_zscore(leader_impulse)
            for symbol in symbols:
                leader = ECONOMIC_LEADERS[symbol]
                if leader not in leader_score:
                    continue
                observed = leader_score[leader]
                for threshold in (0.5, 1.0):
                    for direction in (-1, 1):
                        position = (
                            np.sign(observed * direction)
                            .where(observed.abs().ge(threshold), 0.0)
                            .shift(1)
                            .where(same_session[symbol], 0.0)
                            .fillna(0.0)
                        )
                        dates = pd.to_datetime(date_panel[symbol])
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H005",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=lookback,
                            threshold=threshold,
                            direction=direction,
                            gross_return=_daily_sum(
                                position * execution_return[symbol].fillna(0.0),
                                dates,
                                daily_index,
                            ),
                            trade_sides=_daily_sum(
                                _intraday_trade_sides(position, dates), dates, daily_index
                            ),
                            leader=leader,
                        )

        # H006: own-price impulse conditional on causal volume surprise.
        log_volume = pd.DataFrame(
            np.log1p(volume_panel.clip(lower=0.0).to_numpy(dtype=float)),
            index=volume_panel.index,
            columns=volume_panel.columns,
        )
        volume_surprise = _rolling_zscore(log_volume).clip(lower=0.0)
        for lookback in (2, 4, 8):
            momentum_score = _rolling_zscore(
                close_return.rolling(lookback, min_periods=lookback).sum()
            )
            interaction = momentum_score * volume_surprise
            for symbol in symbols:
                observed = interaction[symbol]
                dates = pd.to_datetime(date_panel[symbol])
                for threshold in (0.5, 1.0):
                    for direction in (-1, 1):
                        position = (
                            np.sign(observed * direction)
                            .where(observed.abs().ge(threshold), 0.0)
                            .shift(1)
                            .where(same_session[symbol], 0.0)
                            .fillna(0.0)
                        )
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H006",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=lookback,
                            threshold=threshold,
                            direction=direction,
                            gross_return=_daily_sum(
                                position * execution_return[symbol].fillna(0.0),
                                dates,
                                daily_index,
                            ),
                            trade_sides=_daily_sum(
                                _intraday_trade_sides(position, dates), dates, daily_index
                            ),
                        )

        session_group = bars.groupby(["symbol", "trading_date"], sort=True)
        session = session_group.agg(
            session_open=("open", "first"),
            session_high=("high", "max"),
            session_low=("low", "min"),
            session_close=("close", "last"),
        ).reset_index()
        session["session_return"] = session["session_close"].div(session["session_open"]).sub(1.0)
        session["range_fraction"] = (
            session["session_high"].sub(session["session_low"]).div(session["session_open"])
        )
        bars["bar_return"] = bars["close"].div(bars["open"]).sub(1.0)
        bars["positive_variance"] = bars["bar_return"].clip(lower=0.0).pow(2)
        bars["negative_variance"] = bars["bar_return"].clip(upper=0.0).pow(2)
        semivariance = (
            bars.groupby(["symbol", "trading_date"], sort=True)[
                ["positive_variance", "negative_variance"]
            ]
            .sum()
            .reset_index()
        )
        session = session.merge(semivariance, on=["symbol", "trading_date"], validate="one_to_one")

        for symbol in symbols:
            product = session.loc[session["symbol"].eq(symbol)].set_index("trading_date")
            product.index = pd.to_datetime(product.index)
            target = product["session_return"].reindex(daily_index).fillna(0.0)
            total_variance = product["positive_variance"].add(product["negative_variance"])
            concentration = (
                product["positive_variance"]
                .sub(product["negative_variance"])
                .abs()
                .div(total_variance.replace(0.0, np.nan))
            )

            # H002: realized semivariance concentration conditions trailing direction.
            for lookback in (2, 5, 10, 20):
                trailing_direction = np.sign(
                    product["session_return"].rolling(lookback, min_periods=lookback).sum()
                )
                observed = (
                    trailing_direction
                    * concentration.rolling(lookback, min_periods=lookback).mean()
                )
                for threshold in (0.25, 0.50, 0.75):
                    for direction in (-1, 1):
                        position = (
                            np.sign(observed * direction)
                            .where(observed.abs().ge(threshold), 0.0)
                            .shift(1)
                            .reindex(daily_index)
                            .fillna(0.0)
                        )
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H002",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=lookback,
                            threshold=threshold,
                            direction=direction,
                            gross_return=position * target,
                            trade_sides=position.abs() * 2.0,
                        )

            product_bars = bars.loc[bars["symbol"].eq(symbol)].copy()
            product_bars["ordinal"] = product_bars.groupby("trading_date").cumcount()
            product_bars["reverse_ordinal"] = product_bars.groupby("trading_date").cumcount(
                ascending=False
            )
            first = product_bars.loc[product_bars["ordinal"].eq(0)].set_index("trading_date")
            last = product_bars.loc[product_bars["reverse_ordinal"].eq(0)].set_index("trading_date")
            first.index = pd.to_datetime(first.index)
            last.index = pd.to_datetime(last.index)

            # H003: a non-overlapping opening-window return predicts the closing window.
            opening_return = first["close"].div(first["open"]).sub(1.0)
            closing_return = last["close"].div(last["open"]).sub(1.0).reindex(daily_index)
            for lookback in (20, 60):
                opening_score = _rolling_series_zscore(opening_return, history=lookback * 2)
                for threshold in (0.0, 0.5, 1.0):
                    for direction in (-1, 1):
                        position = np.sign(opening_score * direction).where(
                            opening_score.abs().ge(threshold), 0.0
                        )
                        position = position.reindex(daily_index).fillna(0.0)
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H003",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=lookback,
                            threshold=threshold,
                            direction=direction,
                            gross_return=position * closing_return.fillna(0.0),
                            trade_sides=position.abs() * 2.0,
                        )

            # H004: a first-bar breakout is traded only after prior-session compression.
            prior_high = product["session_high"].shift(1)
            prior_low = product["session_low"].shift(1)
            first_direction = pd.Series(
                np.select(
                    [first["close"].gt(prior_high), first["close"].lt(prior_low)],
                    [1.0, -1.0],
                    default=0.0,
                ),
                index=first.index,
            )
            after_open_return = (
                last["close"]
                .div(
                    product_bars.loc[product_bars["ordinal"].eq(1)].set_index("trading_date")[
                        "open"
                    ]
                )
                .sub(1.0)
            )
            after_open_return.index = pd.to_datetime(after_open_return.index)
            for lookback in (10, 20, 40):
                prior_range = product["range_fraction"].shift(1)
                for compression_quantile in (0.20, 0.35):
                    cutoff = (
                        product["range_fraction"]
                        .rolling(lookback, min_periods=lookback)
                        .quantile(compression_quantile)
                        .shift(1)
                    )
                    compressed = prior_range.le(cutoff)
                    for direction in (-1, 1):
                        position = (first_direction * direction).where(compressed, 0.0)
                        position = position.reindex(daily_index).fillna(0.0)
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H004",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=lookback,
                            threshold=compression_quantile,
                            direction=direction,
                            gross_return=position
                            * after_open_return.reindex(daily_index).fillna(0.0),
                            trade_sides=position.abs() * 2.0,
                        )

    return specs, pd.DataFrame(gross, index=daily_index), pd.DataFrame(sides, index=daily_index)


def _binary_entropy(positive_fraction: pd.Series) -> pd.Series:
    """Return binary Shannon entropy on [0, 1], with deterministic edge values."""
    probability = positive_fraction.clip(0.0, 1.0)
    complement = 1.0 - probability
    first = pd.Series(0.0, index=probability.index)
    second = pd.Series(0.0, index=probability.index)
    positive_mask = probability.gt(0.0)
    complement_mask = complement.gt(0.0)
    first.loc[positive_mask] = probability.loc[positive_mask] * np.log2(
        probability.loc[positive_mask]
    )
    second.loc[complement_mask] = complement.loc[complement_mask] * np.log2(
        complement.loc[complement_mask]
    )
    return -(first + second)


def build_unique_population_pass2(
    minute_data: pd.DataFrame,
) -> tuple[list[UniqueStrategySpec], pd.DataFrame, pd.DataFrame]:
    """Build preregistered H007-H011 without recomputing rejected hypotheses."""
    daily_index = pd.DatetimeIndex(sorted(pd.to_datetime(minute_data["trading_date"]).unique()))
    symbols = sorted(minute_data["symbol"].astype(str).unique())
    specs: list[UniqueStrategySpec] = []
    gross: dict[str, pd.Series] = {}
    sides: dict[str, pd.Series] = {}

    # H007: cross-sectional high-to-price momentum decomposition.
    sessions = (
        minute_data.sort_values(["symbol", "trading_date", "timestamp_utc"])
        .groupby(["symbol", "trading_date"], sort=True)
        .agg(session_open=("open", "first"), session_close=("close", "last"))
        .reset_index()
    )
    session_open = sessions.pivot(
        index="trading_date", columns="symbol", values="session_open"
    ).reindex(daily_index)
    session_close = sessions.pivot(
        index="trading_date", columns="symbol", values="session_close"
    ).reindex(daily_index)
    session_target = session_close.div(session_open).sub(1.0).fillna(0.0)
    for lookback in (5, 10, 20, 40):
        prior_high = session_close.rolling(lookback, min_periods=lookback).max()
        high_to_price = session_close.div(prior_high).sub(1.0)
        cross_sectional_score = high_to_price.rank(axis=1, pct=True).sub(0.5).mul(2.0)
        for threshold in (0.2, 0.4):
            positions = (
                cross_sectional_score.where(cross_sectional_score.abs().ge(threshold), 0.0)
                .apply(np.sign)
                .shift(1)
                .fillna(0.0)
            )
            for symbol in symbols:
                position = positions[symbol]
                _append_variant(
                    specs,
                    gross,
                    sides,
                    hypothesis_id="H007",
                    symbol=symbol,
                    timeframe=1440,
                    lookback=lookback,
                    threshold=threshold,
                    direction=1,
                    gross_return=position * session_target[symbol],
                    trade_sides=position.abs() * 2.0,
                )

    for timeframe in TIMEFRAMES:
        bars = resample_bars(minute_data, timeframe).sort_values(
            ["symbol", "trading_date", "timestamp_utc"], ignore_index=True
        )
        bars["bar_return"] = bars["close"].div(bars["open"]).sub(1.0)
        for symbol in symbols:
            product = bars.loc[bars["symbol"].eq(symbol)].copy()
            dates = pd.to_datetime(product["trading_date"])
            target = product["bar_return"].fillna(0.0)
            same_session = dates.eq(dates.shift(1))

            # H008: causal historical drift for the same session clock bucket.
            for lookback in (20, 60):
                minimum = max(10, lookback // 2)
                grouped_return = product.groupby("session_bucket", sort=False)["bar_return"]
                prior_mean = grouped_return.transform(
                    lambda values, window=lookback, min_obs=minimum: (
                        values.rolling(window, min_periods=min_obs).mean().shift(1)
                    )
                )
                prior_scale = grouped_return.transform(
                    lambda values, window=lookback, min_obs=minimum: (
                        values.rolling(window, min_periods=min_obs).std().shift(1)
                    )
                ).replace(0.0, np.nan)
                score = prior_mean.div(prior_scale).clip(-6.0, 6.0)
                for threshold in (0.0, 0.10, 0.20):
                    position = (
                        score.where(score.abs().ge(threshold), 0.0).apply(np.sign).fillna(0.0)
                    )
                    _append_variant(
                        specs,
                        gross,
                        sides,
                        hypothesis_id="H008",
                        symbol=symbol,
                        timeframe=timeframe,
                        lookback=lookback,
                        threshold=threshold,
                        direction=1,
                        gross_return=_daily_sum(position * target, dates, daily_index),
                        trade_sides=_daily_sum(
                            _intraday_trade_sides(position, dates), dates, daily_index
                        ),
                    )

            # H009: response to discontinuous variation above a bipower proxy.
            absolute_return = product["bar_return"].abs()
            bipower_innovation = absolute_return.shift(1).mul(absolute_return.shift(2))
            for history in (40, 160):
                continuous_variance = (
                    bipower_innovation.rolling(history, min_periods=max(20, history // 2))
                    .mean()
                    .mul(np.pi / 2.0)
                    .replace(0.0, np.nan)
                )
                jump_variance = (
                    product["bar_return"].pow(2).sub(continuous_variance).clip(lower=0.0)
                )
                jump_score = (
                    np.sign(product["bar_return"]) * jump_variance.div(continuous_variance).pow(0.5)
                ).clip(-10.0, 10.0)
                for threshold in (1.0, 2.0):
                    for direction in (-1, 1):
                        position = (
                            np.sign(jump_score * direction)
                            .where(jump_score.abs().ge(threshold), 0.0)
                            .shift(1)
                            .where(same_session, 0.0)
                            .fillna(0.0)
                        )
                        _append_variant(
                            specs,
                            gross,
                            sides,
                            hypothesis_id="H009",
                            symbol=symbol,
                            timeframe=timeframe,
                            lookback=history,
                            threshold=threshold,
                            direction=direction,
                            gross_return=_daily_sum(position * target, dates, daily_index),
                            trade_sides=_daily_sum(
                                _intraday_trade_sides(position, dates), dates, daily_index
                            ),
                        )

            # H010: liquidity-demand shock, proxied by signed price impact.
            dollar_volume = product["close"].mul(product["volume"]).replace(0.0, np.nan)
            signed_impact = product["bar_return"].div(dollar_volume)
            for history in (80, 160):
                impact_score = _rolling_series_zscore(signed_impact, history=history)
                for threshold in (1.0, 2.0):
                    position = (
                        -np.sign(impact_score)
                        .where(impact_score.abs().ge(threshold), 0.0)
                        .shift(1)
                        .where(same_session, 0.0)
                        .fillna(0.0)
                    )
                    _append_variant(
                        specs,
                        gross,
                        sides,
                        hypothesis_id="H010",
                        symbol=symbol,
                        timeframe=timeframe,
                        lookback=history,
                        threshold=threshold,
                        direction=-1,
                        gross_return=_daily_sum(position * target, dates, daily_index),
                        trade_sides=_daily_sum(
                            _intraday_trade_sides(position, dates), dates, daily_index
                        ),
                    )

            # H011: low binary sign entropy identifies persistent information flow.
            positive = product["bar_return"].gt(0.0).astype(float)
            for lookback in (8, 16, 32):
                fraction_positive = positive.rolling(lookback, min_periods=lookback).mean()
                predictability = 1.0 - _binary_entropy(fraction_positive)
                recent_direction = np.sign(
                    product["bar_return"].rolling(lookback, min_periods=lookback).sum()
                )
                for threshold in (0.10, 0.25, 0.50):
                    position = (
                        recent_direction.where(predictability.ge(threshold), 0.0)
                        .shift(1)
                        .where(same_session, 0.0)
                        .fillna(0.0)
                    )
                    _append_variant(
                        specs,
                        gross,
                        sides,
                        hypothesis_id="H011",
                        symbol=symbol,
                        timeframe=timeframe,
                        lookback=lookback,
                        threshold=threshold,
                        direction=1,
                        gross_return=_daily_sum(position * target, dates, daily_index),
                        trade_sides=_daily_sum(
                            _intraday_trade_sides(position, dates), dates, daily_index
                        ),
                    )

    return specs, pd.DataFrame(gross, index=daily_index), pd.DataFrame(sides, index=daily_index)


def extended_snapshot(returns: pd.Series) -> dict[str, Any]:
    """Add weekly consistency to the standard daily/calendar metrics."""
    snapshot, _ = performance_snapshot(returns)
    weekly = (
        returns.sort_index().resample("W-FRI").apply(lambda values: (1.0 + values).prod() - 1.0)
    )
    snapshot.update(
        {
            "win_weeks": float((weekly > 0.0).mean()),
            "positive_weeks": int((weekly > 0.0).sum()),
            "weeks": len(weekly),
        }
    )
    return snapshot


def walk_forward_select(
    specs: list[UniqueStrategySpec],
    net_returns: pd.DataFrame,
    trade_sides: pd.DataFrame,
    *,
    folds: int = 4,
    per_hypothesis: int = 3,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Select family variants only on folds preceding each evaluation block."""
    registry = pd.DataFrame([asdict(spec) for spec in specs]).set_index("name")
    blocks = np.array_split(np.arange(len(net_returns)), folds)
    sleeve_oos = pd.DataFrame(
        0.0,
        index=net_returns.index,
        columns=sorted(registry["hypothesis_id"].unique()),
    )
    portfolio_oos = pd.Series(np.nan, index=net_returns.index, name="portfolio_return")
    selection_rows: list[dict[str, Any]] = []

    for evaluation_fold in range(1, folds):
        train_rows = np.concatenate(blocks[:evaluation_fold])
        evaluation_rows = blocks[evaluation_fold]
        train = net_returns.iloc[train_rows]
        training_sleeves: dict[str, pd.Series] = {}
        evaluation_sleeves: dict[str, pd.Series] = {}
        for hypothesis in sleeve_oos.columns:
            names = registry.index[registry["hypothesis_id"].eq(hypothesis)].tolist()
            ranked: list[tuple[float, str, list[float], int]] = []
            for name in names:
                prior_sharpes = [
                    sharpe_ratio(net_returns[name].iloc[block])
                    for block in blocks[:evaluation_fold]
                ]
                trades = int((trade_sides[name].iloc[train_rows] > 0.0).sum())
                if trades < max(12, 6 * evaluation_fold):
                    continue
                if not np.isfinite(prior_sharpes).all() or min(prior_sharpes) <= -0.50:
                    continue
                score = float(np.mean(prior_sharpes) - 0.25 * np.std(prior_sharpes))
                if score <= 0.0:
                    continue
                ranked.append((score, name, prior_sharpes, trades))
            ranked.sort(reverse=True)
            selected: list[str] = []
            selected_symbols: set[str] = set()
            for score, name, prior_sharpes, trades in ranked:
                symbol = str(registry.loc[name, "symbol"])
                if symbol in selected_symbols:
                    continue
                if selected:
                    correlation = train[selected].corrwith(train[name]).abs().max()
                    if pd.notna(correlation) and float(correlation) > 0.75:
                        continue
                selected.append(name)
                selected_symbols.add(symbol)
                selection_rows.append(
                    {
                        "evaluation_fold": evaluation_fold + 1,
                        "hypothesis_id": hypothesis,
                        "strategy": name,
                        "symbol": symbol,
                        "training_score": score,
                        "training_trades": trades,
                        "prior_fold_sharpes": json.dumps(prior_sharpes),
                    }
                )
                if len(selected) >= per_hypothesis:
                    break
            if selected:
                training_subset = train.reindex(columns=selected)
                evaluation_subset = net_returns.reindex(columns=selected)
                evaluation_subset = evaluation_subset.take(evaluation_rows, axis=0)
                training_sleeves[hypothesis] = training_subset.mean(axis="columns")
                evaluation_sleeves[hypothesis] = evaluation_subset.mean(axis="columns")

        if len(evaluation_sleeves) < 2:
            continue
        training_frame = pd.DataFrame(training_sleeves)
        weights = capped_inverse_volatility_weights(training_frame, maximum_weight=0.40)
        training_portfolio = training_frame.mul(weights, axis=1).sum(axis=1)
        training_volatility = float(training_portfolio.std(ddof=1) * np.sqrt(252.0))
        leverage = min(2.0, 0.10 / training_volatility) if training_volatility > 0.0 else 1.0
        evaluation_frame = pd.DataFrame(evaluation_sleeves)
        sleeve_oos.loc[evaluation_frame.index, evaluation_frame.columns] = evaluation_frame
        portfolio_oos.loc[evaluation_frame.index] = (
            evaluation_frame.mul(weights, axis=1).sum(axis=1) * leverage
        )
        for raw_hypothesis, weight in weights.items():
            hypothesis = str(raw_hypothesis)
            selection_rows.append(
                {
                    "evaluation_fold": evaluation_fold + 1,
                    "hypothesis_id": hypothesis,
                    "strategy": "__FAMILY_WEIGHT__",
                    "symbol": "",
                    "training_score": float(weight),
                    "training_trades": 0,
                    "prior_fold_sharpes": json.dumps({"leverage": leverage}),
                }
            )

    return sleeve_oos, portfolio_oos.dropna(), pd.DataFrame(selection_rows)


def apply_saved_walk_forward_selection(
    selections: pd.DataFrame,
    net_returns: pd.DataFrame,
    *,
    folds: int = 4,
) -> pd.Series:
    """Replay fixed fold selections under an alternative cost assumption."""
    blocks = np.array_split(np.arange(len(net_returns)), folds)
    output = pd.Series(np.nan, index=net_returns.index, name="portfolio_return")
    for evaluation_fold in range(2, folds + 1):
        rows = blocks[evaluation_fold - 1]
        fold_selection = selections.loc[selections["evaluation_fold"].eq(evaluation_fold)]
        weights = fold_selection.loc[fold_selection["strategy"].eq("__FAMILY_WEIGHT__")].set_index(
            "hypothesis_id"
        )["training_score"]
        leverage_records = fold_selection.loc[
            fold_selection["strategy"].eq("__FAMILY_WEIGHT__"), "prior_fold_sharpes"
        ]
        if weights.empty:
            continue
        leverage = float(json.loads(str(leverage_records.iloc[0]))["leverage"])
        family_returns: dict[str, pd.Series] = {}
        for hypothesis in weights.index:
            names = (
                fold_selection.loc[
                    fold_selection["hypothesis_id"].eq(hypothesis)
                    & ~fold_selection["strategy"].eq("__FAMILY_WEIGHT__"),
                    "strategy",
                ]
                .astype(str)
                .tolist()
            )
            family_subset = net_returns.reindex(columns=names)
            family_subset = family_subset.take(rows, axis=0)
            family_returns[str(hypothesis)] = family_subset.mean(axis="columns")
        frame = pd.DataFrame(family_returns)
        output.loc[frame.index] = frame.mul(weights, axis=1).sum(axis=1) * leverage
    return output.dropna()


def run_factory(
    project_root: Path, *, one_way_cost_bps: float = 3.5, research_pass: int = 2
) -> Path:
    """Run the registered unique-hypothesis walk-forward research pass."""
    started = perf_counter()
    minute_data, source_paths = load_development_minutes(project_root)
    if research_pass == 1:
        hypothesis_ids = ["H002", "H003", "H004", "H005", "H006"]
        specs, gross, sides = build_unique_population(minute_data)
    elif research_pass == 2:
        hypothesis_ids = ["H007", "H008", "H009", "H010", "H011"]
        specs, gross, sides = build_unique_population_pass2(minute_data)
    else:
        raise ValueError("research_pass must be 1 or 2")
    net = gross - sides * one_way_cost_bps / 10_000.0
    sleeves, portfolio, selections = walk_forward_select(specs, net, sides)
    snapshot = extended_snapshot(portfolio)

    fold_rows: list[dict[str, Any]] = []
    evaluation_folds = selections["evaluation_fold"].dropna().astype(int).unique()
    for fold in evaluation_folds:
        blocks = np.array_split(np.arange(len(net)), 4)
        fold_return = portfolio.reindex(net.index[blocks[fold - 1]]).dropna()
        if not fold_return.empty:
            fold_rows.append({"fold": fold, **extended_snapshot(fold_return)})

    cost_rows: list[dict[str, Any]] = []
    for cost in (0.0, 0.5, 1.0, 2.0, 3.5, 5.0):
        stressed_net = gross - sides * cost / 10_000.0
        stressed_portfolio = apply_saved_walk_forward_selection(selections, stressed_net)
        cost_rows.append({"one_way_cost_bps": cost, **extended_snapshot(stressed_portfolio)})

    outlier_rows: list[dict[str, Any]] = []
    for removed in (0, 1, 3, 5, 10):
        stressed = portfolio if removed == 0 else portfolio.drop(portfolio.nlargest(removed).index)
        outlier_rows.append({"best_days_removed": removed, **extended_snapshot(stressed)})
    bootstrap = block_bootstrap_summary(portfolio, block_length=20, samples=5000, seed=20260823)
    audit_gates = {
        "walk_forward_sharpe_at_least_2": float(snapshot["sharpe"]) >= 2.0,
        "winning_weeks_or_months_at_least_70_percent": max(
            float(snapshot["win_weeks"]), float(snapshot["win_months"])
        )
        >= 0.70,
        "every_evaluation_fold_positive": all(float(row["sharpe"]) > 0.0 for row in fold_rows),
        "sharpe_at_3_5_bps_at_least_1_5": float(
            next(row["sharpe"] for row in cost_rows if row["one_way_cost_bps"] == 3.5)
        )
        >= 1.5,
        "bootstrap_fifth_percentile_sharpe_positive": bootstrap["sharpe_p05"] > 0.0,
        "sharpe_after_removing_best_three_days_positive": float(
            next(row["sharpe"] for row in outlier_rows if row["best_days_removed"] == 3)
        )
        > 0.0,
        "at_least_two_unique_hypotheses_selected_each_fold": all(
            group.loc[~group["strategy"].eq("__FAMILY_WEIGHT__"), "hypothesis_id"].nunique() >= 2
            for _, group in selections.groupby("evaluation_fold")
        ),
    }

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = project_root / OUTPUT_ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([asdict(spec) for spec in specs]).to_csv(
        output / "strategy_registry.csv", index=False
    )
    gross.to_parquet(output / "all_strategy_gross_returns.parquet")
    sides.to_parquet(output / "all_strategy_trade_sides.parquet")
    net.to_parquet(output / "all_strategy_net_returns.parquet")
    sleeves.to_parquet(output / "walk_forward_family_sleeves.parquet")
    portfolio.to_frame().to_parquet(output / "walk_forward_portfolio_returns.parquet")
    selections.to_csv(output / "walk_forward_selections.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "fold_audit.csv", index=False)
    pd.DataFrame(cost_rows).to_csv(output / "cost_stress.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(output / "outlier_stress.csv", index=False)
    (output / "bootstrap.json").write_text(json.dumps(bootstrap, indent=2), encoding="utf-8")
    (output / "audit_gates.json").write_text(json.dumps(audit_gates, indent=2), encoding="utf-8")
    manifest = {
        "research_stage": "development_only_walk_forward",
        "sealed_year_accessed": False,
        "research_pass": research_pass,
        "unique_hypotheses": hypothesis_ids,
        "raw_parameter_configurations": len(specs),
        "parameter_configurations_are_not_counted_as_unique_ideas": True,
        "base_one_way_cost_bps": one_way_cost_bps,
        "timeframes_minutes": list(TIMEFRAMES),
        "walk_forward_folds": 4,
        "evaluation_folds": 3,
        "portfolio": snapshot,
        "bootstrap": bootstrap,
        "audit_gates": audit_gates,
        "audit_passed": all(audit_gates.values()),
        "development_inputs": [path.relative_to(project_root).as_posix() for path in source_paths],
        "elapsed_seconds": perf_counter() - started,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    latest = project_root / OUTPUT_ROOT / "latest_run.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        json.dumps({"run_id": run_id, "path": str(output)}, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"Saved run: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unique-hypothesis walk-forward research")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--one-way-cost-bps", type=float, default=3.5)
    parser.add_argument("--research-pass", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run_factory(
        args.project_root.resolve(),
        one_way_cost_bps=args.one_way_cost_bps,
        research_pass=args.research_pass,
    )


if __name__ == "__main__":
    main()
