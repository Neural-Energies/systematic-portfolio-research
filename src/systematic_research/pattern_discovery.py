"""Resumable, declared-space pattern discovery for research only.

Candidates are evaluated strictly on development data.  The sealed holdout is
deliberately neither read nor referenced by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp
from statsmodels.tsa.stattools import coint  # type: ignore[import-untyped]

from systematic_research.metrics import maximum_drawdown, sharpe_ratio

DEVELOPMENT_ROOT = Path("data/processed/databento_research/development_minute_returns")


@dataclass(frozen=True)
class Candidate:
    family: str
    instruments: tuple[str, ...]
    parameters: dict[str, float | int]

    @property
    def fingerprint(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True, default=list).encode()
        return hashlib.sha256(value).hexdigest()


def load_settings(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        return cast(dict[str, Any], yaml.safe_load(source))


def enumerate_candidates(settings: dict[str, Any]) -> list[Candidate]:
    """Fully enumerate the declared grid; no random sampling or early stopping."""
    symbols = [str(value) for value in settings["universe"]]
    families = settings["families"]
    candidates: list[Candidate] = []
    for window, threshold in itertools.product(
        families["mean_reversion_zscore"]["windows"],
        families["mean_reversion_zscore"]["thresholds"],
    ):
        candidates.extend(
            Candidate("mean_reversion_zscore", (s,), {"window": window, "threshold": threshold})
            for s in symbols
        )
    for window, threshold in itertools.product(
        families["trend_momentum"]["windows"], families["trend_momentum"]["thresholds"]
    ):
        candidates.extend(
            Candidate("trend_momentum", (s,), {"window": window, "threshold": threshold})
            for s in symbols
        )
    for window in families["breakout"]["windows"]:
        candidates.extend(Candidate("breakout", (s,), {"window": window}) for s in symbols)
    for first, second in itertools.combinations(symbols, 2):
        for lag, threshold in itertools.product(
            families["lead_lag"]["lags"], families["lead_lag"]["thresholds"]
        ):
            candidates.append(
                Candidate("lead_lag", (first, second), {"lag": lag, "threshold": threshold})
            )
        for lookback, entry_zscore in itertools.product(
            families["cointegration"]["lookbacks"], families["cointegration"]["entry_zscores"]
        ):
            candidates.append(
                Candidate(
                    "cointegration",
                    (first, second),
                    {"lookback": lookback, "entry_zscore": entry_zscore},
                )
            )
    return candidates


def initialize_ledger(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS trials (
        fingerprint TEXT PRIMARY KEY, candidate_json TEXT NOT NULL, status TEXT NOT NULL,
        result_json TEXT, created_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    return connection


def load_returns(symbol: str, bar_minutes: int) -> pd.Series:
    path = DEVELOPMENT_ROOT / f"symbol={symbol}" / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path).set_index("timestamp_utc").sort_index()
    return (
        frame["close"]
        .resample(f"{bar_minutes}min")
        .last()
        .pct_change(fill_method=None)
        .rename(symbol)
    )


def candidate_returns(
    candidate: Candidate, returns: pd.DataFrame, cost_bps_one_way: float = 1.5
) -> pd.Series | None:
    """Create causal, fully specified positions from a declarative candidate."""
    first = returns[candidate.instruments[0]]
    p = candidate.parameters
    if candidate.family == "mean_reversion_zscore":
        rolling = first.rolling(int(p["window"]), min_periods=int(p["window"])).agg(["mean", "std"])
        raw_signal = pd.Series(
            np.sign((first - rolling["mean"]) / rolling["std"].replace(0.0, np.nan)),
            index=first.index,
        )
        signal = -raw_signal.where(
            ((first - rolling["mean"]) / rolling["std"].replace(0.0, np.nan)).abs()
            >= float(p["threshold"]),
            0.0,
        )
    elif candidate.family == "trend_momentum":
        signal = pd.Series(np.sign(first.rolling(int(p["window"])).sum()), index=first.index).where(
            first.rolling(int(p["window"])).sum().abs()
            >= first.rolling(int(p["window"])).std() * float(p["threshold"]),
            0.0,
        )
    elif candidate.family == "breakout":
        levels = first.rolling(int(p["window"])).sum().shift(1)
        signal = pd.Series(np.sign(first), index=first.index).where(
            first.abs() >= levels.abs() / max(int(p["window"]), 1), 0.0
        )
    elif candidate.family == "lead_lag":
        leader = returns[candidate.instruments[0]].shift(int(p["lag"]))
        threshold = leader.rolling(32).std() * float(p["threshold"])
        signal = pd.Series(np.sign(leader), index=leader.index).where(
            leader.abs() >= threshold, 0.0
        )
        first = returns[candidate.instruments[1]]
    elif candidate.family == "cointegration":
        second = returns[candidate.instruments[1]]
        prices = (1.0 + pd.concat([first, second], axis=1).fillna(0.0)).cumprod()
        if len(prices.dropna()) < int(p["lookback"]):
            return None
        if (
            coint(
                prices.iloc[:, 0].dropna(),
                prices.iloc[:, 1].reindex(prices.iloc[:, 0].dropna().index).dropna(),
            )[1]
            > 0.05
        ):
            return None
        spread = prices.iloc[:, 0] - prices.iloc[:, 1]
        zscore = (spread - spread.rolling(int(p["lookback"])).mean()) / spread.rolling(
            int(p["lookback"])
        ).std()
        signal = -pd.Series(np.sign(zscore), index=zscore.index).where(
            zscore.abs() >= float(p["entry_zscore"]), 0.0
        )
        first = (first - second) / 2.0
    else:
        raise ValueError(candidate.family)
    position = signal.shift(1).fillna(0.0)
    costs = position.diff().abs().fillna(0.0)
    return pd.Series(
        position * first - costs * cost_bps_one_way / 10_000.0,
        index=first.index,
        name=candidate.fingerprint,
    )


def split_metrics(returns: pd.Series, minimum_occurrences: int) -> dict[str, float] | None:
    clean = returns.dropna()
    active = clean[clean.ne(0.0)]
    if len(active) < minimum_occurrences:
        return None
    half = len(clean) // 2
    discovery, confirmation = clean.iloc[:half], clean.iloc[half:]
    pvalue = float(ttest_1samp(active, 0.0, alternative="greater").pvalue)
    return {
        "occurrences": float(len(active)),
        "discovery_sharpe": sharpe_ratio(discovery, periods_per_year=252 * 96),
        "confirmation_sharpe": sharpe_ratio(confirmation, periods_per_year=252 * 96),
        "mean_return_per_occurrence": float(active.mean()),
        "max_drawdown": maximum_drawdown(clean),
        "raw_pvalue": pvalue,
    }


def bh_adjust(pvalues: pd.Series, trials: int) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values against every attempted trial."""
    ordered = pvalues.sort_values()
    ranks = np.arange(1, len(ordered) + 1)
    adjusted = (ordered * trials / ranks).iloc[::-1].cummin().clip(upper=1.0)
    return pd.Series(adjusted.reindex(pvalues.index), index=pvalues.index)


def diversify_survivors(
    streams: pd.DataFrame, ranking: pd.Series, maximum_pair_correlation: float
) -> tuple[pd.Series, pd.DataFrame]:
    """Keep only uncorrelated validated sleeves, then apply inverse-volatility weights."""
    accepted: list[str] = []
    for name in ranking.sort_values(ascending=False).index:
        if name not in streams:
            continue
        if (
            not accepted
            or streams[[name, *accepted]].corr().loc[name, accepted].abs().max()
            <= maximum_pair_correlation
        ):
            accepted.append(name)
    selected = streams[accepted].dropna(how="all")
    if selected.empty:
        return pd.Series(dtype=float, name="candidate_portfolio"), selected
    vol = selected.std(ddof=1).replace(0.0, np.nan)
    weights = (1.0 / vol).dropna()
    weights /= weights.sum()
    return selected[weights.index].mul(weights, axis=1).sum(axis=1).rename(
        "candidate_portfolio"
    ), selected


def run(settings: dict[str, Any], ledger_path: Path, budget_seconds: float) -> pd.DataFrame:
    candidates = enumerate_candidates(settings)
    connection = initialize_ledger(ledger_path)
    returns = pd.concat(
        [load_returns(symbol, int(settings["bar_minutes"])) for symbol in settings["universe"]],
        axis=1,
    )
    started = monotonic()
    for candidate in candidates:
        if monotonic() - started >= budget_seconds:
            break
        known = connection.execute(
            "SELECT 1 FROM trials WHERE fingerprint = ?", (candidate.fingerprint,)
        ).fetchone()
        if known:
            continue
        result = candidate_returns(candidate, returns, float(settings["cost_bps_one_way"]))
        metrics = (
            None
            if result is None
            else split_metrics(result, int(settings["minimum_independent_occurrences"]))
        )
        status = "insufficient_evidence" if metrics is None else "evaluated"
        connection.execute(
            "INSERT INTO trials(fingerprint,candidate_json,status,result_json) VALUES(?,?,?,?)",
            (
                candidate.fingerprint,
                json.dumps(asdict(candidate), default=list),
                status,
                None if metrics is None else json.dumps(metrics),
            ),
        )
        connection.commit()
    completed = int(connection.execute("SELECT COUNT(*) FROM trials").fetchone()[0])
    rows = connection.execute(
        "SELECT fingerprint,candidate_json,result_json FROM trials WHERE status='evaluated'"
    ).fetchall()
    connection.close()
    if not rows:
        return pd.DataFrame()
    report = pd.DataFrame(
        [
            {**json.loads(candidate_json), **json.loads(result_json), "fingerprint": fingerprint}
            for fingerprint, candidate_json, result_json in rows
        ]
    )
    report["fdr_adjusted_pvalue"] = bh_adjust(report["raw_pvalue"], len(candidates))
    minimum = float(settings["minimum_discovery_sharpe"])
    tolerance = float(settings["confirmation_sharpe_tolerance"])
    report["qualifies_development"] = (
        (report["discovery_sharpe"] >= minimum)
        & (report["confirmation_sharpe"] >= minimum - tolerance)
        & (report["fdr_adjusted_pvalue"] <= 0.05)
    )
    report["total_declared_trials"] = len(candidates)
    report["completed_trials"] = completed
    report["coverage_percent"] = 100.0 * completed / len(candidates)
    report["holdout_accessed"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable development-only exhaustive pattern discovery."
    )
    parser.add_argument("--config", type=Path, default=Path("config/pattern_discovery.yaml"))
    parser.add_argument("--ledger", type=Path, default=Path("work/pattern_discovery/trials.sqlite"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/pattern_discovery/candidate_report.csv")
    )
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    args = parser.parse_args()
    settings = load_settings(args.config)
    report = run(settings, args.ledger, args.budget_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    survivors = report.loc[report["qualifies_development"]].copy()
    portfolio_path = args.output.with_name("candidate_portfolio_summary.csv")
    if not survivors.empty:
        returns = pd.concat(
            [load_returns(symbol, int(settings["bar_minutes"])) for symbol in settings["universe"]],
            axis=1,
        )
        streams: dict[str, pd.Series] = {}
        definitions = {
            candidate.fingerprint: candidate for candidate in enumerate_candidates(settings)
        }
        for fingerprint in survivors["fingerprint"].astype(str):
            definition = definitions[fingerprint]
            series = candidate_returns(definition, returns, float(settings["cost_bps_one_way"]))
            if series is not None:
                streams[fingerprint] = series
        portfolio, constituents = diversify_survivors(
            pd.DataFrame(streams),
            survivors.set_index("fingerprint")["confirmation_sharpe"],
            float(settings["maximum_pair_correlation"]),
        )
        if not portfolio.empty:
            pd.DataFrame({"portfolio_return": portfolio}).to_parquet(
                args.output.with_name("candidate_portfolio_returns.parquet")
            )
            constituents.corr().to_csv(
                args.output.with_name("candidate_portfolio_correlations.csv")
            )
            pd.DataFrame(
                [
                    {
                        "constituents": len(constituents.columns),
                        "sharpe": sharpe_ratio(portfolio, 252 * 96),
                        "max_drawdown": maximum_drawdown(portfolio),
                        "holdout_accessed": False,
                    }
                ]
            ).to_csv(portfolio_path, index=False)
    if not portfolio_path.exists():
        pd.DataFrame(
            [{"constituents": 0, "status": "no validated sleeves", "holdout_accessed": False}]
        ).to_csv(portfolio_path, index=False)
    print(f"Logged {len(report):,} evaluated trials; output={args.output}; holdout_accessed=False")


if __name__ == "__main__":
    main()
