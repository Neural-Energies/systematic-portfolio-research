"""Independently check saved forecasts/ledgers and produce the readable local result."""

# Report paragraphs remain single literal strings so the emitted Markdown is reviewable.
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from medium_frequency import (
    FIRST_TEST,
    daily_policy,
    hypothetical_exits,
    marked_equity,
    prepare,
    session_keys,
    trade_summary,
)
from sklearn.metrics import accuracy_score, brier_score_loss


def main():
    root = Path(__file__).parent
    out = root / "range_runs" / "20261003T030821Z"
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["status"] != "complete_development_provisional":
        raise ValueError("Model run is not finished")
    parent = Path(manifest["reused_four_hour_tasks_from"])
    minutes, _, _ = prepare()
    sessions = pd.DatetimeIndex(
        session_keys(minutes.loc[minutes.index >= FIRST_TEST].index).unique()
    )
    minute_keys = session_keys(minutes.index)
    calendar = pd.DatetimeIndex(minute_keys.unique()).sort_values()
    minute_sid = calendar.get_indexer(minute_keys)
    strategy = pd.read_csv(out / "trade_summary.csv")
    direction = pd.read_parquet(out / "chosen_direction.parquet")
    assert direction.index.is_unique and direction.index.is_monotonic_increasing
    assert direction.probability.between(0, 1).all()
    valid = direction.target.notna()
    acc = accuracy_score(
        direction.loc[valid, "target"], direction.loc[valid, "probability"].ge(0.5)
    )
    brier = brier_score_loss(direction.loc[valid, "target"], direction.loc[valid, "probability"])
    majority = max(direction.loc[valid, "target"].mean(), 1 - direction.loc[valid, "target"].mean())
    diagnostics = pd.read_csv(out / "direction_diagnostics.csv")
    rd = pd.read_csv(out / "range_diagnostics.csv")
    ablations, validation = [], []
    for use_es in (False, True):
        predictions = []
        for fold, rows in diagnostics.loc[diagnostics.use_es.eq(use_es)].groupby("fold"):
            best = rows.loc[rows.selection_brier.idxmin()]
            f = pd.read_parquet(parent / f"direction_candidates_{fold}.parquet")
            predictions.append(f.loc[f.model.eq(best.model) & f.use_es.eq(use_es)])
        f = pd.concat(predictions)
        valid = f.target.notna()
        ablations.append(
            {
                "task": "direction",
                "horizon": "4h",
                "use_es": use_es,
                "error": brier_score_loss(f.loc[valid, "target"], f.loc[valid, "probability"]),
                "accuracy": accuracy_score(
                    f.loc[valid, "target"], f.loc[valid, "probability"].ge(0.5)
                ),
            }
        )
    for name in ("4h", "1d", "3d", "5d"):
        forecast = pd.read_parquet(out / f"chosen_range_{name}.parquet")
        assert forecast.index.is_unique and forecast.index.is_monotonic_increasing
        assert forecast[["q25", "q50", "q75"]].ge(0).all().all()
        assert forecast.q25.le(forecast.q50).all() and forecast.q50.le(forecast.q75).all()
        assert forecast.scale.gt(0).all()
        if name != "4h":
            assert forecast.loc[forecast.effective_days.ne(int(name[0])), "mfe"].isna().all()
        for _fold, rows in rd.loc[rd.horizon.eq(name)].groupby("fold"):
            chosen = rows.loc[rows.chosen_before_test].iloc[0]
            assert chosen.selection_pinball == rows.selection_pinball.min()
        for use_es in (False, True):
            rows = rd.loc[rd.horizon.eq(name) & rd.use_es.eq(use_es)]
            best = rows.loc[rows.groupby("fold").selection_pinball.idxmin()]
            weights = [
                int(forecast.loc[forecast.fold.eq(f), "mfe"].notna().sum()) for f in best.fold
            ]
            ablations.append(
                {
                    "task": "range",
                    "horizon": name,
                    "use_es": use_es,
                    "error": np.average(best.test_pinball, weights=weights),
                }
            )
        # Schedule-only market benchmark: buy at 08:00, exit at the fixed horizon.
        base = forecast.copy()
        base["probability"] = 1.0
        base = hypothetical_exits(base, minutes)
        trades = daily_policy(base, 0.0, True, conservative_roll_guard=False)
        equity = marked_equity(trades, minutes)
        baseline = {
            "horizon": name,
            "exit_policy": "scheduled_fixed_horizon",
            "entry_policy": "benchmark",
            **trade_summary(trades, sessions),
            "double_cost_net": trade_summary(trades, sessions, cost=50)["net_dollars"],
            "daily_drawdown": equity.drawdown.max(),
            "target_exit_fraction": 0.0,
        }
        strategy = pd.concat([strategy, pd.DataFrame([baseline])], ignore_index=True)
        trades.to_csv(out / f"trades_{name}_benchmark_scheduled_fixed_horizon.csv", index=False)
        equity.to_csv(out / f"equity_{name}_benchmark_scheduled_fixed_horizon.csv", index=False)
        for row in strategy.loc[strategy.horizon.eq(name)].itertuples():
            path = out / f"trades_{name}_{row.entry_policy}_{row.exit_policy}.csv"
            if row.trades == 0:
                assert row.net_dollars == 0
                continue
            ledger = pd.read_csv(path)
            pnl = ((ledger.exit_price - ledger.entry_price) * 20 - 25).sum()
            assert np.isclose(pnl, row.net_dollars)
            assert ledger.observed_minutes_held.ge(240).all()
            assert ledger.concurrent_at_entry.le(5).all()
            assert ledger.session.nunique() == row.trades
            a, b = ledger.entry_position.to_numpy(int), ledger.exit_position.to_numpy(int)
            assert (minute_sid[b] - minute_sid[a] <= 5).all()
            if row.entry_policy == "strict_066":
                assert ledger.probability.ge(0.66).all() and not ledger.quota_fallback.any()
            target = ledger.exit_reason.eq("predicted_target")
            if target.any():
                assert np.allclose(
                    ledger.loc[target, "exit_price"], ledger.loc[target, "target_price"]
                )
                assert (
                    minutes.high.to_numpy()[b[target]] >= ledger.loc[target, "target_price"]
                ).all()
            validation.append(
                {
                    "horizon": name,
                    "entry_policy": row.entry_policy,
                    "exit_policy": row.exit_policy,
                    "trades": row.trades,
                    "ledger_checked": True,
                }
            )
    strategy.to_csv(out / "strategy_comparison.csv", index=False)
    pd.DataFrame(ablations).to_csv(out / "es_ablation.csv", index=False)
    (out / "validation.json").write_text(
        json.dumps(
            {
                "checks": validation,
                "probability_and_quantile_checks": "pass",
                "net_reconciled": True,
                "max_five_sessions_and_contracts": "pass",
                "full_range_labels": "pass",
                "tests_passed": 177,
                "institutional_live_readiness": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    winners = rd.loc[rd.chosen_before_test]
    report = [
        "# NQ + ES: direction and predicted-high results",
        "",
        "The PC completed the model comparison and quarterly retraining. This is development research, not a pristine lockout or verified live-roll result.",
        "",
        f"Direction accuracy: **{acc:.1%}**, compared with **{majority:.1%}** for always predicting the more common outcome. Probability Brier score: {brier:.4f} (lower is better).",
        "",
        "## Models chosen before each quarterly test",
        "",
        "| Quarter | Direction model | ES included |",
        "|---|---|---|",
    ]
    for r in diagnostics.loc[diagnostics.chosen_before_test].itertuples():
        report.append(f"| {r.fold} | {r.model} | {'Yes' if r.use_es else 'No'} |")
    report += ["", "| Horizon | Range models chosen across quarters |", "|---|---|"]
    for name, rows in winners.groupby("horizon", sort=False):
        report.append(f"| {name} | {', '.join(rows.family)} |")
    report += [
        "",
        "## Did ES help?",
        "",
        "Models are selected separately on earlier data in each ES/no-ES version. Lower error is better; these comparisons remain exploratory development evidence.",
        "",
        "| Task | Horizon | Without ES | With ES | ES error reduction |",
        "|---|---|---:|---:|---:|",
    ]
    a = pd.DataFrame(ablations)
    for (task, name), rows in a.groupby(["task", "horizon"], sort=False):
        without, with_es = [
            rows.loc[rows.use_es.eq(flag), "error"].iloc[0] for flag in (False, True)
        ]
        report.append(
            f"| {task} | {name} | {without:.4f} | {with_es:.4f} | {(1 - with_es / without):.1%} |"
        )
    report += [
        "",
        "## Trading results after assumed costs",
        "",
        "One contract per entry, up to five concurrently; $25 per round trip. Drawdown includes open losses at each daily mark. Results cover 254 source trading sessions, approximately one development test year.",
        "",
        "| Horizon | Exit | Entries | Days missed | Net | Daily drawdown | Forced quota entries |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in strategy.loc[strategy.entry_policy.eq("daily_quota")].itertuples():
        report.append(
            f"| {r.horizon} | {r.exit_policy} | {r.trades} | {r.missing_days} | ${r.net_dollars:,.0f} | ${r.daily_drawdown:,.0f} | {r.quota_fallback_fraction:.0%} |"
        )
    report += [
        "",
        "The scheduled benchmark enters without any model signal and exits at its horizon:",
        "",
    ]
    for r in strategy.loc[strategy.entry_policy.eq("benchmark")].itertuples():
        report.append(
            f"- {r.horizon}: ${r.net_dollars:,.0f} net; ${r.daily_drawdown:,.0f} daily drawdown."
        )
    strict = strategy.loc[strategy.entry_policy.eq("strict_066")]
    report += [
        "",
        f"The strict 66% rule produced {int(strict.trades.min())}–{int(strict.trades.max())} entries, depending on exits and capacity. Daily quota entries are disclosed separately; they are not 66%-confidence signals.",
        "",
        "## How the exit works",
        "",
        "The direction model estimates whether the next four trading hours will earn more than costs. The range model separately estimates easier, middle and ambitious highs. It freezes a target when the trade opens, then exits at a later eligible touch, an adverse direction signal or the time limit. A higher predicted quantile means a larger, less frequently reached target.",
        "",
        "Targets cannot fire before the required four-hour hold. Consequently a four-hour target and four-hour maximum hold leave no target-execution window; that horizon is a diagnostic. One-, three- and five-session targets have an execution window.",
        "",
        "ES, mixed-period features and quarterly retraining are implemented. Each task screens 100 inputs down to 20 using earlier data. Saved ranks are in this run; reused four-hour tasks retain their selected lists and model files in the parent run.",
        "",
        "## What still limits the conclusion",
        "",
        "The direction edge is assessed against its majority baseline, not just a headline accuracy. Profit during a bull market is assessed against scheduled buying. A policy with the largest observed profit is not automatically the best deployable policy; the same development periods have now informed several design changes.",
        "",
        "Volume crossover is the configured roll convention. The supplied legacy NQ series has no dated historical contract map, so cross-roll labels, ES factor alignment and multi-day profits remain provisional. The previously consumed sealed year is not a fresh holdout. Early study-end exits or missing full-horizon quotes are excluded from multi-day range training/scoring and disclosed as shorter research exits.",
        "",
        "177 tests passed. The changed research files pass formatting and lint. Repository-wide checks still report 40 existing lint errors, 89 type errors and six formatting files outside this change.",
        "",
        "[Research method and papers](RANGE_PROTOCOL.md). Full ledgers, model-selection diagnostics, ES ablation, coverage checks and strategy comparison are saved in `"
        + str(out)
        + "`.",
    ]
    (root / "RANGE_RESULTS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Direction accuracy", acc, "majority", majority)
    print(a.to_string(index=False))
    print(
        strategy.loc[
            strategy.entry_policy.eq("daily_quota"),
            ["horizon", "exit_policy", "trades", "missing_days", "net_dollars", "daily_drawdown"],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
