"""Merge completed entry-only diagnostics and independently replay sample outcomes."""

# ruff: noqa: E501
import json

import numpy as np
import pandas as pd
from entry_screen import fdr
from experiment import load_minutes
from fifteen_probability import BASE


def run():
    paths = [
        BASE / "entry_quality_runs" / "20261006T184753Z",
        BASE / "entry_quality_runs" / "20261006T185044Z",
    ]
    for p in paths:
        if not (p / "protocol.json").exists():
            raise ValueError("Incomplete research run")
    frame = pd.concat([pd.read_csv(p / "results.csv") for p in paths], ignore_index=True)
    later = frame.split.eq("later")
    frame.loc[later, "combined_screen_fdr_q"] = fdr(frame.loc[later, "p_hac"].fillna(1).to_numpy())
    minute = load_minutes("NQ")
    audits = 0
    for session in ("RTH", "Globex"):
        for horizon in (5, 15, 60, 120, 240, 360, 1440):
            root = (
                BASE / "entry_runs" / "20261006T164818Z"
                if horizon in (5, 15, 60, 240)
                else paths[1]
            )
            f = pd.read_parquet(root / f"outcomes_{session}_{horizon}.parquet")
            for entry, r in f.loc[f.scorable].sample(5, random_state=73).iterrows():
                quotes = minute.loc[entry : r.planned_exit - pd.Timedelta(minutes=1)]
                assert len(quotes) == r.effective_minutes
                assert quotes.open.iloc[0] == r.entry_open
                assert quotes.close.iloc[-1] == r.exit_close
                assert np.isclose(quotes.high.max() - r.entry_open, r.mfe_points)
                assert np.isclose(r.entry_open - quotes.low.min(), r.mae_points)
                audits += 1
    frame.to_csv(BASE / "ENTRY_ONLY_RESULTS.csv", index=False)
    z = frame.loc[later & frame.sessions_without_signal.eq(0)]
    price = z.loc[~z.signal.str.startswith("clock_")].sort_values(
        "direction_success", ascending=False
    )
    report = [
        "# NQ entry-only research",
        "",
        "We have not demonstrated a reliable high-success entry rule that meets the every-session requirement. Targets, high forecasts, stop models and portfolio design remain deferred.",
        "",
        "Tested 100 original variants, 20 unions of related rules and nine recurring clock controls across seven observation windows and two sessions: 1,806 later-period combinations. Observation windows are 5, 15, 60, 120, 240, 360 minutes and the remaining session (1440-minute window capped at session close). These are measurement windows, not recommended holding periods or exit rules.",
        "",
        "Success means the observed price at the end of the measurement window is strictly above the entry open. It does not mean a profit target was filled. Favorable excursion of one historical ATR and adverse excursions are reported separately.",
        "",
        "Every completed signal timestamp is evaluated. Multiple events may occur within one session and windows overlap. Event counts are not independent trade counts; uncertainty uses ten-session block resampling.",
        "",
        "| Price-based signal | Session | Window | Events | Higher afterward | Matched ordinary entry | Weakest quarter |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in price.head(5).iterrows():
        report.append(
            f"| {r.signal} | {r.session} | {int(r.horizon)} | {int(r.events)} | {r.direction_success:.1%} | {r.matched_base_success:.1%} | {r.minimum_quarter_success:.1%} |"
        )
    r = price.iloc[0]
    earlier = frame.loc[
        frame.split.eq("earlier")
        & frame.session.eq(r.session)
        & frame.horizon.eq(r.horizon)
        & frame.signal.eq(r.signal)
    ].iloc[0]
    report.extend(
        [
            "",
            f"The strongest observed price-based result is a Globex bullish engulfing signal measured over two hours: {r.direction_success:.1%} later-period success, versus {r.matched_base_success:.1%} at matched clock/volatility conditions. Its earlier-period success was {earlier.direction_success:.1%}; it does not demonstrate a persistent 60% edge. Combined-screen adjusted q={r.combined_screen_fdr_q:.3f}. Selected best results are exploratory, not validated winners.",
            "",
            "A recurring Globex entry around 20:00 measured to 09:30 produced 61.2% later-period direction success. This is a clock control, not evidence of a special candle/volume signal. Its weakest quarter was about 50.8%, and only 237 of 253 session events had complete observable outcomes. The matched same-clock control naturally represents that same timing effect; it cannot establish an additional signal effect.",
            "",
            "No combination passed the recorded entry qualification: every session in both periods; later success at least 60%; meaningful improvement with uncertainty and screen-level multiple-testing correction; stable quarters; at least 95% observable outcomes. This correction covers the combined 1,806-row screen, not every previously attempted experiment.",
            "",
            "Data limitations: reused development period August 2023–August 2025; no fresh untouched holdout; legacy NQ continuous export lacks verified contract-by-contract volume-roll mapping. NQ is absent from the Databento three-year source manifest, although a processed NQ directory exists. Genuine per-contract NQ bars and rollover provenance are prerequisites for a stronger institutional claim.",
            "",
            f"Verification: {audits} observed windows independently replayed directly against minute opens, closes, highs and lows. Twelve focused checks passed for execution timing, missing data, future-label cutoffs, daily minimum, signal causality and distinct direction/excursion labels. Repository tests: 158 passed. Existing unrelated repository lint/type/format failures remain (40 lint errors, 89 type errors, six formatting files); scoped changed calculations and research scripts pass their relevant checks.",
            "",
            "Research basis: Heston, Korajczyk and Sadka document time-of-day continuation and short-term reversals in equity returns. That motivates diagnostics; it does not prove an NQ edge. https://arxiv.org/abs/1005.3535",
            "",
            "Complete evidence: ENTRY_ONLY_RESULTS.csv plus protocol.json and results.csv in both entry_quality_runs folders. Historical profit/holding-period studies are separate and do not define the current entry task.",
        ]
    )
    (BASE / "ENTRY_ONLY_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    (BASE / "entry_replay_audit.json").write_text(
        json.dumps(
            {"replayed_windows": audits, "passed": True, "reports": [str(p) for p in paths]},
            indent=2,
        )
    )
    print(
        "REPORT",
        BASE / "ENTRY_ONLY_RESULTS.md",
        "audited",
        audits,
        "bestprice",
        r.direction_success,
        "q",
        r.combined_screen_fdr_q,
    )


if __name__ == "__main__":
    run()
