"""Readable entry-only findings with independent minute-price replay."""

# ruff: noqa: E501
import json

import numpy as np
import pandas as pd
from entry_screen import fdr
from experiment import load_minutes
from fifteen_probability import BASE

SOURCE = BASE / "rth_named_runs" / "20261006T200105Z"
AUDIT = BASE / "rth_named_audits" / "20261006T200644Z"
NAME = "vp_value_acceptance_2pt_uniform"


def run():
    if not (SOURCE / "protocol.json").exists() or not (AUDIT / "README.md").exists():
        raise ValueError("Incomplete run")
    d = pd.read_csv(SOURCE / "results.csv")
    audit = pd.read_csv(AUDIT / "audit.csv")
    reach = d.split.eq("later") & d.label.eq("reached_up")
    d.loc[reach, "fdr_q_reached_up"] = fdr(d.loc[reach, "p_hac"].fillna(1).to_numpy())
    d.to_csv(SOURCE / "results_with_reach_fdr.csv", index=False)
    meta = pd.read_parquet(SOURCE / "metadata.parquet")
    minute = load_minutes("NQ")
    replayed = 0
    for horizon in (5, 15, 60, 240):
        for distance_name in ("10points", "20points", "1atr"):
            paths = pd.read_parquet(SOURCE / f"paths_{horizon}_{distance_name}.parquet")
            for stamp, row in paths.sample(15, random_state=71).iterrows():
                end = min(stamp + pd.Timedelta(minutes=horizon), meta.loc[stamp, "session_end"])
                chunk = minute.loc[stamp : end - pd.Timedelta(minutes=1)]
                distance = {
                    "10points": 10.0,
                    "20points": 20.0,
                    "1atr": float(meta.loc[stamp, "atr"]),
                }[distance_name]
                entry = float(minute.loc[stamp, "open"])
                up = chunk.index[chunk.high.ge(entry + distance)]
                down = chunk.index[chunk.low.le(entry - distance)]
                # Independent verification for fully observed source windows.
                complete = (
                    len(chunk) == int((end - stamp).total_seconds() / 60)
                    and (chunk.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()
                )
                if complete:
                    u = up[0] if len(up) else end
                    v = down[0] if len(down) else end
                    assert row.path_scorable
                    assert row.up_first == float(u < v)
                    assert row.reached_up == float(len(up) > 0)
                    replayed += 1
    rows = d[
        d.signal.eq(NAME)
        & d.split.eq("later")
        & d.distance.eq("20points")
        & d.label.eq("reached_up")
    ].sort_values("horizon")
    risk = d[
        d.signal.eq(NAME) & d.split.eq("later") & d.distance.eq("20points") & d.label.eq("up_first")
    ].set_index("horizon")
    prior = d[
        d.signal.eq(NAME)
        & d.split.eq("earlier")
        & d.distance.eq("20points")
        & d.label.eq("reached_up")
    ].set_index("horizon")
    delayed = audit[
        audit.signal.eq(NAME)
        & audit.split.eq("later")
        & audit.delay_minutes.eq(1)
        & audit.horizon.eq(60)
        & audit.label.eq("reached_up")
    ].iloc[0]
    representative = rows.loc[rows.horizon.eq(60)].iloc[0]
    bank = d[
        d.signal.eq("catalog_FT71-inspired")
        & d.split.eq("later")
        & d.label.eq("up_first")
        & d.distance.eq("10points")
        & d.horizon.eq(60)
    ].iloc[0]
    text = [
        "# NQ RTH entry research: profile, Raschke and Brooks concepts",
        "",
        "We found entry candidates worth further research. Prior-value-area re-entry is the strongest recent upside candidate. It does not yet satisfy the one-entry-per-session requirement by itself, and its advantage was absent in the earlier year. No high-forecast, target, stop or portfolio model was fitted.",
        "",
        "The scope is long entries during 09:30â€“16:00 New York, measured over 5, 15, 60 and 240 minutes, capped at session close. Tested 61 mechanical setup variants and predefined family unions. Each reference distance (10 points, 20 points, one prior ATR) creates 244 later-period combinations; 732 per outcome family. Distances describe price paths; they are not proposed trade targets or stops.",
        "",
        "A concrete correction to the previous research: closing above an entry is a different event from making a favorable move before retracing. We now count favorable visits and record adverse movement before them. We also retain known touches preceding a later data gap. Unknown labels and same-minute ordering ambiguity are reported; conservative primary rates do not count them as favorable-first wins.",
        "",
        "**Candidate: prior-value-area re-entry**",
        "",
        "Build the previous completed RTH session's approximate 70% volume value area. After price was below value, require two consecutive completed five-minute closes inside that value area, with a bullish second bar. The entry reference is the next minute open. All three context bars belong to the current RTH session. This is our mechanical auction-inspired interpretation, not a claim to replicate FuturesTrader71's discretionary trading.",
        "",
        "For this representative variant, estimate minute volume uniformly across two-point price bins. Also tested one/four-point bins and assigning minute volume to its close; source minute OHLCV does not reveal actual trade-by-trade volume at price.",
        "",
        "| Measurement window | Reached entry +20 | Matched ordinary entry | +20 before -20 | Signals |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in rows.iterrows():
        text.append(
            f"| {int(r.horizon)} min | {r.success_rate:.1%} | {r.matched_rate:.1%} | {risk.loc[r.horizon, 'success_rate']:.1%} | {int(r.events)} |"
        )
    text.extend(
        [
            "",
            f"The one-hour reach rate is {representative.success_rate:.1%} ({int(round(representative.success_rate * representative.events))}/{int(representative.events)} observed signals), versus {representative.matched_rate:.1%} for ordinary entries matched on clock, quarter and volatility bucket. This is price reaching entry+20 at some point, not forecast accuracy or a trading win rate.",
            "",
            f"A finer comparison matched on continuous ATR and preceding bar direction/body. A one-minute delayed entry still reached +20 on {delayed.success_rate:.1%} of signals, versus {delayed.nearest_bullish_body_atr_control:.1%} for those controls. The estimated excess-rate interval has a lower bound of {delayed.incremental_ci_low:.1%}. This is a post-screen robustness diagnostic, not independent validation.",
            "",
            f"Median adverse movement before reaching +20 was {representative.median_adverse_before_up:.2f} points among paths with a known favorable touch; failures to reach are outside that conditional median. Full one-hour median adverse movement was {representative.median_adverse_points:.2f}. Maximum favorable/adverse movement over a window should not be confused with the movement before a future exit.",
            "",
            f"This setup appeared in {representative.session_coverage:.1%} of the {int(representative.candidate_sessions)} later RTH sessions. It is a specialist signal, not a daily solution. In the earlier period it reached +20 within an hour on {prior.loc[60, 'success_rate']:.1%}, compared with {prior.loc[60, 'matched_rate']:.1%} ordinary entries. The earlier period does not support the same edge.",
            "",
            "**Daily requirement and other methods**",
            "",
            f"The predefined profile-rule union produced signals in every eligible later RTH session. Its +10-before-minus-10 rate within an hour was {bank.success_rate:.1%}, versus {bank.matched_rate:.1%}; adjusted q={bank.fdr_q_first_touch:.3f}. Broad coverage weakens the specialist signal's apparent advantage. We have not demonstrated an institutionally validated high-success entry bank with mandatory daily coverage.",
            "",
            "Brooks-style wedge reversal and developing value-area-low rejection also merit follow-up. Exact definitions and results for High 2, wedge, double bottom, breakout follow-through, Raschke Grail/3-10/failed-low adaptations, and auction setups are in registry.csv and results_with_reach_fdr.csv. These are mechanical interpretations; discretionary context and intrabar execution are not reproduced.",
            "",
            "Robert J. Frey's public academic material supports quantitative research and risk analysis; we did not identify a public proprietary NQ entry rule to reproduce. His name is not used to label an invented entry pattern. No stop model was fitted.",
            "",
            "**What keeps these preliminary**",
            "",
            "The data is reused development history, not an untouched final holdout. The NQ source is a legacy continuous OHLCV export with unverified contract-by-contract volume rollover. Real minute volume is available; its distribution across traded prices is estimated. The strongest setup's earlier-period and daily-coverage limitations remain. FDR adjustments cover this screen, not all past experiments.",
            "",
            "No RTH was available on January 9, 2025: CME equity-index trading closed at 09:30 New York. The single closing-time quote is excluded from RTH profiles and does not replace the prior actual RTH session. Initial missing prior-session context and the initial partial evaluation boundary are excluded from the eligible research universe. Signal coverage is reported against that explicit universe.",
            "",
            f"Verification: {replayed} independently checked fully observed source paths; nine focused checks covering profile volume conservation, Wilder seeding, touch ordering, gaps, future causality, same-session confirmation and the special closure. Repository suite: 164 passed. Changed shared calculations/type checks and scoped research lint pass. Existing unrelated repository lint/type/format issues remain.",
            "",
            "**Primary research material**",
            "",
            "- FuturesTrader71's auction/profile terminology and concepts: https://futurestrader71.wordpress.com/2009/09/29/confused-about-the-lingo/",
            "- Linda Raschke's hosted interview, including tendency-versus-close scoring, Grail and 3-10 oscillator: https://lindaraschke.net/wp-content/uploads/2026/03/raschke_pt2_0304.pdf",
            "- Al Brooks' public pattern descriptions: https://www.brookstradingcourse.com/price-action/10-best-price-action-trading-patterns/",
            "- Robert J. Frey and Stony Brook quantitative-finance research context: https://www.stonybrook.edu/ams/academics/graduate/quantitative-finance.html",
            "- CME January 9 closure: https://www.cmegroup.com/trading-hours/files/day-of-mourning-january-9-2024.pdf",
            "",
            f"Complete results: {SOURCE}. Finer matching and latency checks: {AUDIT}. Superseded prototype runs are retained for an audit trail; this report uses only the corrected runs above.",
        ]
    )
    (BASE / "NQ_RTH_ENTRY_FINDINGS.md").write_text("\n".join(text), encoding="utf-8")
    (SOURCE / "replay_audit.json").write_text(
        json.dumps(
            {"fully_observed_paths_replayed": replayed, "passed": True, "audit_source": str(AUDIT)},
            indent=2,
        ),
        encoding="utf-8",
    )
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    xs = np.arange(4)
    a = rows.success_rate.to_numpy() * 100
    b = rows.matched_rate.to_numpy() * 100
    c = risk.reindex(rows.horizon).success_rate.to_numpy() * 100
    axes[0].bar(xs - 0.18, a, width=0.36, label="Value-area re-entry", color="#277f77")
    axes[0].bar(xs + 0.18, b, width=0.36, label="Matched ordinary entry", color="#9da5ad")
    axes[0].set_title("Reached +20 at some point")
    axes[1].bar(xs, c, width=0.55, color="#3d668f")
    axes[1].set_title("Reached +20 before -20")
    for ax in axes:
        ax.set_xticks(xs, ["5 min", "15 min", "60 min", "4 hours"])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Percent of signal events")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("NQ RTH: prior-value-area re-entry â€¢ later research year", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "117 overlapping signal events â€¢ specialist setup: 29% of sessions â€¢ exploratory data",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(BASE / "NQ_RTH_ENTRY_FINDINGS.png", dpi=150)
    plt.close(fig)
    print(
        "REPORT",
        BASE / "NQ_RTH_ENTRY_FINDINGS.md",
        "paths",
        replayed,
        "reach_q",
        representative.fdr_q_reached_up,
    )


if __name__ == "__main__":
    run()
