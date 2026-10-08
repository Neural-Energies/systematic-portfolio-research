"""Post-screen latency and finer matching diagnostics; not a new holdout."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from adaptive_entries import period_scope
from entry_screen import incremental_test
from experiment import END, SEED, aggregate, load_minutes
from fifteen_probability import BASE
from medium_frequency import FIRST_TEST

from systematic_research.auction_entry import entry_path_diagnostics

SOURCE = sorted(p for p in (BASE / "rth_named_runs").iterdir() if (p / "protocol.json").exists())[
    -1
]


def nearest_control(scope, labels):
    result = pd.Series(np.nan, index=scope.index)
    for _, g in scope.groupby(["quarter", "slot"]):
        x = g[["log_atr", "body_atr"]].to_numpy(dtype=float)
        scale = np.nanquantile(x, 0.75, axis=0) - np.nanquantile(x, 0.25, axis=0)
        scale = np.maximum(scale, 1e-6)
        dist = (((x[:, None, :] - x[None, :, :]) / scale) ** 2).sum(axis=2)
        same_day = g.anchor.to_numpy()[:, None] == g.anchor.to_numpy()[None, :]
        opposite_sign = np.sign(x[:, 1])[:, None] != np.sign(x[:, 1])[None, :]
        dist[same_day | opposite_sign] = np.inf
        neighbors = np.argsort(dist, axis=1)[:, : min(8, len(g))]
        finite = np.isfinite(np.take_along_axis(dist, neighbors, axis=1))
        y = labels.reindex(g.index).to_numpy(dtype=float)[neighbors]
        count = finite.sum(axis=1)
        avg = np.where(finite, y, 0).sum(axis=1) / np.maximum(count, 1)
        avg[count < 4] = np.nan
        result.loc[g.index] = avg
    return result


def run():
    out = BASE / "rth_named_audits" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    signals = pd.read_parquet(SOURCE / "signals.parquet")
    meta = pd.read_parquet(SOURCE / "metadata.parquet")
    registry = pd.read_csv(SOURCE / "registry.csv")
    names = registry.loc[
        registry.signal.str.startswith(
            ("vp_value_acceptance", "vp_open_rejection", "vp_developing_val")
        )
        | registry.signal.isin(["brooks_wedge_reversal", "catalog_FT71-inspired"]),
        "signal",
    ].tolist()
    minute = load_minutes("NQ")
    bars = aggregate(minute, 5)
    body = (
        pd.Series(
            (bars.close - bars.open).to_numpy(), index=pd.DatetimeIndex(bars.available_at)
        ).reindex(signals.index)
        / meta.atr
    )
    rows = []
    for horizon in (5, 15, 60, 240):
        original = pd.read_parquet(
            BASE / "entry_runs" / "20261006T164818Z" / f"outcomes_RTH_{horizon}.parquet"
        ).reindex(signals.index)
        original["log_atr"] = np.log(meta.atr)
        original["body_atr"] = body
        for delay in (0, 1):
            entries = signals.index + pd.Timedelta(minutes=delay)
            ends = pd.DatetimeIndex(
                np.minimum(
                    (entries + pd.Timedelta(minutes=horizon)).asi8,
                    pd.DatetimeIndex(meta.session_end).asi8,
                ),
                tz="UTC",
            )
            paths = entry_path_diagnostics(minute, entries, ends, np.full(len(entries), 20.0))
            paths.index = signals.index
            frame = original.join(paths)
            frame["planned_exit"] = ends
            for split, start, end in (
                ("earlier", frame.index.min(), FIRST_TEST),
                ("later", FIRST_TEST, END),
            ):
                scope = period_scope(frame, start, end).copy()
                scope["scorable"] = scope.planned_exit.lt(end)
                days = pd.DatetimeIndex(scope.anchor.unique()).sort_values()
                for label in ("up_first", "reached_up"):
                    scope["label"] = scope[label].fillna(0)
                    scope["control"] = nearest_control(scope, scope.label)
                    for name in names:
                        event = scope.loc[signals[name].reindex(scope.index)].copy()
                        event["net_dollars"] = event.label
                        stats = incremental_test(event, days, SEED)
                        rows.append(
                            {
                                "signal": name,
                                "horizon": horizon,
                                "delay_minutes": delay,
                                "split": split,
                                "label": label,
                                "events": len(event),
                                "success_rate": float(event.label.mean()),
                                "nearest_bullish_body_atr_control": float(event.control.mean()),
                                "session_coverage": event.anchor.nunique() / len(days),
                                **stats,
                            }
                        )
                print(horizon, delay, split, "audited", flush=True)
                pd.DataFrame(rows).to_csv(out / "audit.csv", index=False)
    (out / "README.md").write_text(
        (
            "Post-screen exploratory audit, not independent validation. "
            "Same-clock/quarter nearest eight other-session controls matched on "
            "continuous lagged ATR and normalized signal-bar body, requiring the "
            "same body sign. Outcomes unknown because of coverage gaps remain "
            "conservative failures. Test measurement windows from "
            "one-minute-delayed observed entry price. No target/stop/portfolio "
            "model fitted."
        ),
        encoding="utf-8",
    )
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    run()
