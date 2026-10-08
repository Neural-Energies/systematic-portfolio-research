# NQ simulation after the new-import timestamp audit

**One rule passes the declared post-2024 gates under both clock interpretations: NQ_000748828115. No rule passes the stricter all-period/all-scenario diagnostic.** One entry per session is no longer required. All100 original entry definitions and numerical thresholds remain frozen.

The original minute-start assumption produced13 provisional post-2024 survivors. The new5m export matches6849 overlapping relativeOHLC shapes and volumes exactly after shifting old minute timestamps−1minute; matched prices differ by+297.25points. Under the corresponding minute-end sensitivity, only one passes the post-2024 gates. This qualifies earlier optimistic fill assumptions; human endpoint confirmation is still pending.

| NQ_000748828115, minute-end assumption | Trades | Net profit | Trade PF | Net-winning trades |
|---|---:|---:|---:|---:|
| Discovery2022–2024 | 147 | $5,045 | 1.12 | 76.2% |
| Previously used2025 | 182 | $24,390 | 1.40 | 82.4% |
| Previously seen2026 | 126 | $26,630 | 1.83 | 78.6% |
| Combined2025–2026 | 308 | $51,020 | 1.55 | 80.8% |

Combined delay/slippage/penetration/$50cost stress over2025–2026: $36,560, PF1.43, 77.3% net winners. Base combined minute-close drawdown is$15,160. Discovery stress is weaker: $100round-trip costs yield−$5,980 and combined stress−$315, so this is a research lead, not an all-regime certified system.

The prior highlightedNQ_000926441771 has minute-end base PNL−$95 in2025 and+$7,695 in2026. The original$101,4352026 figure belongs to the old clock case and is not the corrected-clock estimate.

The new attachment is preserved, with9402 clean unique5m bars fromApril23 throughOctober7,2026. Three malformed rows are quarantined and86 identical repeats logged; no conflicting duplicates.32 dates follow the oldAugust20cutoff. We froze13 candidates before reading the new file. These dates are not scored as a fresh holdout because the attachment lacks minute/full-GlobexNQ and contemporaneousES needed to calculate the frozen signals and delayed fills. Missing features are not converted into zero trades.

Scope: two clock cases×2700 period/scenario replays, plus1800 combined diagnostics,3200 quarter records,400k randomized timing trials and weekly block bootstrap checks. The9 fill/cost scenarios and gate definitions are saved inPROTOCOL.json. Independent minute-engine checks passed in both clock cases. Original138 atom thresholds reproduce packed masks exactly; two history-prefix spot checks reproduce66 required features.181 repository tests passed; scoped code checks pass; pre-existing repository lint/type/format issues remain.

All historical post-2024 data were previously seen, and all100 definitions were selected after10milliontrials.100-comparisonHolm and weekly bootstrap diagnostics are retrospective and do not correct that full search. No full-searchDSR/PBO is claimed. Underlying volume-roll contract provenance remains unverified. MinuteOHLCcannot reproduce actualbidask, queueposition or trueintraminute ordering; penalties are assumptions. No stop, sizing or margin model is added. These are simulated historical outcomes, not live/paper execution.

The new clock audit supersedes the earlier13-survivor conclusion. Old results are retained for reproducibility. Local source data and locked strategies were not deleted; generated redundant report data were reduced and generated files transparently compressed after thedrive filled.

[Selection-bias research](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) explains why search breadth and return distribution matter; it does not validate this survivor.
