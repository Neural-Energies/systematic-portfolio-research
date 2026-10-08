# Frozen NQ entries and the first extended-hold ML test

The same100 entry definitions are locked. Original target/hour exits remain the reference; no ML exit has been approved as an upgrade. This is research, not live deployment.

## What was tested

Ridge regression, boosted regression trees and extra trees estimate remaining60-minute and240-minute price movement. Features use completed five-minute NQ and aligned ES observations across RTH and Globex. These are expected movement estimates, not calibrated probability scores. The continuation policy checks after the first60 minutes, then every five minutes; it holds while either estimated remaining move is positive. Both120-hour and five-trading-session caps were evaluated on2025. The selected exit was frozen before its separate2026 trade replay.

## The candidate selected in the reviewed view

| 2026 result | Original target/hour exit | First ML continuation exit |
|---|---:|---:|
| Trades | 201 | 139 |
| Net profit, one contract, assumed $25 round trip | $101,435 | $155,505 |
| Net win rate | 86.6% | 64.0% |
| Trade profit factor | 3.76 | 2.63 |
| Mean hold | Original target/hour window | 3.29 hours |
| Longest ML hold | — | 60.0 hours |

The ML candidate has $16,330 drawdown measured at observed five-minute marks. The original report has $9,955 minute-close drawdown; different marking frequencies prevent treating these as identical risk measurements. The extension increases observed profit but reduces win rate and trade PF.

## Across the shortlist

99 of100 selected ML candidate replays have no unresolved deadline trade. Of these,85 have more observed profit than their original exit, and none has a higher net win rate. One remaining candidate has an unresolved trade. Correlated candidates are not independent strategies and their profits must not be summed as a portfolio.

The2026 forward-movement forecasts do not improve RMSE over a zero-change forecast at either horizon. Increased historical profit alone therefore does not establish that the models reliably identify when the edge disappears. A rolling-horizon expected-return screen is not a solved dynamic-programming optimal-stopping policy.

## Why this remains preliminary

Some requested fixed-hold deadlines land in closures or absent quotes. Those trades remain unresolved, with complete net PNL unavailable; resolved-only profits are not fair full-strategy comparisons. Model-held paths also traverse quote gaps: the exchange calendar and all overnight/roll execution remain uncertified. The source export lacks verified volume-roll provenance.2026 was previously used in research. Constant$25 costs omit overnight spread changes and rollover trades; margin and financing are not modeled. Session frequency is still a separate unmet requirement.

A proper extension needs matched-duration benchmarks with certified full-session quotes and contract-roll handling, followed by conditional continuation evaluation near the existing target/time exit. Keep the original entries and exits while that work is developed.

[Optimal-stopping and statistical-learning research](https://arxiv.org/abs/math/0408276) supports comparing continuation value with immediate exit value; this pilot is a simpler screening model, not a replication of that paper.

## Reproduction and checks

Runner: work/nq_long_ml/extended_hold_pilot.py. Fixed seeds and chronological2022–2024 training/2025 validation; crossing training labels purged. All100 locked-entry rules use their original IS-fitted thresholds. Execution-boundary tests exclude pre-entry candles, enforce fixed hold deadlines and prevent a fabricated cross-period exit. Selected trade PNL independently reconciles to actual entry/exit quotes and$20-per-point NQ value.177 repository tests passed; baseline lint/type/format failures are unchanged.
