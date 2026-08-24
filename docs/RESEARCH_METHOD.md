# Research Method

## Objective

Develop systematic futures signals, evaluate them with realistic timing and costs, and combine complementary rules into portfolios without using holdout results for model selection.

The repository contains two distinct research tracks:

1. A-Tier Strategy 1, a seven-strategy multi-market portfolio whose rule sets were frozen in an automated research environment before holdout evaluation.
2. A custom three-rule CL / 6J / NG session-reversal portfolio developed with explicit temporal folds, neighbor tests, and a physically separated sealed year.

Neither track represents live trading performance.

## Data and timing controls

- One-minute source data are normalized into session and intraday research panels.
- Signal information is lagged beyond its availability time before a position can earn returns.
- Missing minute intervals are not bridged when computing high-frequency return features.
- Holdout samples are separated from development and selection activity.
- Search, fitting, parameter selection, and portfolio weighting use development information only.
- Raw licensed market data are excluded from version control.

## Research pipeline

1. Validate timestamps, duplicates, OHLC consistency, symbol coverage, and missingness.
2. Build point-in-time features and explicitly defined signal variants.
3. Apply delayed execution, holding rules, slippage, and transaction costs.
4. Evaluate development-period temporal stability and parameter sensitivity where supported.
5. Combine strategies using constrained weights and correlation evidence.
6. Freeze the candidate definition, inputs, and portfolio composition.
7. Evaluate the isolated holdout without refitting or reselection.
8. Preserve summary statistics, return series, manifests, and reconciliation evidence.

## A-Tier Strategy 1 holdout design

The seven rule sets cover 6E, 6J, CL, ES, GC, HG, and NG. The holdout ran from August 22, 2025 through August 21, 2026 using hourly signals, one-minute execution precision, one contract per rule, one tick of slippage, and the observed $7 round-turn ledger cost.

The holdout evidence is frozen. The package includes the summary JSON, visual report, validation record, signal implementation, and point-in-time tests. It does not include the licensed input data or private trade ledgers.

## Custom-research validation design

The CL / 6J / NG candidate used a five-day embargo and a physically separated final year. Pre-holdout checks included four development folds, parameter-neighbor tests, cost stress, block bootstrap, and portfolio concentration review. Strategy definitions and weights were frozen before the sealed evaluator was run once.

## Audit principle

A holdout is evidence only while it remains outside the research loop. Repeated evaluation, threshold changes, or parameter changes informed by holdout results would convert it into development data. This repository therefore preserves the frozen artifacts and explicitly identifies any criterion adopted after a result was observed.
