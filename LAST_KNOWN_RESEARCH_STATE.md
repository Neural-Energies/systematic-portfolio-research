# Last Known Research State (Codex -> OpenCode)

## 1. Last completed milestone

Pass 7 of the continuous research loop: preregistered hypothesis H018 (CFTC producer/merchant
processor hedging-pressure cross-sectional premium) was implemented, executed on development
data with Friday publication delay, and REJECTED (gross Sharpe -0.25, -0.36 after 1 bp/side,
44.2 % winning weeks, 3 of 4 folds negative, bootstrap median Sharpe -0.33). Registry and
learning log were updated.

## 2. Last completed experiment

`research-cot-hedging-pressure` (H018), followed by the final registered-family test
`research-ml-hypothesis-factory` run `20260823T213054Z` (H017 pooled ML; ensemble Sharpe
-1.59 at 1 bp/side; all four audit gates false; archived under
`saved_strategies/hypothesis_H017_20260823T213054Z`). File timestamps show this was the last
write activity in the workspace on 2026-08-23 (~21:30 Z).

## 3. Work in progress

None found: no partial scripts, no pending queues, no interrupted runs, empty `logs/`.
The only open item is a REGISTERED but UNTESTED hypothesis:

- H019: temporary product-specific dislocations mean-revert after removing the common
  cross-futures return factor. Causally estimated trailing principal-component residual,
  ranked across products, faded at 15-240 minute horizons.
  Falsification rule (preregistered): "Reject if the chained out-of-fold portfolio is not
  positive gross or fails 1-bp-per-side cost stress."

No H019 module exists in `src/systematic_research/`, no tests, no run artifacts.

## 4. Likely intended next task

Implement H019 following the established per-hypothesis pattern (standalone module + CLI entry
point + unit tests + preregistered expanding walk-forward evaluation + manifest archive +
registry/learning-log update). This matches the loop protocol step order and the registry state.

## 5. Evidence

- `research_program/hypothesis_registry.csv` line 20: H019 status = `registered`; all prior
  rows are `sealed_validated` or `rejected`.
- `research_program/learning_log.md` ends at Pass 7 (H018); no Pass 8 entry.
- `src/systematic_research/` contains modules for every other hypothesis family but none for H019.
- `data/processed/` contains run folders for all executed families; none for H019.
- Newest files in the tree are the H017 run/archive at 21:30 Z on 2026-08-23.

## Context needed to continue

- Development data only: `data/processed/databento_research/development_minute_returns/symbol=*/returns.parquet`
  for 6E, 6J, CL, ES, GC, HG, NG, NKD, ZC, ZN. Sealed partitions must remain untouched.
- Engine conventions to reuse (do not rebuild): bar resampling, causal lagging, turnover with
  forced session close, chained expanding folds, cost grid, manifests, SHA-256 snapshots -
  implemented in `unique_hypothesis_factory.py` (passes 3-5) and `cot_hedging_pressure.py` /
  `ml_hypothesis_factory.py` (per-hypothesis module pattern).
- Prior passes learned: intraday cost survival is the binding constraint (most families died
  between gross and net); positive family-level training expectancy is required before
  portfolio weight; winner's curse reversal across folds was pervasive.
