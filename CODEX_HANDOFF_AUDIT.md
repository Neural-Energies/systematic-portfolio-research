# Codex Handoff Audit

Date: 2026-08-23. Auditor: OpenCode (ox-alpha). Method: filesystem inspection plus execution
(122/122 pytest cases passed in 21.26 s via `.venv\Scripts\python.exe -m pytest -q`).
Ground truth was taken from artifacts, not from handoff notes.

## COMPLETED (implemented, executed, and evidenced)

- Data foundation: Databento 3-year 1-minute portfolio (2023-08-22 to 2026-08-22) cataloged,
  normalized (`data/processed/databento_canonical`, 10 roots), split into physically separate
  development and sealed-holdout minute partitions (`data/processed/databento_research`).
- Legacy Investor/RT canonical layer for 8 U6 contracts (`data/processed/canonical`).
- Validation stack: anchored 12-month holdout, 5-day embargo, permission-gated holdout access,
  one-shot sealed evaluator with lock/hash pinning (`temporal_validation.py`,
  `sealed_evaluator.py`, `daily_session_sealed.py`). All unit-tested.
- Strategy engines: broad search, legacy factory, VectorBT factory, paper factory,
  daily-session factory, unique-hypothesis factory, ML-hypothesis factory, COT module.
- Research program memory: `research_program/hypothesis_registry.csv` (H001-H019),
  `learning_log.md` (Passes 0-7), `RESEARCH_LOOP_PROTOCOL.md`.
- Sealed evaluation consumed exactly once per candidate:
  - `stable_daily_session_2026-08-23` PASSED (+54.92 % total, Sharpe 2.42, MaxDD -5.72 %)
    and is archived as Tier A under `saved_strategies/BEST_STRATEGIES/tier_a_sealed_validated/`.
  - `combined_session_intraday_2026-08-23` FAILED (-2.82 % total, Sharpe -0.52), retained as a
    permanently rejected candidate.
- CFTC disagg archives 2023-2025 with fingerprints (`data/external/cftc`, `work/cftc_downloads`).
- Handoff-note claims verified from files:
  - "candidate dev Sharpe ~1.81" -> `data/processed/candidate_portfolio/portfolio_aggregate_summary.csv`
    shows candidate_sleeves_equal_weight Sharpe 1.8082 (226 sessions).
  - "CL trend sleeve dev Sharpe ~2.06" -> same folder sleeve_aggregate_summary.csv shows
    ml_cl_trend Sharpe 2.0555, positive in all four folds.

## PARTIALLY COMPLETED

- Tier B pipeline: no current strategy meets Tier B thresholds; ml_cl_trend (2.06) and the
  1.81 candidate remain development-only research evidence, correctly not promoted.
- Portfolio breadth: only H001-family survives; every other registered family is rejected.
- Notebooks/outputs duplication between `notebooks/` and
  `outputs/Systematic Portfolio Research/` (same content, two locations).

## FAILED (documented, must not be silently retried)

- H002-H018 all rejected after preregistered walk-forward falsification
  (see hypothesis_registry.csv rows and learning_log.md Passes 3-7).
- Invalidated run `20260823T203416Z` (turnover-accounting bug, later fixed and regression-tested).
- Combined session+intraday candidate failed its single sealed evaluation.

## PROMISING (preserved, not yet advanced)

- ml_cl_trend sleeve: Sharpe 2.06, 4/4 positive folds, negatively correlated (-0.51) with
  multi_asset_trend - genuine portfolio diversification value; development-only.
- Candidate equal-weight portfolio: Sharpe 1.81, 3 positive folds, worst fold -0.54.
- H001 sealed-validated reversal family (Tier A archive) is the reference product.

## NOT STARTED

- H019 "PCA-residual cross-sectional reversion" is REGISTERED in the hypothesis registry but
  has no module, no test, no run directory, and no result. This is exactly where work stopped.
- No fresh untouched holdout exists for future candidates (sealed year consumed once).

## BLOCKED

- Contract-consistent multi-session / close-to-close testing: canonical parquet contract
  month/year fields exist in schema but are null; filenames carry only root symbols.
  Session-only adaptations remain labeled as such.
- Git binary unavailable in this shell; history inspected via `.git` HEAD/refs only (branch main).

## Infrastructure verification executed this session

- `.venv\Scripts\python.exe -m pytest -q`: 122 passed, 21.26 s.
- Raw Databento source path from config/data_sources.yaml verified to exist on disk.
