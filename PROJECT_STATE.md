# PROJECT_STATE

CURRENT MILESTONE: Pass 12 complete (H024 executed and rejected); OpenCode is primary
orchestrator following Codex's clean handover after Pass 11

CURRENT RESEARCH PHASE: development_only falsification; sealed year consumed once (2026-08-21);
new work proceeds PROMISING -> VALIDATION CANDIDATE -> HISTORICALLY ROBUST -> FORWARD VALIDATION

ACTIVE WRITER: opencode/ox-alpha (see research_program/WRITER_LOCK.json; Codex terminated
cleanly ~18:02 local, verified by process scan + 13-minute write silence)

ACTIVE EXPERIMENT: none in flight

QUEUED EXPERIMENTS: H023 BOCPD local trend; H025 wavelet multiresolution trend;
H026 copula tail-dependence contagion; H027 EIA surprise (BLOCKED - needs timestamped
official releases). Director note: trend-family items deprioritized per breadth mandate.

DATA STATUS: 26 datasets inventoried in DATA_CATALOG.json (dev/sealed 1m partitions x10 roots,
session features, CFTC archives 2023-25, legacy U6 layer, external raw pointer). Contract IDs
null -> session-safe adaptations only; sealed partitions access=false for research.

CANONICAL DATA PATHS: data/processed/databento_research/development_minute_returns/symbol=*/
(absolute paths in DATA_CATALOG.json)

ENGINE STATUS: full suite 135 passed (31s); mypy strict green on all OpenCode-authored files;
54 pre-existing mypy errors in Codex-era modules documented as debt (eia 18, cot 17, pca 8,
market_structure 4, features 3, managed_money 1) - untouched to avoid collisions

VALIDATION STATUS: chained expanding walk-forward standard (select on earlier folds only);
preregistered gates incl. gross-positive AND 1bp-stress AND outlier-removal survival;
bootstrap moving-block b=20 n=5000 seed 20260823

MEMORY STATUS: registry H001-H024 (23 rows); learning_log Passes 0-12; experiments_index.csv
44 runs; promise_audits/ robustness audit; WRITER_LOCK.json + ORCHESTRATION.md added;
DATA_CATALOG.json added this session

STRATEGY LIBRARY STATUS: Tier A validated =1 (stable_daily_session_2026-08-23, sealed Sharpe
2.42); rejected families =22 of 24 registrations; promising =2 fragile sleeves (rho 0.94,
top-10-day concentration 83%/260%); validation candidates =0

TOTAL EXPERIMENTS: 24 registered hypotheses; >4,000 parameter variants; 44 manifest runs indexed

REJECTED COUNT: H002-H024 (23 minus H001) including both H019 passes and H024

PROMISING COUNT: 2 sleeves + preserved observation from H024 (body-not-tails dislocation
reversion contrast vs H001 extreme-tail reversal)

CANDIDATE COUNT: 2 historical candidates (both consumed their one sealed evaluation)

VALIDATED COUNT: 1

PORTFOLIO STATUS: no live-ready portfolio; independent alpha remains unproven across 23
falsification attempts spanning intraday asymmetry, positioning, network-RV, event drift

CURRENT BOTTLENECK: portfolio breadth / independent alpha (unchanged by H024 verdict)

ACTIVE TASK: complete H024 cycle (done: implement->test->execute->verify->classify->record)

NEXT TASK: H026 copula tail-dependence contagion at session scale (breadth-aligned, novel
family, preregister before testing); then reassess H023/H025 priority against the H024
body-vs-tails observation

KNOWN PROBLEMS: no untouched holdout; contract metadata null; uv trampoline error (use
.venv python directly); git binary absent; pre-existing mypy/ruff debt in Codex modules

LAST CHECKPOINT: 2026-08-23 22:35 local - H024 run 20260823T223316Z archived with SHA256SUMS
