# Single-Writer Orchestration

## Authority chain

1. The Orchestrator (currently `opencode/ox-alpha`) is the single authority for shared state:
   - `research_program/hypothesis_registry.csv`
   - `research_program/learning_log.md`
   - `research_program/experiments_index.csv`
   - `PROJECT_STATE.md`, `DATA_CATALOG.json`
2. Workers/subagents may write ONLY inside their own run directories (`data/processed/<module>/runs/<id>/`,
   `saved_strategies/<archive>/`) and `work/` scripts. They return results to the orchestrator,
   which commits shared-state changes.
3. `research_program/WRITER_LOCK.json` records the current holder with a heartbeat. Before any
   shared-file edit: re-check mtimes of the target files; if another writer has touched them
   within its normal activity window, yield or coordinate.

## Handover history

- Codex ran passes 0-11 (H001-H022) and terminated cleanly at ~18:02 local on 2026-08-23,
  leaving `research_program/CURRENT_STATUS.md` and a five-item queue (H023-H027).
- OpenCode assumed orchestration at ~18:20 local after verifying: no live python processes,
  13+ minutes of filesystem silence, clean handover document.

## Bounded subloop limits (initial)

- hypothesis revisions per family: 2
- implementation repair attempts: 3
- tool retries: 3 unless clearly transient
- agent debate: ends when no new information is produced

## Compute bounds

- One hypothesis implementation + evaluation per active session block by default.
- Long jobs (>10 min) require a stated justification before launch.
