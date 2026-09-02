# Research Program Registry

This directory preserves the research-governance layer behind the public strategy examples.

- `RESEARCH_LOOP_PROTOCOL.md` defines hypothesis, validation, promotion, and archival rules.
- `hypothesis_registry.csv` records hypothesis definitions and research metadata.
- `experiments_index.csv` preserves successful, rejected, incomplete, and development-only experiment records.

## Provenance paths

Some `manifest_path` values in `experiments_index.csv` point to `data/processed/...`. Those are provenance references to generated local research artifacts that are intentionally excluded from version control with the raw/licensed data boundary. They are retained in the registry rather than rewritten because changing them would weaken the historical audit trail.

Checked-in evidence intended for public review is stored under `saved_strategies/`, `deliverables/`, `docs/`, and selected `outputs/` artifacts.

## Negative results

Rejected hypotheses and negative results are deliberately retained. The registry is an audit trail, not a leaderboard, and failed research is not removed simply to improve the appearance of the repository.
