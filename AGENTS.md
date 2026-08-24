# Workspace guidance for Codex

## Purpose

This repository supports reproducible, institutional-quality research across liquid futures,
FX, and crypto. Prefer transparent research code and defensible assumptions over elaborate
frameworks.

## Working rules

- Use `uv sync` for environments and `uv run ...` for commands. Do not install packages ad hoc.
- Keep reusable code in `src/systematic_research`; notebooks should orchestrate and visualize.
- Keep credentials in `.env` and large/vendor data under `data/`; neither belongs in Git.
- Avoid look-ahead bias, survivorship bias, silent timezone conversion, and accidental contract
  roll assumptions. State units, calendars, lag conventions, and transaction costs explicitly.
- Add or update tests for reusable calculations. Use deterministic seeds in stochastic research.
- Before handing off changes, run `uv run ruff format --check .`, `uv run ruff check .`,
  `uv run mypy`, and `uv run pytest`.

