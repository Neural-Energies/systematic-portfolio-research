# Reproducibility

## Environment

- Python 3.12
- Dependencies locked in `uv.lock`
- Package and command definitions in `pyproject.toml`

Create the environment and run the verification suite:

```powershell
uv sync --frozen
uv run ruff format --check src tests
uv run ruff check src tests
uv run python -m mypy src
uv run python -m pytest -q
```

## Data boundary

Raw market data are intentionally excluded because vendor data may be licensed and too large for source control. The repository includes:

- schema and validation code;
- example universe and research configuration;
- tests using controlled fixtures;
- selected derived evidence for the frozen candidate; and
- source fingerprints and manifests for auditability.

To reproduce full backtests, provide compatible one-minute data at the paths defined in the local data-source configuration, run the catalog and normalization stages, and then execute the documented research commands.

## Key commands

```powershell
uv run research-data
uv run research-normalize
uv run research-panel
uv run research-features
uv run research-strategy-factory
uv run research-candidate-portfolio
```

The sealed evaluator is intentionally separate from development search. A new holdout should remain locked until a candidate is frozen; repeated evaluation on the same holdout converts it into development data.
