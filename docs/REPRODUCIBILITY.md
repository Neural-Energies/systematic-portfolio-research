# Reproducibility

## Root research environment

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

The same checks run in GitHub Actions on pushes and pull requests.

## Frozen multi-market holdout package checks

The seven-strategy holdout package has an isolated Python configuration. Its implementation and point-in-time tests can be checked without the private market data:

```powershell
cd deliverables/*_holdout_portfolio
uv sync --frozen
uv run ruff format --check src tests scripts
uv run ruff check src tests
uv run pytest -q
```

The frozen report cannot be regenerated without the excluded private ledgers. Rebuilding it would not constitute an independent holdout test and is not required for repository review.

## Data boundary

Raw market data are intentionally excluded because vendor data may be licensed and too large for source control. The repository includes:

- schema and validation code;
- example universe and research configuration;
- tests using controlled fixtures;
- selected derived evidence for frozen candidates; and
- source fingerprints and manifests for auditability.

The public `config/data_sources.yaml` uses a generic relative source path. Local machine-specific data locations should be supplied through ignored local configuration rather than committed to the repository.

To reproduce the root research pipeline, provide compatible one-minute data at the paths defined in the local data-source configuration, then run the catalog, normalization, and development stages.

## Root pipeline commands

```powershell
uv run research-data
uv run research-normalize
uv run research-panel
uv run research-features
uv run research-strategy-factory
uv run research-candidate-portfolio
```

The sealed evaluators are intentionally separate from development search. A new holdout should remain locked until a candidate is frozen; repeated evaluation on the same holdout converts it into development data.
