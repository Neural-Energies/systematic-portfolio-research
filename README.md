# Systematic Portfolio Research

A compact, reproducible Python workspace for multi-asset futures, FX, and crypto research.

## First use

1. Open this folder in VS Code.
2. Run `uv sync` in the integrated terminal. This creates `.venv` from `uv.lock`.
3. If VS Code does not select it automatically, run **Python: Select Interpreter** and choose
   `.venv\\Scripts\\python.exe`.
4. For notebooks, choose the same `.venv` kernel.
5. Run `uv run python -m systematic_research.smoke` and `uv run pytest`.

The workspace recommends Python, Pylance, Ruff, mypy, Jupyter, and Codex extensions. Ruff formats,
sorts imports, and lints; mypy checks types; pytest and Hypothesis cover example- and
property-based tests. The default VS Code test task runs all four quality checks.

## Layout

- `config/`: versioned research assumptions and universe definitions
- `data/`: ignored raw, interim, and processed local datasets
- `notebooks/`: exploration and research narratives
- `src/systematic_research/`: reusable, tested research code
- `tests/`: unit and property-based tests
- `reports/figures/`: ignored generated figures

## Research foundations included

- Typed, fail-fast YAML configuration for research assumptions, portfolio limits, and costs
- Market-data validation for UTC timestamps, duplicates, nulls, sorting, and positive prices
- Bias-aware position lagging and net returns after turnover-based implementation costs
- Inverse-volatility and constrained minimum-variance portfolio construction
- Annualized volatility, Sharpe ratio, drawdowns, and historical expected shortfall
- An example multi-asset universe with contract multipliers and exchange timezones

Use the modules as tested building blocks, not as a claim that a research design is production
ready. Futures rolls, FX funding, crypto venue fragmentation, margin, corporate actions, benchmark
definitions, and live order management depend on the actual data vendors and brokers selected.

## Reproducibility

Commit `pyproject.toml` and `uv.lock`. Use `uv sync --frozen` to reproduce the exact dependency
graph on another machine. Do not commit secrets, licensed market data, large extracts, or notebook
outputs containing sensitive data.

Common commands:

```powershell
uv sync
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
uv run jupyter lab
uv run research-data
uv run research-normalize
uv run research-regression-displacement
uv run research-rates-reversion
uv run research-candidate-portfolio
uv run research-broad-search
```

## Broad Databento strategy search

`uv run research-broad-search` evaluates 60 development-only variants: ten each for momentum,
trend, mean reversion, breakout, relative value, and lead/lag. Signals are implemented one session
later, carry overlapping one-to-five-session holding cohorts, and pay 3.5 bps of one-way turnover
cost. Every strategy is retained and labeled with a descriptive research tier; robustness evidence
does not act as an all-or-nothing deletion gate.

The command writes ranked strategy and fold tables, raw strategy/theme/portfolio returns, a run
manifest, and correlations under `data/processed/broad_search/`. It also creates
`outputs/broad_strategy_search_tear_sheet.html` with QuantStats. The development-ranked portfolios
are research candidates, not holdout or live results; the sealed panel remains inaccessible to the
search.

## Regression-displacement research

`uv run research-regression-displacement` builds development-only hourly bars, 26 predeclared
regression/state features, multi-horizon return and first-passage targets, and fixed Ridge
walk-forward diagnostics for 8-hour, one-session, and five-session horizons. Labels must be
realized before each fold's training cutoff, daily regime inputs are lagged one session, and the
sealed holdout is never loaded.

Outputs are written under `data/processed/regression_displacement/`. They are predictive-research
evidence, not a promoted trading rule or portfolio backtest.

`uv run research-rates-reversion` performs the explicitly post-selection development retest for
ZT at 23 hours and ZN/ZT at 115 hours. It compares a displacement-only probability baseline with
a full regression-state model, then applies a one-hour delay, volatility sizing, position caps,
and conservative implementation costs. CL remains excluded from this mean-reversion branch.

## Combined candidate portfolio

`uv run research-candidate-portfolio` combines the costed development-stage strategy artifacts.
The 50 mean-reversion variations are first equal-weighted inside their five economic families so
families with more parameter variations cannot receive more capital. It also evaluates a fixed
price-location breakout sleeve, the multi-asset trend baseline, intraday reversal, and screened CL
linear trend.

The command reports two distinct books. `all_sleeves_*` is a diagnostic showing what happens when
every researched idea receives capital. `candidate_sleeves_*` funds only sleeves with positive
aggregate development returns and at least two positive predefined outer folds. That is a
permissive discovery gate, not a live-trading approval. Selection and scoring both use development
folds, so a candidate must still pass the future locked sequential evaluation before promotion.
All source sleeves retain their stated execution delays and costs; no FDR gate is used.

## Market-data intake gate

`config/data_sources.yaml` points to the current Databento three-year, one-minute portfolio
download. Raw exports are read-only. The VS Code task **Data: import Databento portfolio** runs the
strict catalog, normalization, instrument reconciliation, panel build, and physical temporal split
in sequence. Open `systematic-portfolio.code-workspace` to show the research repository and the raw
Databento folder together without copying or editing licensed source files.

The Databento files are UTC, unadjusted, volume-rolled front-contract series. They are normalized
separately under `data/processed/databento_canonical/`; their development and sealed evaluation
panels live under `data/processed/databento_research/`. This prevents them from silently replacing
the earlier Investor/RT research snapshot.

The catalog supports both Databento continuous exports and the earlier Investor/RT format.
After adding a file, run `uv run research-data`. The command writes a local catalog under
`data/catalog/databento/` containing SHA-256 identities, one-file-per-symbol-and-data-type enforcement,
filename-to-payload agreement, schema and timestamp checks, duplicate detection, OHLC validity,
and coverage warnings.

Run `uv run research-data --strict` before promoting data into a normalized research dataset. It
exits unsuccessfully while any critical or high-severity issue remains. Generated catalogs stay
local because they contain machine-specific paths and change as exports arrive.

## Portfolio research stack

- **QuantStats:** performance statistics and HTML tear sheets once valid returns exist
- **PyPortfolioOpt:** covariance estimators, HRP, Black-Litterman, and constrained allocation
- **VectorBT Portfolio:** order, trade, stop, target, fee, and slippage accounting for strategy
  backtests
- **ARCH:** conditional-volatility research and diagnostics
- **exchange-calendars:** session-aware completeness and timestamp alignment
- **Pandera and Pydantic:** dataframe and configuration contracts

`uv run research-strategy-factory` is the primary product-level search. It reads only the ten
Databento development partitions, executes next-bar orders through VectorBT, fits PyPortfolioOpt
weights on the selection window, and saves a timestamped run with its full trade ledger, source
fingerprint, parameters, returns, weights, and tear sheet. Resampled 30-, 60-, and 240-minute bars
are cached and invalidated whenever the source file identities change. QuantStats is used only for
the final HTML report. The former custom factory remains available as
`uv run research-legacy-strategy-factory` for comparison and must not be used for promotion.

## Canonical minute-bar layer

`uv run research-normalize` promotes only quality-approved minute bars into partitioned,
Zstandard-compressed Parquet under `data/processed/canonical/symbol=<contract>/bars.parquet`.
Every row retains the raw symbol, parsed contract root/month/year, local and UTC timestamps,
source filename, and source SHA-256. Uniformly zero open interest becomes null rather than a false
measurement. Blocked inputs are recorded as quarantined in `normalization_manifest.csv`; raw files
are never edited.

The current `America/New_York` export timezone is an explicit working assumption inferred from
the futures session times. Confirm it in Investor/RT before using cross-asset timestamps for
signals or execution modeling.

## Instrument master and roll policy

`config/instruments.yaml` is the controlled source for contract multipliers, tick sizes, tick
values, asset classes, exchanges, settlement types, valid delivery months, and roll policy. Run
`uv run research-instruments` after adding a new market; the command fails if any observed contract
root lacks metadata.

Roll rules are configured as two-session volume crossover with product-specific calendar
fallbacks. Actual continuous-contract construction is intentionally deferred until adjacent
expiries exist. Session labels currently use the 18:00–17:00 New York Globex convention; holiday
and early-close overrides must come from exchange calendars before production scheduling.

## Minute and session research panels

`uv run research-panel` creates gap-aware one-minute returns for each normalized symbol and a
cross-market session panel. Returns are calculated only across exactly consecutive minutes inside
the same trading session: the pipeline never forward-fills missing prices and never treats a data
gap or session boundary as a one-minute return. It also writes observed gap counts, descriptive
annualized volatility, and pairwise session-return correlations under `data/catalog/`.

These are descriptive research inputs, not signals or backtest results.

## Sealed temporal validation

`config/validation.yaml` reserves the latest 12 months as a physically separate holdout, exceeding
the six-month minimum. A five-day embargo separates development from evaluation. Development code
uses `load_development_panel()` and is denied holdout access; only the future sequential evaluator
may unlock the sealed panel. Correlations and volatility diagnostics are calculated from the
development period only. Random train/test splitting is prohibited for market data.

The eventual simulator will reveal sealed observations in chronological order and update strategy
state only after each observation, matching a live decision process. Creating the split does not
run that simulation or expose holdout results.

Combined full-history research panels are deleted after splitting. Minute and session data are
stored in separate development and sealed-holdout locations; ordinary research access defaults to
development only. Raw and canonical layers retain full provenance but are not strategy APIs.

## Quantitative feature layer

`uv run research-features` reads only the guarded development session panel and development
minute-return partitions. It writes `data/processed/features/development_session_features.parquet`
with point-in-time, close-available features and no prediction targets. The first feature set
contains log returns, volatility-standardized 5/20/60-session returns, trailing volatility and
volume state, realized variance and volatility, upside/downside semivariance, bipower variation,
jump variation, and realized quarticity. Missing minute intervals are not bridged when calculating
bipower variation.

These are statistical state variables rather than trading rules. Any future forecast or position
must lag them beyond `feature_available_at_utc`; feature selection and model fitting may use the
development partition only.

Backtesting frameworks and vendor SDKs are intentionally omitted. Start with the included
vectorized, tested portfolio logic; add integrations only after execution timing, contract rolls,
costs, data licenses, and asset coverage requirements are concrete.
