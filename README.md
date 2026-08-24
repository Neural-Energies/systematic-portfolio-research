# Systematic Portfolio Research

An interview-focused Python research repository for systematic futures portfolio design. The project covers data validation, signal research, bias-aware backtesting, portfolio construction, temporal validation, cost stress testing, and a single locked holdout evaluation.

> **Research status:** The highlighted portfolio passed the predefined research audit, including one sealed holdout evaluation. Results are hypothetical and support paper trading and contract-level execution validation—not immediate live deployment.

## Highlighted result

The selected candidate is a three-rule, cross-market session-reversal portfolio using crude oil, Japanese yen, and natural gas futures signals. Parameters and weights were frozen before the sealed year was accessed.

| Metric | Sealed holdout |
|---|---:|
| Period | 2025-08-21 to 2026-08-21 |
| Total return | 54.9% |
| Annualized return | 44.1% |
| Annualized volatility | 18.2% |
| Sharpe ratio | 2.42 |
| Maximum drawdown | -5.7% |
| Positive months | 10 of 13 (76.9%) |
| Positive quarters | 4 of 5 (80.0%) |

![Hypothetical sealed holdout equity and drawdown](docs/assets/sealed_holdout_performance.png)

The sealed result was run once, with no refitting or reselection. The [post-sealed audit](saved_strategies/BEST_STRATEGIES/tier_a_sealed_validated/stable_daily_session_2026-08-23/post_sealed_audit.json) records independent reproduction, subperiod, bootstrap, concentration, cost, and data-quality checks. The complete [audit report](saved_strategies/BEST_STRATEGIES/tier_a_sealed_validated/stable_daily_session_2026-08-23/AUDIT_REPORT.html) and [candidate index](saved_strategies/BEST_STRATEGIES/tier_a_sealed_validated/stable_daily_session_2026-08-23/MASTER_INDEX.md) are included.

## What this repository demonstrates

- Point-in-time feature engineering and explicit execution delays
- Development/holdout separation with an embargo and physical access controls
- Strategy research across momentum, trend, mean reversion, breakout, relative value, lead/lag, and research-derived hypotheses
- VectorBT-based trade accounting and PyPortfolioOpt portfolio construction
- Turnover-based costs, cost stress tests, parameter-neighbor checks, walk-forward folds, and block-bootstrap analysis
- Reproducible environments with a locked dependency graph
- Automated tests covering data, signals, portfolios, temporal validation, and sealed evaluation

## Repository guide

- [`src/systematic_research`](src/systematic_research): reusable Python research and backtesting modules
- [`tests`](tests): unit, property-based, and regression tests
- [`config`](config): research, validation, universe, instrument, and portfolio assumptions
- [`saved_strategies/BEST_STRATEGIES`](saved_strategies/BEST_STRATEGIES): the selected portfolio and audit evidence
- [`outputs`](outputs): selected notebooks, diagnostics, and HTML tear sheets
- [`research_program`](research_program): hypothesis and experiment registries
- [`docs/RESEARCH_METHOD.md`](docs/RESEARCH_METHOD.md): research and validation design
- [`docs/RESULTS_AND_LIMITATIONS.md`](docs/RESULTS_AND_LIMITATIONS.md): result interpretation and material caveats
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): exact setup and verification commands

## Reproduce the code checks

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync --frozen
uv run ruff format --check src tests
uv run ruff check src tests
uv run python -m mypy src
uv run python -m pytest -q
```

Raw and licensed market data are intentionally excluded. Configuration examples and saved evidence make the research design reviewable without redistributing vendor data.

## Important limitations

- The sealed sample contains one year. It is independent, but still a limited regime sample.
- Crude oil generated most sealed profit; the portfolio is not broadly diversified.
- The ten best sealed days account for essentially all profit, although the portfolio remained strong after removing the best day and profitable after removing the best five days.
- Costs are modeled rather than based on broker fills; financing, taxes, margin constraints, and market impact are excluded.
- Continuous-series inputs require contract-level execution validation before live use.

See [Results and Limitations](docs/RESULTS_AND_LIMITATIONS.md) for the complete interpretation.

## Disclaimer

This repository documents quantitative research and hypothetical backtests. It is not investment advice, a solicitation, or a claim of live trading performance.
