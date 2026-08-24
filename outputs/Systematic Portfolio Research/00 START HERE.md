# Systematic Portfolio Research

This is the organized, user-facing research bundle. Start with:

1. `01 Research Foundation/Systematic Research Foundation.ipynb`
2. `04 Portfolio Readiness/portfolio_design_readiness.ipynb`
3. `06 Candidate Portfolio/portfolio_design_06_combined_candidate.ipynb`

## Folder guide

- `01 Research Foundation` — the comprehensive, executed record of everything completed.
- `02 Data Quality and EDA` — raw-data inventory, data-quality audit, market EDA, and instrument-quality tables.
- `03 Features and Diagnostics` — feature engineering audit, missingness, drift, redundancy, and the final feature registry.
- `04 Portfolio Readiness` — preprocessing readiness and development-only walk-forward folds.
- `05 Universe and Symbols` — the proposed liquid universe and Rithmic symbol reference.
- `06 Candidate Portfolio` — the executed combined development portfolio, fold results, QA, and caveats.

## Important status

- Development data ends August 15, 2025.
- The sequential holdout remains sealed.
- A costed development candidate has been built from CL linear trend and the moving-average
  mean-reversion family; it is not approved for live trading.
- The equal-weight candidate returned 11.0% annualized at 6.1% volatility with a 1.81 development
  Sharpe across 226 validation sessions, but candidate selection reused development folds.
- ES and GC need older development data.
- 6J is quarantined because its supplied price series is invalid.
- Investor/RT continuation-series roll and adjustment rules still require confirmation.

The working source code, configurations, tests, and generated datasets remain in the main project tree so the VS Code build tasks continue to work.
