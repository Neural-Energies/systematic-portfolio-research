# 6J — daily_0100

This folder is the self-contained record for one component of the frozen
`stable_daily_session_2026-08-23` portfolio.

## Rule

- Product: 6J
- Family: session_reversal
- Lookback: 1 sessions
- Entry threshold: 1.5
- Signal observed: session d close
- Execution: session d+1 open to session d+1 close
- Overnight exposure: none
- Frozen portfolio weight: 0.500000000000
- Frozen portfolio leverage: 1.738167350607

## Saved performance

- Development raw Sharpe: 1.5405
- Development weighted-contribution return: 11.86%
- Sealed raw Sharpe: 0.0840
- Sealed weighted-contribution return: 0.22%
- Sealed weighted-contribution Sharpe: 0.0840

`development_returns.parquet` and `sealed_returns.parquet` contain both the
raw strategy return and its contribution after applying the frozen portfolio
weight and leverage. No strategy was refit or rerun while creating this folder.
