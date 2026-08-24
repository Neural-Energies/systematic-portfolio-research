# NG — daily_0442

This folder is the self-contained record for one component of the frozen
`stable_daily_session_2026-08-23` portfolio.

## Rule

- Product: NG
- Family: session_reversal
- Lookback: 40 sessions
- Entry threshold: 0.5
- Signal observed: session d close
- Execution: session d+1 open to session d+1 close
- Overnight exposure: none
- Frozen portfolio weight: 0.083411579434
- Frozen portfolio leverage: 1.738167350607

## Saved performance

- Development raw Sharpe: 1.2380
- Development weighted-contribution return: 17.47%
- Sealed raw Sharpe: 1.2366
- Sealed weighted-contribution return: 11.50%
- Sealed weighted-contribution Sharpe: 1.2366

`development_returns.parquet` and `sealed_returns.parquet` contain both the
raw strategy return and its contribution after applying the frozen portfolio
weight and leverage. No strategy was refit or rerun while creating this folder.
