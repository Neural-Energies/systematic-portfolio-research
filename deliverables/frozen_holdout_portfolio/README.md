# Systematic Futures Portfolio

Native Python implementation of seven frozen futures rule sets for 6E, 6J,
CL, ES, GC, HG, and NG. The package creates point-in-time stop-order signals;
it does not connect to a broker or transmit live orders.

## Research status

The rules were selected in an automated strategy-research environment and then
frozen before the holdout test. They were not independently invented in this
repository. The untouched test covers 2025-08-22 through 2026-08-21 using H1
signals, one-minute execution precision, one contract per strategy, and one
tick of slippage. The retest task requested a $5 size-based commission, but
trade-ledger reconciliation shows an effective $7 round-turn charge on every
ordinary closed trade. The published performance uses the ledger's more
conservative $7 charge; the source of the $2 configuration difference remains
an explicit reconciliation item.

The reference holdout produced $194,682 net profit on $700,000 initial capital,
1,471 closed trades, a 1.1765 profit factor, and a 1.89 daily Sharpe (252-day
annualization). It clears the subsequently adopted 1.5 deployment-candidate
threshold, but it did **not** meet the original 2.0 aspirational target. The 1.5
cutoff was set after the holdout result was observed and therefore is not
presented as predeclared. Nothing in this repository should be treated as final
deployment approval or an assurance of future performance.

The completed visual review is bundled at
`reports/holdout_visual_report/holdout_visual_report.html`. It includes the
closed-trade equity curve, drawdown, monthly results, strategy contributions,
trade distribution, and exact strategy-level audit table.

## Input data

CSV files are headerless and contain:

`date,time,open,high,low,close,volume`

Timestamps must be UTC. Raw market data is deliberately excluded from version
control. Continuous-contract construction, rolls, exchange calendars, and
symbol mapping must be approved by the firm's market-data owner before use.

To regenerate the report data locally, point `FUTURES_HOLDOUT_REFERENCE` at the
private directory containing `6E/strategy_6E.csv` through
`NG/strategy_NG.csv`, then run:

```powershell
$env:FUTURES_HOLDOUT_REFERENCE = "C:\private\holdout_reference"
python scripts/build_holdout_visual_report.py
```

The private ledgers are required for reconciliation and are intentionally not
included in this repository.

## Generate candidate orders

```powershell
python -m systematic_futures.cli signals --symbol 6E --input C:\data\DB_6E_M1.csv --output signals_6E.csv
```

Output fields are `timestamp_utc`, `model_version`, `strategy_id`, `symbol`,
`action`, `side`, `order_type`, `quantity`, `stop_price`, `valid_bars`,
`valid_until_utc`, the proposed protective target and stop prices,
`profit_target_atr_multiple`, `profit_target_atr_period`,
`stop_loss_atr_multiple`, `stop_loss_atr_period`, and `reason`.

Candidate orders are instructions for a downstream, independently controlled
order-management layer. That layer remains responsible for tick rounding,
contract mapping, position/risk limits, duplicate suppression, Friday exit at
20:40 UTC, protective orders, reconciliation, and kill-switches.

## Validation gates

Before any firm handoff:

1. Match development-period reference entries and exits without changing the
   frozen parameters.
2. Run the final holdout parity comparison once.
3. Reconcile fees, slippage, contract point values, sessions, and roll logic.
4. Run a paper/shadow period with daily signal and fill reconciliation.
5. Obtain independent risk and code review.
