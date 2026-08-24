# A-Tier Strategy 1

Interview-facing package for seven frozen futures rule sets covering 6E, 6J, CL, ES, GC, HG, and NG. The package creates point-in-time stop-order signals; it does not connect to a broker or transmit live orders.

## Holdout result

The rule sets were frozen before the holdout test. The untouched test covers August 22, 2025 through August 21, 2026 using hourly signals, one-minute execution precision, one contract per strategy, and one tick of slippage.

| Metric | Result |
|---|---:|
| Initial capital | $700,000 |
| Net profit | $194,682 |
| Return | 27.81% |
| Closed trades | 1,471 |
| Daily Sharpe | 1.89 |
| Profit factor | 1.1765 |
| Win rate | 54.66% |
| Average trade | $132.53 |
| Chronological daily closed-trade max drawdown | -$53,353.25 / -7.37% |

The reference ledgers show an observed $7 round-turn cost on ordinary closed trades, compared with a requested $5 commission setting. Published performance uses the more conservative observed cost; the $2 difference remains an explicit reconciliation item.

The result did not meet the original 2.0 Sharpe aspiration. A 1.5 deployment-candidate threshold was adopted after the holdout result was observed and is therefore not presented as a predeclared pass criterion.

## Individual strategies

Five of seven individual strategies were profitable during the holdout period.

| Market | Trades | Net profit | Profit factor | Win rate |
|---|---:|---:|---:|---:|
| GC | 635 | $150,995.00 | 1.246 | 58.7% |
| CL | 119 | $27,077.00 | 1.343 | 55.5% |
| ES | 203 | $14,911.00 | 1.071 | 47.8% |
| HG | 93 | $7,031.00 | 1.093 | 59.1% |
| 6J | 37 | $559.75 | 1.169 | 70.3% |
| NG | 321 | -$2,607.00 | 0.977 | 49.5% |
| 6E | 63 | -$3,284.75 | 0.583 | 44.4% |

GC contributed approximately 77.6% of combined net profit. The combined result is diversified by strategy count and market coverage, but not evenly by profit contribution.

## Evidence map

- [`reports/holdout_summary.json`](reports/holdout_summary.json): frozen portfolio and strategy statistics
- [`reports/holdout_visual_report/holdout_visual_report.html`](reports/holdout_visual_report/holdout_visual_report.html): equity, drawdown, monthly results, contributions, trade distribution, and strategy audit table
- [`reports/holdout_visual_report/validation.json`](reports/holdout_visual_report/validation.json): report validation record
- [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md): source commit, integrity boundary, and editorial changes
- [`src/systematic_futures`](src/systematic_futures): point-in-time signal implementation
- [`tests`](tests): indicator and look-ahead controls

## Drawdown definitions

The frozen summary's chronological daily closed-trade series reports -$53,353.25 (-7.37%). The visual report's trade-sequence series reports -$56,463.25 (-8.07%). Both frozen values are preserved. They use different aggregation/order definitions and remain an explicit reconciliation item.

## Deployment boundary

Before any live-capital use:

1. Reconcile fees, slippage, contract point values, sessions, and roll logic.
2. Complete contract-level expiry and exchange-calendar validation.
3. Run a paper/shadow period with daily signal and fill reconciliation.
4. Evaluate capacity and market impact above one contract.
5. Obtain independent risk and code review.

These are hypothetical backtest results and should not be treated as final deployment approval or an assurance of future performance.
