# Results and Limitations

## A-Tier Strategy 1

Seven frozen strategies were evaluated from August 22, 2025 through August 21, 2026. Five of seven were individually profitable.

| Market | Trades | Net profit | Profit factor | Win rate |
|---|---:|---:|---:|---:|
| GC | 635 | $150,995.00 | 1.246 | 58.7% |
| CL | 119 | $27,077.00 | 1.343 | 55.5% |
| ES | 203 | $14,911.00 | 1.071 | 47.8% |
| HG | 93 | $7,031.00 | 1.093 | 59.1% |
| 6J | 37 | $559.75 | 1.169 | 70.3% |
| NG | 321 | -$2,607.00 | 0.977 | 49.5% |
| 6E | 63 | -$3,284.75 | 0.583 | 44.4% |

A-Tier Strategy 1 produced a hypothetical $194,682 net profit on $700,000 initial capital, a 27.81% return, 1.89 daily Sharpe, 1.1765 profit factor, 54.66% win rate, and 1,471 closed trades.

The frozen artifacts contain two differently aggregated drawdown statistics. The chronological daily closed-trade series reports -$53,353.25 (-7.37%). The visual report's trade-sequence series reports -$56,463.25 (-8.07%). Both are retained; the difference is an unresolved aggregation/reconciliation item rather than silently choosing one definition.

## CL / 6J / NG sealed candidate

The separate custom-research candidate combines three session-reversal rules:

| Product | Rule | Portfolio weight |
|---|---|---:|
| Crude oil (CL) | Session reversal, 20-session lookback, 1.5 threshold | 41.66% |
| Japanese yen (6J) | Session reversal, 1-session lookback, 1.5 threshold | 50.00% |
| Natural gas (NG) | Session reversal, 40-session lookback, 0.5 threshold | 8.34% |

Its one-year sealed result was 54.9% total return, 44.1% annualized return, 18.2% annualized volatility, 2.42 Sharpe, and -5.7% maximum drawdown. Ten of thirteen months and four of five quarters were positive.

## Material limitations

1. **Hypothetical results.** Neither portfolio represents live trading performance.
2. **Limited independent history.** Each highlighted holdout covers one year and may not represent a complete market cycle.
3. **Contribution concentration.** GC generated approximately 77.6% of the seven-strategy portfolio's net profit. CL generated most of the custom portfolio's sealed profit.
4. **Individual failures.** NG and 6E were unprofitable in the seven-strategy holdout; the 6J sleeve was nearly flat in the custom portfolio.
5. **Tail dependence.** The custom portfolio's result depended materially on its best days.
6. **Modeled execution.** Slippage and costs are modeled. Live fills, queue position, spread changes, financing, taxes, margin constraints, and operational failures may differ.
7. **Fee reconciliation.** The multi-market task requested a $5 commission, while the ledgers show an observed $7 round-turn cost. Published performance uses the more conservative observed amount.
8. **Drawdown reconciliation.** Daily aggregation and trade-sequence reporting produce different maximum drawdowns in the frozen artifacts.
9. **Contract and roll risk.** Continuous futures series require contract mapping, exchange-calendar, expiry, and roll validation before deployment.
10. **Market impact.** One-contract tests do not establish capacity or impact at institutional size.
11. **Post-result criterion.** The seven-strategy portfolio did not meet its original 2.0 Sharpe aspiration. A 1.5 candidate threshold was adopted after the result and is not represented as predeclared.

## Research decision

Both portfolios are research candidates. The appropriate next stage is paper/shadow trading, broker-fill reconciliation, contract-level roll validation, independent code and risk review, and a new forward sample. The evidence does not support immediate live-capital deployment.
