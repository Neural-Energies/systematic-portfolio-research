# Results and Limitations

## Selected candidate

The highlighted candidate combines three session-reversal rules:

| Product | Rule | Portfolio weight |
|---|---|---:|
| Crude oil (CL) | Session reversal, 20-session lookback, 1.5 threshold | 41.66% |
| Japanese yen (6J) | Session reversal, 1-session lookback, 1.5 threshold | 50.00% |
| Natural gas (NG) | Session reversal, 40-session lookback, 0.5 threshold | 8.34% |

The rules avoid overnight exposure and use same-session open-to-close returns. Portfolio leverage was frozen before sealed evaluation.

## Development robustness

At the modeled baseline of 1 bp per side, the development sample produced a 2.52 Sharpe ratio. The result remained above 2.2 Sharpe at 3.5 bps per side. All four predefined development folds were profitable, with fold Sharpe ratios of 1.72, 2.05, 3.72, and 2.78. A block bootstrap placed the fifth percentile of development Sharpe at 1.76.

These results measure robustness within the development period; they are not independent out-of-sample evidence.

## Sealed evaluation

The one-year sealed holdout was accessed once after the strategy and portfolio were frozen. It produced:

- 54.9% total return;
- 44.1% annualized return;
- 18.2% annualized volatility;
- 2.42 Sharpe ratio;
- -5.7% maximum drawdown;
- 76.9% positive months; and
- 80.0% positive quarters.

Saved component returns reproduce the portfolio exactly, with a maximum absolute difference of zero. First-half and second-half sealed Sharpe ratios were 2.16 and 2.66. The portfolio Sharpe remained 2.26 after removing the best day and 1.32 after removing the best five days.

## Material caveats

1. **Limited independent history.** The sealed period is one year and may not represent a complete market cycle.
2. **Profit concentration.** Crude oil supplied most sealed profit, while the Japanese yen sleeve was nearly flat.
3. **Tail-day dependence.** Removing the ten best sealed days reduced total return to approximately zero.
4. **Volatility drift.** Sealed volatility was 18.2% versus the 10% development target because leverage was frozen rather than refitted.
5. **Modeled execution.** Costs use basis-point assumptions rather than broker-specific fills. Market impact, financing, taxes, margin, and operational failures are not modeled.
6. **Continuous-series limitation.** Contract identifiers in the local continuous files are insufficient for a production roll and execution study.
7. **Benchmark limitation.** The S&P 500 comparison uses a price index and excludes dividends.

## Research decision

The candidate passed the stated research audit and sealed Sharpe target. The appropriate next stage is paper trading and contract-level execution validation. The evidence does not support immediate live-capital deployment.
