# Continuous Systematic Strategy Research Protocol

## Objective

Continuously research, falsify, backtest, audit, and archive distinct systematic
futures strategies. The portfolio target is a net Sharpe ratio above 2.0, at
least 70% winning weeks or months, useful trade frequency, and evidence strong
enough to discuss with a quantitative investment firm.

The target is an aspiration, not permission to alter methodology until a result
passes. Failed and negative hypotheses remain in the research record.

## What Counts as a Unique Strategy Idea

A strategy family is unique only when its predictive mechanism differs from
previous families. Changing a lookback, threshold, holding period, stop,
timeframe, volatility filter, or allocation weight creates a parameter variant,
not a new idea.

Every new hypothesis must record:

1. a stable hypothesis ID;
2. the proposed economic or statistical mechanism;
3. the observable signal and target return;
4. the information timestamp and executable entry timestamp;
5. the closest prior hypothesis and the material novelty;
6. a primary academic or technical source when one exists;
7. a falsification condition stated before testing;
8. all tested variants, including failures.

Ideas with the same signal primitive, conditioning variables, traded target,
and execution logic are one family even if thousands of configurations are run.

## Research Loop

1. **Research:** Review primary literature and identify a mechanism compatible
   with the available fields and instruments.
2. **Register:** Assign the hypothesis ID and record novelty and falsification
   criteria before seeing performance.
3. **Implement:** Create causal signals with explicit observation, entry, exit,
   holding, cost, and roll rules.
4. **Unit test:** Prove signal lagging, session boundaries, transaction costs,
   and absence of future information on synthetic examples.
5. **Development search:** Explore parameter variants only within the declared
   family and only on development data.
6. **Nested validation:** Select variants on earlier folds and score them on
   later chronological folds. Never select on the final evaluation fold.
7. **Family aggregation:** Treat neighboring variants as one evidence cluster;
   do not count correlated configurations as independent discoveries.
8. **Portfolio construction:** Combine complementary families with fixed,
   concentration-capped weights learned only from permitted folds.
9. **Audit:** Independently reproduce returns and test costs, outliers,
   parameter neighborhoods, subperiods, product concentration, missing data,
   and multiple-testing risk.
10. **Archive:** Save every family, configuration, result, failure, audit, code
    snapshot, input fingerprint, and promotion decision.
11. **Learn:** Record what failed, what survived, and how the next pass differs.
12. **Repeat:** Begin a new set of genuinely distinct hypotheses.

## Time and Data Rules

- The sealed period ending August 21, 2026 has already been consumed once. It
  may be reported but cannot be used to choose or tune future strategies.
- Future candidates are `development_only` until data strictly later than the
  consumed sealed period becomes available.
- Signals must be lagged to the next executable price.
- Session returns may not bridge unverified continuous-contract roll gaps.
- Intraday positions must be closed before the session boundary unless verified
  contract identifiers and explicit roll handling are available.
- Missing prices are not forward-filled into tradable returns.

## Promotion Tiers

### Tier A — Sealed validated

- frozen before evaluation;
- exactly one evaluation on untouched data;
- net Sharpe at least 2.0;
- at least 70% winning weeks or months;
- positive chronological subperiods;
- survives realistic cost and outlier stress;
- independently reproduced;
- implementation limitations documented.

### Tier B — Development robust

- net full-development Sharpe at least 2.0;
- at least 70% winning weeks or months;
- positive performance in every required outer fold;
- family-level parameter support;
- cost, concentration, and outlier audits pass;
- no access to a future holdout;
- explicitly labeled development-only.

### Tier C — Research candidate

- economically coherent and causally implemented;
- positive expectancy in more than one fold;
- sufficient frequency for further study;
- not yet strong enough for Tier A or B.

Rejected families stay in the negative-results registry and are not silently
reintroduced under new names.

## Portfolio Standards

- Maximum 50% weight per strategy and 60% per product.
- At least two genuinely distinct hypothesis families.
- Report raw and volatility-scaled results separately.
- Report weekly and monthly hit rates, trade count, active days, turnover,
  annualized return and volatility, Sharpe, drawdown, and tail days.
- Stress 0, 0.5, 1, 2, 3.5, and 5 basis points per side where appropriate.
- Report results after removing the best 1, 3, 5, and 10 days.
- Use block bootstrap confidence intervals and chronological fold results.
- Benchmark against the S&P 500 when dates overlap, while labeling price-only
  versus total-return comparisons.

## Interview-Ready Evidence

Every promoted strategy receives its own folder containing the hypothesis,
paper references, exact rules, causal timing diagram, parameter variants,
development and evaluation returns, trade ledger, costs, fold metrics,
robustness tests, limitations, code snapshot, and checksums. The best-strategies
index distinguishes verified evidence from claims that still require live or
contract-level validation.
