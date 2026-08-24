# Research Method

## Objective

Develop systematic futures signals individually, test them with realistic timing and costs, and combine complementary positive-expectancy rules into a portfolio. The search includes momentum, trend, mean reversion, breakout, relative value, lead/lag, and independently specified research hypotheses.

## Data and timing controls

- One-minute source data are normalized into session and intraday research panels.
- Signal information is lagged beyond its availability time before a position can earn returns.
- Missing minute intervals are not bridged when computing high-frequency return features.
- The latest twelve months are physically separated as a sealed holdout, with a five-day embargo.
- Search, model fitting, parameter selection, and portfolio weighting use development data only.

## Research pipeline

1. Validate timestamps, duplicates, OHLC consistency, symbol coverage, and missingness.
2. Build point-in-time features and predeclared signal variants.
3. Apply delayed execution, overlapping holding cohorts, and turnover costs.
4. Evaluate predefined temporal folds and parameter neighborhoods.
5. Combine strategies using constrained portfolio weights and correlation evidence.
6. Freeze the candidate definition, inputs, and weights.
7. Run the sealed period once without refitting or reselection.
8. Reproduce saved returns and apply post-holdout sensitivity checks.

## Backtesting controls

- VectorBT is used for product-level position and trade accounting where applicable.
- PyPortfolioOpt supports constrained allocation and covariance-aware portfolio research.
- QuantStats produces selected tear sheets after return streams are finalized.
- Every promoted artifact includes definitions, manifests, metrics, and checksums.
- Raw data and generated development panels are excluded from version control.

## Audit evidence

The selected portfolio package contains:

- frozen strategy definitions and portfolio weights;
- development and sealed return series;
- temporal-fold and parameter-neighbor audits;
- cost-stress and block-bootstrap results;
- a sealed-evaluation lock and source-data fingerprint;
- an independent post-sealed audit and reproducibility checksums.

The full experimental archive is retained separately from this interview-facing branch.
