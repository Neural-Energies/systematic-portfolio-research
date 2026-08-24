# Current Research Status — 2026-08-23

## Best strategy retained

The only promoted strategy remains the sealed-validated Tier A daily-session
reversal portfolio at
`saved_strategies/BEST_STRATEGIES/tier_a_sealed_validated/stable_daily_session_2026-08-23`.
Its frozen sealed-year Sharpe is 2.42, total return is 54.92%, maximum drawdown
is -5.72%, monthly hit rate is 76.9%, and quarterly hit rate is 80.0%.

The sealed year has already been consumed once. It must not be accessed again
for selection, tuning, or comparison of new strategies.

## Distinct hypotheses completed

- H002-H006: five intraday asymmetry, opening, compression, lead-lag, and
  volume-confirmed mechanisms; rejected.
- H007-H011: five decomposition, clock, jump, liquidity-impact, and entropy
  mechanisms; rejected.
- H012-H016: five lower-turnover calendar, overnight, variance-ratio, and
  skewness mechanisms; rejected.
- H017: pooled regularized/nonlinear information-diffusion model; rejected.
- H018: producer hedging pressure from official CFTC reports; rejected.
- H019 and H019-R1: causal PCA residual reversal and turnover refinement;
  rejected.
- H020: CFTC managed-money information diffusion; rejected.
- H021: causal regularized full-network VAR lead-lag; rejected.
- H022: EIA scheduled-release post-announcement drift; rejected.

Parameter configurations are sensitivity tests, not unique hypotheses. All
completed passes have manifests, cost stress, chronological audits, source
snapshots, per-strategy folders, and SHA-256 inventories under
`saved_strategies/`.

## Next distinct research queue

1. H023: Bayesian online change-point detection applied to causal local trend,
   with state resets driven by posterior regime-change probability rather than
   a volatility filter.
2. H024: graph-Laplacian relative value using a prior-window futures dependency
   network, distinct from one-factor PCA residuals and fixed pairs.
3. H025: causal multiresolution Haar/wavelet trend separation, testing whether
   low-frequency directional energy predicts the next nonoverlapping bar.
4. H026: tail-dependence contagion using empirical copula exceedances, testing
   conditional continuation versus normalization after joint extreme moves.
5. H027: actual EIA inventory-change surprise relative to a causal seasonal
   model, contingent on obtaining timestamped official historical releases.

Each will be registered before performance inspection and rejected without
sign-flipping if its prespecified direction fails.
