# Research Learning Log

## Pass 0 — Broad random grids

- Large randomized parameter grids produced visually impressive results but
  counted configurations rather than independent hypotheses.
- The lesson is to aggregate variants at the family level and record the
  effective number of ideas separately from raw configurations.
- Generic continuous futures labels do not provide trustworthy contract IDs.
  Overnight and roll-gap returns are excluded until contract-level data is
  available.

## Pass 1 — Roll-safe session reversal

- Extreme, slow session reversal in CL and NG survived the one-shot sealed
  evaluation; the three-rule portfolio achieved a sealed Sharpe of 2.42 and a
  76.9% monthly hit rate.
- Profit was concentrated: CL supplied most performance, 6J was nearly flat,
  and the ten best days explained essentially all net profit.
- The next pass prioritizes genuinely distinct mechanisms and family breadth,
  weekly consistency, capped product contribution, and outlier-resistant
  scores rather than further optimization of the winning reversal family.

## Pass 2 — Registered before testing

- H002 through H006 were registered before performance inspection.
- Parameter variants will be treated as sensitivity tests under their parent
  hypothesis, not as separate discoveries.

## Pass 3 — H002 through H006 falsification

- The first diagnostic run exposed incorrect turnover accounting for persistent
  intraday positions: it charged a round trip on every active bar. The run at
  `20260823T203416Z` is retained but invalidated for decision use.
- Turnover was corrected to charge actual position transitions plus a forced
  close at each session boundary, with a synthetic regression test.
- The corrected preregistered run at `20260823T204208Z` tested 2,880 parameter
  configurations representing five hypotheses on 15, 30, 60, and 240-minute
  bars. Configurations were selected only from earlier chronological folds.
- The corrected portfolio lost 25.36%, with Sharpe -2.63, 34.6% winning weeks,
  and 21.1% winning months at 3.5 bps per side. The zero-cost replay was also
  negative, so costs were not the main explanation.
- All five family sleeves had negative out-of-fold Sharpe. Strong early-fold
  variants reversed later, a direct example of winner's curse. H002-H006 are
  rejected as implemented and may not be renamed or recycled as new ideas.
- Next-pass selection will require positive family-level training expectancy;
  weak families will not receive portfolio weight merely to satisfy breadth.

## Pass 4 — H007 through H011 falsification

- Pass 2 tested 1,160 variants representing five new ideas. Positive-score
  eligibility reduced forced allocation to weak candidates but did not solve
  out-of-fold reversal.
- The costed walk-forward portfolio Sharpe was -2.89; zero-cost Sharpe was
  -1.23. The failure therefore precedes transaction costs, though the four
  intraday families were also economically untradeable at 3.5 bps per side.
- Equal-weight family diagnostics showed only H010 slightly positive before
  costs (gross Sharpe 0.45); it was deeply negative after turnover costs.
- All H007-H011 implementations were rejected. No result was promoted.

## Pass 5 — Lower-turnover H012 through H016

- Pass 3 reduced the grid to 220 variants and ran in 12.4 seconds. It tested
  five fixed-mechanism, lower-turnover ideas, including 15-minute overnight and
  realized-skewness signals.
- The costed walk-forward portfolio Sharpe was -1.72 and the zero-cost Sharpe
  was -0.20. All three evaluation folds lost money.
- H016 realized skewness was approximately flat out of fold (Sharpe 0.03); the
  remaining family sleeves were negative. None qualified for Tier C.
- Canonical parquet files expose contract month/year fields in their schema,
  but every value is null. Source filenames also contain only generic root
  symbols. Contract-consistent multi-session and paper-faithful close-to-close
  testing therefore remains blocked; session-only adaptations are labeled as
  such and not overstated.

## Paper-faithful H008 refinement

- The exact ES European-open window (23:30 through 03:30 New York time) was
  tested as a labeled refinement, not counted as a new hypothesis.
- Development gross Sharpe was 0.26. It fell to -0.46 at 0.5 bp per side and
  -4.79 at 3.5 bps; all four chronological folds lost after 3.5 bps.
- The result rejects this implementation and prevents a loose session-window
  adaptation from being presented as support for the paper's effect.

## Pass 6 — H017 pooled machine learning

- Ridge, histogram gradient boosting, and their fixed ensemble used only
  nonoverlapping information available by 10:00 New York time to predict the
  10:00-16:00 return. Selection and scaling were fitted inside expanding
  chronological folds.
- At 1 bp per side, ensemble Sharpe was -1.59 and total return was -24.86%; the
  zero-cost Sharpe was still -1.15. Ridge and nonlinear sleeves were both
  negative.
- Prediction-target correlations were -0.03 to -0.04 and sign accuracy was
  approximately 50%. This is absence of signal, not evidence that an
  after-the-fact inversion should be promoted.

## Pass 7 — H018 CFTC producer hedging pressure

- Official CFTC futures-only disaggregated reports for 2023-2025 were archived
  with source URLs and SHA-256 fingerprints. Tuesday positions were delayed to
  Friday publication and traded only on the following available session.
- The fixed cross-sectional long-high/short-low portfolio had gross Sharpe
  -0.25 and Sharpe -0.36 after 1 bp per side. Only 44.2% of weeks and 36.0% of
  months were positive.
- Three of four chronological folds lost money and bootstrap median Sharpe was
  -0.33. H018 failed every preregistered gate and remains a rejected archive.

## Pass 8 — H019 PCA residual reversal

- The initial causal PCA pass had a small positive zero-cost return, but only
  Sharpe 0.12. At 1 bp per side, Sharpe fell to -0.67 and two of three chained
  evaluation folds lost money.
- H019-R1 addressed the diagnosed turnover problem with higher residual
  thresholds and two-bar confirmation across all four bar sizes. It remained
  one hypothesis despite 48 parameter sensitivities.
- The refinement removed the gross edge (Sharpe -0.01 before costs) and had
  Sharpe -0.58 at 1 bp. The mechanism is rejected as implemented; neither run
  was promoted.

## Pass 9 — H020 managed-money diffusion

- Managed-money net-long share was tested in the paper-motivated same
  direction over the next five sessions, with one- and three-session holds
  retained only as labeled sensitivity variants.
- The primary five-session portfolio had Sharpe -0.47 after 1 bp per side,
  45.2% winning weeks, and 44.0% winning months. Fewer than two chronological
  folds were positive and the bootstrap median Sharpe was -0.45.
- H020 is rejected. The direction will not be reversed after observing the
  result and presented as a new discovery.

## Pass 10 — H021 regularized VAR network

- A rolling multi-output ridge VAR learned the complete cross-futures
  lead-lag matrix from prior sessions and forecast the next 15-, 30-, 60-, or
  240-minute bar. Thirty-two parameter sensitivities remained one hypothesis.
- The chained portfolio had Sharpe -1.41 at 1 bp per side and was negative
  even before costs. Its bootstrap 95th-percentile Sharpe was -0.04.
- H021 is rejected. The full-network adaptive model did not rescue the earlier
  fixed-pair lead-lag failure.

## Pass 11 — H022 EIA release drift

- Standard EIA petroleum Wednesday and natural-gas Thursday 10:30 ET release
  clocks were tested after excluding federal-holiday-disrupted weeks and a
  known exceptional release. Response and target windows were nonoverlapping.
- The chained CL/NG event portfolio had Sharpe -0.58 at 1 bp per side, 32.1%
  winning weeks, and negative gross expectancy. The bootstrap median Sharpe
  was -0.55.
- H022 is rejected. A post-release reversal would be a separately registered
  economic hypothesis, not an after-the-fact sign flip of this result.

## Pass 12 - H024 dependency-network relative value (OpenCode)

- Registered before testing: fade cross-sectional dislocations between each
  product's standardized momentum and a trailing correlation-graph Laplacian
  estimate, held as overlapping 1..K session cohorts. Sixteen sensitivity
  variants across correlation window, momentum window, threshold, and hold.
- The chained out-of-fold portfolio lost 15.6 percent with Sharpe -0.94 at
  1 bp per side; gross expectancy was already negative, so costs were not the
  binding failure. Bootstrap median Sharpe was -0.99.
- Full-period variant diagnostics: best region (w60, m5, t1.0, k3) reached
  only gross Sharpe 0.44; the low-threshold variant dominated its high-threshold
  sibling in every matched pair, and short momentum windows beat long ones.
- H024 is rejected as formulated. Preserved observation: session-scale network
  dislocation reversion, if present, lives in the body of the cross-section
  rather than the tails - the opposite of the Tier A own-product extreme
  reversal. Any successor must start from positive family-level training
  expectancy before portfolio weight.
