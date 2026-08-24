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
