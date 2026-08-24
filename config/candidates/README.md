# Saved candidate baselines

This directory preserves exact, reproducible development candidates and their artifact hashes.

`theme_champions_baseline_2026-08-23.yaml` is the first positive multi-theme Databento portfolio.
It is a saved comparison baseline, not the final candidate frozen for holdout evaluation. The
sealed year remains unopened while the intraday and portfolio-weighting expansions continue.

`combined_session_intraday_2026-08-23.yaml` freezes the first materially improved development
candidate. Its aggregate development Sharpe is 2.0466 using a 120-session lagged inverse-volatility
combination of the saved session baseline and the costed intraday lead-lag sleeve.

The one authorized sealed-year evaluation has now been executed. It returned -2.82% total with a
-0.5181 Sharpe and -5.88% maximum drawdown. The candidate is not live-ready, the holdout run count
is permanently one, and the sealed year must not be used for reselection or rerun.
