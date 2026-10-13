# Stress test: "buy last year's best S&P 500 stock"

**Hypothetical backtest, not live results.** Every performance number below is a simulation on imperfect data (see "Data: read this first"). Full tables: `outputs/results_tables.md`. Code: `src/`. Re-run: `python3 -I src/01_download.py && python3 -I src/03_run_all.py && python3 -I src/04_charts_tearsheet.py`.

## UPDATE: reconciliation with the post's own exhibits (supersedes the "not reproduced" wording below for the 2012 version)

The post's screenshots show its year-by-year picks and returns (2012 → 2026 YTD to Oct 9). `src/06_reconcile_post.py` compares them with this data. Hypothetical backtest, not live results.

* **The 2012 claim reproduces.** My own ranking picks the same stock as the post in **14 of 15 years**. Compounding the post's picks on my prices gives **$177,161** (Dec-31-close entry, the post's convention) or **$174,561** (Jan-open entry) vs the post's **$176,114**; SPY **$800** vs the post's $800. The "11 of 15 years ahead" also holds (lagged only in 2013, 2014, 2016, 2023). My earlier $50,730 was a **data miss on my side, not a flaw in the post's arithmetic**: (1) 2012 pick **COG** (Cabot Oil & Gas, now CTRA) is no longer served by Yahoo, so my run picked ISRG (+7%) instead of COG (+31% per the post, which I cannot verify with Yahoo data); (2) the 2026 pick **SNDK** has only a partial 2025 (listed Feb 13, 2025), so my full-lookback rule skipped it (WDC, +131%).
* **What the headline hides:**
  * **The 2026 pick is a partial-year ranking.** SNDK's "2025 return" (+559%) runs only from its Feb-2025 spin-off listing. A strict full-prior-year rule picks WDC (+131% YTD), and the 2012 total falls from **$177,161 to about $61,377**. SNDK alone is ~65% of the headline.
  * **2026 is an open, unrealised 9-month position (+566%).** Through 12/2025 the same strategy is **$26,586** (SPY ~$683).
  * **Concentration:** the best 3 years (2026 SNDK, 2024 NVDA, 2019 AMD) are 51% of log-growth; without them $100 ends at **$3,945**.
  * **Start date:** the post shows 2012 and 1997 starts. My 1997-start tests give a CAGR of -14.6% to +15.7% depending on the start month, so the result is very start-date dependent.
* **Last 10 years only (2017–2026, post's picks):** $116,189 (Dec-31-close entry) / $113,993 (Jan-open) vs SPY ~$400. Through 2025 only: **$17,436 / $17,609**. With a full-prior-year rule for 2026 (WDC) it is about **$40,300** (Dec-31-close entry).
* **The 1997 and 35%-stop versions cannot be verified with yfinance.** The post says it hand-adjusted delisted names (Dynegy's 2000 gain, Reebok's 2001 return, AOL excluded) and used **price returns before 2012**; stops were filled **exactly at the stop price intraday** (it notes gaps would make real exits worse), versus next-open fills here. My 1997-sample numbers ($3,763 / $36,157) therefore reflect a Yahoo gap that is 35–66% missing before 2012, and are not a refutation. Notably, my stop years 2000 (QCOM), 2002 (NVDA), 2008 (NOV) and 2011 (NFLX) coincide with the post's four stop years (2000, 2002, 2008, 2011); the extra hits in my run (2001 EOG and 2009 HRB) fall in years where the post's picks were delisted names Yahoo does not carry.
* **Revised verdict on "does it replicate":** the 2012 numbers are arithmetically sound and reproducible. The skeptical points are about *robustness* (partial-year ranking in the final pick, concentration in a few years, unrealised YTD, no tradeable-at-scale or start-date test, intraday stop fills), not about a miscalculation.

## Data: read this first

| Item | What was used | Consequence |
|---|---|---|
| Membership | fja05680/sp500 point-in-time lists (original file to 2019-01-11, "Updated" file after). Members as of each signal date, incl. later-removed names | Good: no "today's list" bias |
| Prices | **yfinance** daily, auto-adjusted (dividends reinvested) | **Yahoo does not serve most delisted tickers.** 759 of 1,210 ever-members returned data |
| Ticker reuse | Original file tags departed names `-YYYYMM`; those never match Yahoo, so they count as *unavailable* rather than risk pulling a different company's history under a reused ticker (an early run picked a bogus `SNT`) | Conservative but removes real names |
| Coverage of PIT members with usable prices | 35% (Dec 1996) · 49% (2002) · 66% (2011) · 71% (2014) · 88% (2020) · 98% (2025) | **Every result is survivorship-biased in prices. Pre-2012 results are close to unusable; 2012+ is much better but still incomplete.** Not a Norgate/CRSP replacement; direction of the bias is unknowable (missing bankruptcies would hurt picks, missing buy-outs would help). Early picks (TJX '97, CINF '98) look implausible as the index's true best performers, consistent with real winners being absent |
| Delisting returns | A held stock whose series ends is sold at its last Yahoo close, then cash. Yahoo often truncates before bankruptcy wipe-outs | Flatters |
| Execution | Signal at last close of the prior month; buy next day's open (close as variant); 10 bps/side (50 bps stress); stop checked on closes, filled at the **next open** (gaps go through); cash earns 0%; Sharpe uses rf=0 | |
| Partial year | 2026 is YTD (to 2026-10-13). "End $" columns are shown both through 2026-10-13 and through 12/2025; all annual statistics use complete years only | |

Sanity check on the benchmark: SPY from 2012 ends at **$785** here vs the viral **$800**. Hypothetical backtest, not live results.

## Replication (Test 1) — Hypothetical backtest, not live results

| Variant (top-1, open-entry, 10 bps) | Viral claim | This run | Max DD | Beat SPY (complete yrs) | Stops |
|---|---|---|---|---|---|
| From 2012, no stop | $176,114; 11 of 15 yrs | **$50,730**; CAGR 47.8% (SPY $785, 14.7%) | -47.2% | 8/14 | 0 |
| From 2012, 50 bps | | $44,977 | -48.0% | 8/14 | 0 |
| From 2012, close-entry | | $45,050 | -46.5% | 9/14 | 0 |
| From 1997, no stop | $89,418; -97% DD | **$3,763**; CAGR 10.4% (SPY $1,738, 9.8%) | **-98.2%** | 14/29 | 0 |
| From 1997, 35% stop | $1,514,581; -59% DD; 4 stops | **$36,157**; CAGR 19.3% | **-82.4%** | 14/29 | 6 |

The headline dollar figures are **not reproduced** (3.5x lower from 2012, 24x lower from 1997 without stop, 42x lower with stop), and the 11-of-15 beat count is not reproduced (8/14). The -97% drawdown without a stop *is* reproduced (-98.2%) but the -59% with the stop is not (-82.4%). Costs are not the issue: 50 bps trims the 2012 result by ~11%. The 2012+ stop never triggered, so stop/no-stop are identical there.

## Rank sensitivity (Test 2) — Hypothetical backtest, not live results

| Basket (no stop) | 2012: End $ | Sharpe | Max DD | 1997: End $ | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| Top-1 | $50,730 | 1.07 | -47% | $3,763 | 0.45 | -98% |
| EW top-3 | $31,409 | 1.16 | -39% | $82,282 | 0.71 | -76% |
| EW top-5 | $6,172 | 0.97 | -38% | $8,686 | 0.55 | -76% |
| EW top-10 | $2,241 | 0.85 | -39% | $5,596 | 0.55 | -72% |

Returns are not monotone in rank: top-3 beats top-1 from 1997 and top-5 and top-10 beat only modestly. The "best stock" is not special; the outcome is driven by a handful of names.

## Stop sweep (Test 3) — whole grid, nothing selected; Hypothetical backtest, not live results

Top-1 from 1997, End $ (Max DD): none $3,763 (-98%) · 15% $1,185 (-82%) · 20% $4,047 (-90%) · 25% $29,179 (-85%) · **30% $48,100 (-79%)** · 35% $36,157 (-82%) · 40% $26,409 (-87%) · 45% $18,903 (-91%) · 50% $14,578 (-93%). From 2012: stops of 25–50% never trigger ($50,730 for all), 20% gives $14,010 and 15% gives $3,816. The 1997 "benefit" of a stop comes from ~6 specific blow-ups (QCOM 2000, NVDA 2002, NFLX 2011, ...) and is hump-shaped in the stop level, so no stop setting is robust. All 4 ranks x 9 stops x 2 costs x 2 samples are in `outputs/stop_sweep_full_grid.csv`.

## Statistics, DSR, bootstrap (Test 4) — Hypothetical backtest, not live results

170 distinct strategy configurations were run and counted as trials (36 = the rank x stop grid, used as a lower-N variant).

| Variant | Sharpe | Sortino | Worst yr | DSR (N=170 / 36) | Bootstrap 5th/median/95th of $100 |
|---|---|---|---|---|---|
| SPY from 2012 | 0.91 | 1.29 | -18.0% | - | $282 / $703 / $1,473 |
| Top-1 from 2012 | 1.07 | 1.66 | -6.4% | 0.98 / 1.00 | $2,617 / $23,164 / $243,203 |
| SPY from 1997 | 0.58 | 0.82 | -36.8% | - | $299 / $1,589 / $6,799 |
| Top-1 no stop from 1997 | 0.45 | 0.66 | -82.2% | 0.57 / 0.74 | **$5** / $1,991 / $367,315 (19.5% chance of ending below $100) |
| Top-1 35% stop from 1997 | 0.62 | 0.93 | -37.7% | 0.87 / 0.95 | $315 / $17,209 / $924,538 |

Over 1997–2025 the unstopped top-1 strategy is statistically indistinguishable from luck (DSR 0.57) and its Sharpe is below SPY's. The 2012+ DSR is high, but it measures only trial-selection luck on 14 years of data that has the best coverage and the most extreme winners; it does not correct for data bias. The bootstrap treats years as iid, which understates the tail.

## Concentration (Test 5) — Hypothetical backtest, not live results

| Variant | Best 3 years | % of log-growth | End $100 as reported → without them |
|---|---|---|---|
| Top-1 from 2012 | 2024 +176%, 2019 +160%, 2025 +137% | 51.6% | $24,113 → $1,418 (CAGR 27.3%; SPY ex-best-3: 10.8%) |
| Top-1 no stop from 1997 | 2024, 2019, 1999 | **99.8%** | $1,789 → **$101** (CAGR 0.0%) |
| Top-1 35% stop from 1997 | same | 55.9% | $17,186 → $966 (CAGR 9.1%) |
| SPY from 1997 | | 30.6% | $1,522 → $661 |

## Robustness (Test 6) — Hypothetical backtest, not live results

Starting in a different month changes top-1 no-stop CAGR from **-14.6% (May start, $1)** to **+15.7% (Dec start, $5,235)**; with a 35% stop it ranges from -8.9% to +20.3% (`outputs/start_month_robustness.csv`). Alternative lookbacks (January starts, 1997): 6-month no stop $35,010 / 35% stop $17,955; 12-1 no stop $9,683 / 35% stop $75,508. Same rule, different calendar choices, nearly four orders of magnitude range ($1 to $5,235) in outcomes: a sign the result is path luck, not a stable edge.

## Out of sample (Test 7) — Hypothetical backtest, not live results

Pre-declared rule: pick the stop with the best 1997–2011 Sharpe → **30%** (IS Sharpe 0.18, IS CAGR -0.4%, IS max DD -76%). Scored once on 2012–2026: $50,730, CAGR 47.8%, Sharpe 1.07, max DD -47%, beat SPY 8/14 (stop never triggered). In-sample 1997–2011 the unstopped strategy had a CAGR of -16% and a -98% drawdown. The OOS "edge" is therefore a regime/data effect (very high coverage, mega-winners NVDA/AMD/TSLA/PLTR), not evidence that a design choice generalised.

## Is it momentum or luck? — Hypothetical backtest, not live results

* **Fama-French + UMD regression (monthly, 1997–2025):** top-1 no stop: alpha 6.0%/yr (t=0.63), market beta 1.45, UMD loading 0.42 (t=1.8). Top-1 with 35% stop: alpha 12.1% (t=1.65), UMD 0.57 (t=4.1). EW top-10: alpha -0.4% (t=-0.14), UMD 0.52 (t=8.4), R² 0.71. No statistically significant alpha; the diversified version is a plain levered-momentum/beta exposure.
* **Pick quality:** the average forward-year percentile rank of the top-1 pick among eligible stocks is 52.7%, i.e. median. Prior-year winners did not beat prior-year losers (decile 10 avg +12.8% vs decile 1 +15.4%; caution: the loser deciles are survivor-inflated in this data).
* **Luck test:** random single-stock picks from the same eligible lists (10,000 paths, 1997–2025) end at $55 / $828 / $9,574 (5th/50th/95th); the strategy's $1,789 is only the 69th percentile.

## Verdict

1. **The viral numbers do not replicate** under point-in-time membership, costs, and next-open stop fills: $50,730 (not $176,114) from 2012 and $3,763 / $36,157 (not $89,418 / $1.5M) from 1997, and 8/14 years beating SPY rather than 11/15. Only the unstopped -98% drawdown reproduces.
2. **Does the edge survive?** Not as claimed. From 2012 the strategy still beats SPY in this data (CAGR 47.8% vs 14.7%, Sharpe 1.07 vs 0.91), but that is half-dependent on 3 years, comes with 46% volatility and a -47% drawdown, is hit-or-miss by start month, and sits on a data set that omits delisted names. Over the full 1997–2025 sample the unstopped version does no better than SPY on a risk-adjusted basis, has a -98% drawdown and 99.8% of its growth in 3 years.
3. **Momentum or luck?** It is a concentrated, high-beta momentum bet whose single-stock selection adds nothing measurable beyond a momentum factor tilt and a few lucky holdings. Alpha is insignificant, the picks are median-ranked going forward, and results swing from -15% to +16% CAGR depending on the start month. The 35% stop looks helpful only in 1997–2011 because of ~6 specific collapses; it is non-monotone across stop levels and irrelevant from 2012.
4. **Bottom line:** do not treat this as a tradeable edge. A proper conclusion needs Norgate or CRSP data (delisting returns, full PIT universe); with Yahoo data, pre-2012 numbers are unreliable and 2012+ numbers are upper-bound-ish.

## Files

`outputs/results_tables.md` (all tables incl. appendix of every annual pick), `log_equity_from_1997.png`, `log_equity_from_2012.png`, `yearly_bars_vs_spy.png`, `quantstats_top1_nostop.html` (main variant), `quantstats_top1_stop35.html`, `stop_sweep_full_grid.csv`, `start_month_robustness.csv`, `yearly_picks.csv`. Hypothetical backtest, not live results.
