# Leakage audit, 7 October 2026

Read-only review of branch `audit/leakage-2026-10-07`. No research code, config, or saved result was changed. Market parquet and the NQU6 export are not in the repo, so every date range below is what the code asks to read, checked against the saved protocols and result notes. Nothing here was re-fit, and the ten-million-definition search was not re-run. Dollar and Sharpe figures quoted from saved outputs are hypothetical backtest results, not live or paper trading.

Intended protocol for this NQ program: train on calendar 2024, validate on calendar 2025, and touch calendar 2026 once, in a single final test. The code does not do that. It calls October 2022–2024 “in sample” or “discovery,” 2025 “validation” or “previously used,” and 2026 “final confirmation” or “previously seen.”

## Plain-English result

Calendar 2026 is not an untouched holdout.

The first time 2026 prices changed a keep-or-reject decision was 23 August 2026, in the older multi-market sealed test, not in the NQ entry search. That test’s window runs from 21 August 2025 through 21 August 2026. One frozen daily-session portfolio was kept for paper-trading research. One combined candidate was rejected. Both decisions use a return series that includes 2026.

Inside the NQ 100-rule program, the ten-million search and the freeze of the 100 do not put 2026 profit into the ranking formula. The shortlist file is written before that script reads 2026 labels. The same script says those 2026 dates had already been used in earlier research and in aggregate inspection, so the freeze was not blind. The first NQ decision that is actually computed from 2026 results is the refusal to promote the machine-learning exit. The first filter that drops entry candidates for weak 2026 results is the later “simulation survivor” gate.

Calendar 2025 was used to choose the 100, to choose exits, and to refit exit models. It was not held back as validation after a freeze.

The 100 entry thresholds were fit on 2022–2024 eligible rows. They were not fit on 2025 or 2026. That fit window is wider than “train = 2024 only.” It does not include the validation or confirmation years.

A later clock check is consistent with minute-end labels, while the research code assumes minute-start labels. If the export is minute-end, buying the labeled bar’s open uses a minute that has already finished. Human confirmation of the vendor convention is still open. The one rule that still passes the declared post-2024 gates under the minute-end sensitivity is a hypothetical backtest lead, and that sensitivity was chosen after seeing 2026.

Bars after 7 October 2026 are the only stretch that this snapshot has not already opened. They are not in the repo. They can be a forward holdout only if they stay unread until one final test.

## 1. Scripts that load data

“Touches 2026” means the loaded frame contains calendar-2026 timestamps, or a feature, label, threshold, fit, rank, or report is computed from those rows. A mask that drops 2026 from a score is noted. Test modules under `work/nq_long_ml/test_*.py` build synthetic 2024 clocks and do not read the research files. Modules that only transform an in-memory frame (`metrics`, `fixed_hold`, `clock_excursion`, `execution_stress`, `auction_entry`, `reference_backtest`, `research_partitions`, `data.py`) are omitted from the tables.

### 1.1 `work/nq_long_ml`

Development-only NQ experiments share one loader. `experiment.load_minutes` (`experiment.py` lines 35–60) reads `data/processed/databento_research/development_minute_returns/symbol=*/returns.parquet` with `timestamp_utc >= 2023-08-22` and `< 2025-08-16`. That partition ends 15 August 2025. These scripts do not read calendar-2026 bars:

| Script | What it loads | Range it uses |
|---|---|---|
| `experiment.py` `load_minutes`, `run` | NQ plus ES, ZN, GC, CL, 6J development minutes | 2023-08-22 to 2025-08-16 exclusive. Outer folds start 2024-08-22, 2024-11-22, 2025-02-22, 2025-05-22 (`experiment.py` lines 349–359). |
| `medium_frequency.py` `prepare` | same, via `load_minutes` | same end. Hourly clock stops at `END`. |
| `direction_expanded.py` `load_extended` | development minutes, plus `databento_portfolio_extension` minute bars filtered to the same `[START, END)` | same. Added contracts can start September 2024. |
| `fifteen_probability.py`, `hour_high.py`, `range_forecast.py` | `load_minutes` / `load_extended` | same. Hour-high report text says the sealed year was not opened. |
| `entry_screen.py`, `entry_quality.py`, `rth_named_setups.py`, `clock_entries.py`, `clock_entry_gauntlet.py`, `adaptive_entries.py`, `session_high.py`, `audit_rth_named.py`, `full_reference_backtest.py` | `load_minutes("NQ")` and sometimes ES, plus earlier outcome parquets from those runs | decisions through 2025-08-16 exclusive. |
| `runs/`, `direction_runs/`, `range_runs/`, `medium_runs/` `source_snapshot.py` | copies of the loaders above | same. |

The October 2022 NQ export is the series that contains 2026. `prepare_oct2022_discovery.py` lines 13–23 read `../Market Data/NQU6_Export.csv` with no date filter and build an XNYS RTH schedule from 12 October 2022 through 20 August 2026. Line 47 labels year ≤ 2024 in sample, 2025 validation, and everything else `final_confirmation`. Lines 49–53 embargo the first five cash sessions of 2025 and of 2026. Lines 54–73 write hit labels and hypothetical dollar paths for every eligible row, including 2026, into `opportunities.parquet`.

| Script | Load | 2026 role |
|---|---|---|
| `prepare_oct2022_discovery.py` | full export, schedule through 2026-08-20 | Writes 2026 labels before any rule is scored. Does not rank rules. |
| `build_oct2022_atoms.py` lines 10, 46, 69–72 | opportunities, minutes, 5-minute bars, RTH schedule, and ES minutes from both `development_minute_returns` and `sealed_holdout_minute_returns` | 2026 ES bars are in the feature frame. Quantile thresholds use in-sample rows only (lines 79–98). |
| `search_ten_million.py` `run` lines 28–34 | full `opportunities.parquet` and `atom_bits.npy` | 2026 rows are in memory. Stage masks and `execution_metrics` skip them. Save rule is the in-sample hit rate. |
| `audit_ten_million.py` | same full files | Assertions cover in-sample and validation only. One-hour validation paths do not reach 2026. |
| `confirm_ten_million.py` lines 30–34 | after the shortlist is written, `split == final_confirmation` plus the full minute file | 2026 replay. Does not change the 100. |
| `build_shortlist_metrics.py` | full opportunities, atom bits, and the 2026 trade logs | Sorts a report of the same 100 by 2026 Sharpe (line 58). Membership unchanged. |
| `extended_hold_pilot.py` lines 38–66 | completed 5-minute bars, minutes, opportunities, RTH schedule, and sealed-plus-development ES | Fits on labels that mature before 2025-01-01. Scores 2025 and 2026 forecast error (lines 86–90). Ranks exit variants on 2025 before the 2026 trade replay (lines 129–135). |
| `full_locked_validation.py` lines 29–50 and `full_locked_validation_minute_end.py` | discovery minutes, opportunities, features, atom bits, schedule, and `databento_canonical/symbol=NQ/bars.parquet` | Scores 2022–2024, 2025, and 2026. Survivor rule uses 2025 and 2026 together (lines 157–166). |
| `finish_locked_validation.py`, `finish_clock_end_metrics.py` | same discovery or minute-end files, plus saved verdicts | Report generation on those replays, including 2026. |
| `prepare_clock_end_scenario.py` | re-executes the discovery and atom builders on a clock-shifted copy | Rebuilds the series through 2026-08-20. Keeps the original numeric thresholds. |
| `run_four_timeframes.py` lines 20–26, 60 | minute-end scenario minutes, schedule, original discovery minutes, opportunities, and features | Refits chart thresholds on 2022–2024. Scores 2026. Paper screen requires 2026 profit and profit factor (lines 106–110). |
| `train_mae_exits.py` | that four-timeframe preparation, then 2026 scoring after a 2025 choice | Training labels are 2022–2024. Selection is 2025 profit (lines 77–86). 2026 is scored after the selection file is written. |
| `audit_feature_prefix.py` | discovery `features.parquet` | Spot-checks 2025-04-15 and 2026-06-15. |
| `audit_new_nq_import.py` lines 9–20 | a new 5-minute export, 23 April 2026 through 7 October 2026, and old minutes from 20 April 2026 to 21 August 2026 | Alignment study. Does not score the post-20 August dates as a holdout. |
| `separate_exit_components.py` | four-timeframe loader, then line 133 clips minutes and the schedule to before 14 August 2025 09:30 New York | `require_before_lockout` refuses a timestamp on or after 21 August 2025. 2026 is not scored. Final refit uses rows through 13 August 2025 (lines 115–125). |
| `transfer_frozen_components.py` | same clip at line 134 | Replays frozen methods on the clipped 2025 development window. |
| `explore_component_sensitivity.py` | reuses the clipped fitter | January–13 August 2025 only. Saved note: no reserved outcomes. |

Report and packaging scripts read saved csv, parquet, and json under the run folders or `saved_strategies/`. They do not open the vendor export. These still publish 2026 hypothetical results when their inputs are 2026 replays: `save_extended_hold_findings.py`, `package_full_validation_report.py`, `export_shortlist_quantstats.py`, `export_clock_survivor_quantstats.py`, `finish_four_timeframes.py`, `finalize_component_artifacts.py`, and the `package_*` / `finalize_*` / `report_*` helpers. `report_hour_high.py`, `report_direction.py`, `report_range.py`, `report_fifteen.py`, `report_entry_quality.py`, `report_rth_named.py`, `report_clock_entry.py`, `validate_results.py`, `replay_medium.py`, and `combine_hour_high.py` read the development-only run outputs (through 15 August 2025).

### 1.2 `src/systematic_research`

The Databento research cut is an anchored 12-month holdout with a 5-day embargo (`config/validation.yaml`, `temporal_validation.create_temporal_split` lines 45–48). The saved sealed result window is 21 August 2025 through 21 August 2026, 260 sessions. Development manifests end 15 August 2025. So “development only” means no calendar-2026 bars. The sealed file is mostly late 2025 plus 2026, not calendar 2026 alone.

| Script | Function | Source and range | 2026 |
|---|---|---|---|
| `research_panel.py` | `build_research_panel` | every `symbol=*/bars.parquet` under the canonical root, then `write_temporal_split` / `seal_minute_partitions` | Reads the full vendor span, including 2026, in order to cut the sealed file. No strategy choice. |
| `normalization.py` | csv readers | raw vendor files. Catalog names include `*_2023-08-22_to_2026-08-22.csv` | Reads 2026 into the canonical layer. |
| `temporal_validation.py` | `seal_minute_partitions`, `load_development_panel`, `load_sealed_holdout` | combined minute returns, then the two partitions | `load_sealed_holdout` raises unless `sequential_evaluator=True` (lines 143–149). Direct parquet reads elsewhere bypass that. |
| `daily_session_factory.py` | `load_development_panel` | `development_session_panel.parquet` | No. Selection uses this panel only. |
| `daily_session_sealed.py` | `evaluate_sealed_once` | development panel plus `sealed_holdout_session_panel.parquet` (line 164) | Yes. Result window 2025-08-21 to 2026-08-21. |
| `sealed_evaluator.py` | `run_once`, `_load_minute_partitions` | development and sealed session panels and minute returns (lines 154–157) | Yes. Same window. |
| `databento_portfolio_extension.py` | `read_dbn_outrights`, `main`, `evaluate_portfolio_extension` | DBN batch `GLBX-20260916` (raw period recorded as 2024-09-16 through 2026-09-15) and both the development and sealed panels (lines 574–577) | Yes. Weights use the first 60 percent of the overlap, recorded as ending 2025-11-12, which is inside the old sealed year. Correlations use the full overlap through 2026-08-21 (line 317). Evaluation is recorded as 2025-11-13 through 2026-08-21. |
| `vectorbt_strategy_factory.py` | `load_development_minutes` | development minute partitions only | No. Callers: `unique_hypothesis_factory`, `paper_strategy_factory`, `ml_hypothesis_factory`, `graph_laplacian_rv`, `regularized_var_network`, `managed_money_diffusion`, `pca_residual_reversal`, `eia_release_drift`, `cot_hedging_pressure`. |
| `pattern_discovery.py` | `load_returns` | development minutes | No. Ledger flag `holdout_accessed` is false. |
| `intraday_strategy_search.py`, `strategy_factory.py`, `regression_displacement.py`, `intraday_mean_reversion.py` | `run` | `development_minute_returns` | No. |
| `price_action_probability_engine.py`, `event_probability_engine.py`, `es_session_reversion.py`, `dynamic_state_scan.py` | `main` | development minutes for the requested roots, including NQ where listed | No. |
| `features.py`, `labels.py` | panel builders | `load_development_panel` and development minutes | No. |
| `institutional_eda.py` | `load_development_bars` | development minutes; refuses a path that contains `sealed_holdout` | No. |
| `saved_strategy_runner.py` | `run_saved_strategy` | feature and panel paths from the candidate yaml | Development reruns. Writes `holdout_accessed: false`. |
| `broad_strategy_search.py`, `candidate_portfolio.py`, `mean_reversion.py`, `ml_trend.py`, `trend.py`, `rates_reversion.py`, `regimes.py`, `preprocessing.py` | `run` / writers | development features, panels, and `outputs/development_walk_forward_folds.csv` | No. |
| `cot_hedging_pressure.py`, `managed_money_diffusion.py` | `run` | development minutes plus CFTC csv archives described as 2023–2025 | Minute bars stop before the sealed cut. The CFTC files are not the NQ minute holdout. |
| `archive_daily_session_candidate.py`, `archive_unique_hypothesis_run.py` | archive writers | already saved return parquets | The daily-session archive copies the sealed result, which includes 2026. It does not refit. |
| `market_structure_review.py` | `main` | development session panel, plus whatever is under `data/raw/free_api` for FX and crypto | Futures panel is development-only. |
| `residual_signal_research.py` | `main` | `data/raw/free_api` parquet paths | Range is whatever those downloads contain. |
| `free_market_data.py` | `fetch_binance_klines` | Binance public klines, start and end from the caller | Can include 2026 if the caller asks. |
| `oanda_historical.py` | importer | OANDA practice candles, caller range | Same. |
| `crypto_volume_profile_bootstrap.py` | file scan | local M1 csv/zip | Eligibility requires the file to reach `LOCKOUT_END` 2026-08-31 (lines 22–23 and 103–105). Selection notional uses only 1–31 August 2023. Lockout returns are not a fit sample in this module. |
| `data_catalog.py`, `instruments.py` | scanners | headers and normalization manifests | No outcome modeling. |

`entry_discovery_search.py` does not load files. `screen_rules` (lines 36–43) documents that it counts only two stages and that no confirmation mask is supplied. `execution_metrics` (lines 112–114) skips `stage < 0`, which is how embargo and 2026 rows are dropped when the caller sets them to −1.

## 2. First point where 2026 influenced a choice

Two different programs touch 2026. The earlier one is the multi-market sealed year. The NQ entry search is later, and its own ranking formula does not use 2026 profit. Both matter.

### 2.1 Earliest keep-or-reject in the repo

`daily_session_sealed.evaluate_sealed_once` (`src/systematic_research/daily_session_sealed.py` line 164) is the first function that reads the sealed session panel and turns it into a performance snapshot. The frozen specs were chosen on the development panel, which ends 15 August 2025 (`freeze_candidate`, lines 49–63 and 112–116). The function does not refit. It concatenates development and sealed panels, replays the frozen rules, and slices returns from the first sealed session through the last (lines 182–186).

The saved snapshot is `saved_strategies/stable_daily_session_2026-08-23/sealed_result.json`: 260 sessions, 21 August 2025 through 21 August 2026. The hypothetical backtest Sharpe in that file is about 2.42. `post_sealed_audit.json` records the decision to treat the candidate as a pass and as suitable for paper-trading research. That decision uses a P&L path that includes 2026.

`sealed_evaluator.run_once` (`sealed_evaluator.py` lines 154–157) reads the same sealed panel and the sealed minute partitions for `combined_session_intraday_2026-08-23`. The saved hypothetical backtest Sharpe on that same window is about −0.52. That result is why the combined candidate was kept as a rejection rather than as a research lead. `assert_holdout_unused` (lines 28–36) blocks a second sealed run. It does not stop other modules from opening the parquet by path.

No earlier strategy module scores 2026. Development runners stop at 15 August 2025. `databento_portfolio_extension.main` also opens the sealed panel, and its correlations include 2026, but the raw batch id is `GLBX-20260916`, a 16 September 2026 drop, after this sealed test.

### 2.2 NQ entry program

`prepare_oct2022_discovery.py` line 13 is the first NQ script that loads 2026 NQ bars. It writes 2026 hit labels into `opportunities.parquet` (lines 54–73) before the search. That is exposure, not yet a ranking.

`build_oct2022_atoms.py` lines 69–72 then open the sealed ES holdout, so 2026 ES bars sit in the feature matrix before any rule is tested. Thresholds themselves are quantiles of in-sample rows (lines 79–98). The feature code uses trailing rolls, shifts, and a one-session profile lag. This audit did not find a full-sample scaler. Opening the sealed ES file still breaks the “sealed file stays closed” rule, and a one-minute clock mismatch between that ES file and the NQ export would shift ES features by a bar.

`search_ten_million.run` (`search_ten_million.py` lines 32–34 and 72) saves a definition when the in-sample mean-70 hit rate is strictly above 53 percent. Validation counts are stored and used only to choose the zip folder `passed_validation` versus `validation_failed` (line 93). The protocol string at line 50 says final confirmation is not scored by the search and that validation does not change thresholds or definitions. `entry_discovery_search.screen_rules` has no third-stage mask. `execution_metrics` ignores negative stages. A shared `busy` flag cannot carry a 2026 position back into 2024 or 2025, because 2026 is later and those rows are skipped before `busy` is updated.

`confirm_ten_million.py` lines 16–30 build the 100 from `passed_validation` only. The heap key is the worse of in-sample and 2025 accuracy, then the worse of the two profit factors, then the id. Line 30 writes `frozen_shortlist.json` before line 32 reads `final_confirmation`. The 100’s membership is not a function of 2026 profit inside this script.

Line 54 of that same file states that the 2026 dates “have been used in prior research and aggregate inspection” and that the replay is not an untouched holdout. `monitor_ten_million.py` line 22 says the same. `full_locked_validation.py` line 24 repeats that the selected universe used 2025 and previously seen 2026. The prior research use with a recorded keep-or-reject is the 23 August sealed test above. This audit did not find a second script that feeds a 2026 NQ statistic into the 53 percent cutoff or the atom list. The cutoff is a constant. The contamination at the freeze is the researcher’s already-opened 2026, which the code itself documents, plus the fact that 2026 labels already existed on disk.

The first NQ choice computed from 2026 numbers is the exit decision in `save_extended_hold_findings.py` line 13: keep the original exits, because the pilot did not improve win rates and 2026 forecast RMSE did not beat a zero-change forecast. Line 22 sets `exit_upgrade_accepted` to false. The variant that was replayed had been chosen on 2025 only, in `extended_hold_pilot.py` lines 129–135, and the 2026 error was computed at lines 86–90. Promoting or not promoting that exit is a choice 2026 changed.

The first filter that drops NQ entry candidates because of 2026 trading results is `full_locked_validation.py` lines 157–166. `r` is the union of validation and `final_confirmation`. A candidate fails `simulation_survivor` if either year has fewer than 50 trades, a non-positive net profit, or a profit factor under 1.10, or if a post-2024 quarter fails. That gate is what produced the provisional survivor list. The minute-end rerun uses the same gate.

`run_four_timeframes.py` lines 106–110 are a later paper screen: a chart variant must also show positive 2026 profit and profit factor above 1. `build_shortlist_metrics.py` line 58 sorts the frozen 100 by 2026 Sharpe for a table. That sort does not add or remove names.

`audit_new_nq_import.py` lines 20–29 then use the April–August 2026 overlap (6,849 bars) to pick a −1 minute shift. The note on line 29 says 13 candidates were frozen before that new file was read. The clock choice itself was fit to 2026 overlap. The minute-end survivor count is a result after that choice. The new file’s 32 dates after 20 August 2026 were counted and were not scored, because the attachment has no matching minute path or ES. They are still seen data. The file runs through 7 October 2026.

## 3. How 2025 was used

2025 was a fitting and selection sample, and also a reported “validation” column. It was not locked after the entry rules were frozen.

| Step | File and line | What 2025 did |
|---|---|---|
| Search folders | `search_ten_million.py` line 93 | Hit rate above 53 percent and at least 50 validation events puts a rule in `passed_validation`. Only that folder can enter the top 100. |
| Top 100 | `confirm_ten_million.py` lines 18–28 | 2025 accuracy and profit factor are half of the sort key. Positive 2025 dollars, at least 100 events, and profit factor at least 1.1 are required. |
| Early NQ models | `experiment.py` lines 349–359 | Folds that test late 2024 or early 2025 become training rows for the next fold. The last test block ends 2025-08-16. There is no untouched 2025. |
| Exit variant | `extended_hold_pilot.py` lines 64–83 and 129–135 | Ridge, trees, and boosted trees fit on labels maturing before 1 January 2025. The exit variant is the one with the best median 2025 expectancy. |
| Learned target and stop | `train_mae_exits.py` lines 77–86 | Winner is maximum 2025 net profit, with a trade-count floor. 2026 is scored after that write. |
| Separate target and MAE models | `separate_exit_components.py` lines 100–125 | Each component is chosen on stitched 2025 out-of-fold profit. The winner is then refit on every permitted row through 13 August 2025, so the selection year is also in the final fit. |
| Transfer and sensitivity | `transfer_frozen_components.py`, `explore_component_sensitivity.py` | No new search of 2026. Comparisons that decide which benchmark “looks competitive” are on January–13 August 2025, a year already used to pick entries and exits. |
| Four-timeframe paper screen | `run_four_timeframes.py` lines 105–110 | “Post-2024” pools 2025 with 2026. |
| Survivor gate | `full_locked_validation.py` lines 157–166 | 2025 and 2026 must both pass. |

`separate_exit_components.py` does keep a hard wall at 21 August 2025 for that one experiment. That wall is the old sealed boundary. It is not “2025 was validation only.”

## 4. The 100 locked thresholds

The original atom thresholds were fit on training rows only, with this program’s training definition: in sample, which is calendar year ≤ 2024, and eligible, and not in the five-session embargo.

`build_oct2022_atoms.py` line 79 sets `train` to `split == in_sample` and eligible. Lines 89–98 take the unique 10th through 90th percentiles of `value.loc[train]` and store predicates of the form `feature > q` and `feature < q`. Boolean features become true/false atoms without a numeric quantile. The manifest written at line 108 sets `training_only_thresholds` to true. The in-sample label starts in October 2022, so 2022 and 2023 are inside the fit. 2025 and 2026 are not.

`same_clock_history` (`clock_excursion.py` lines 67–97) builds the target from the previous 20 sessions at that clock, excluding the current session. That target is a label, not the atom threshold. The IQR trim is applied inside that past window only.

`run_four_timeframes.py` lines 73–77 refit the same quantile rank on each chart’s own 2022–2024 rows. Those are new timeframe thresholds. The protocol text in that file says they are not the original 100 numeric rules. `prepare_clock_end_scenario.py` lines 18–26 keeps `exact_frozen_atoms.csv` and does not refit on the shifted clock.

Two caveats, neither of which is “the percentile was computed on 2025”:

- The fit window is October 2022–2024, not calendar 2024 alone.
- Features are built on the full series and then sliced. Trailing windows do not pull 2025 into a 2024 value. The sealed ES series is concatenated before that slice (`build_oct2022_atoms.py` lines 69–77). A clock offset between ES and NQ could move ES features by one bar on every year, including the fit year.

`extended_hold_pilot.py` and the later component scripts do not change these entry thresholds. Exit models are a separate fit.

## 5. Bar timestamp convention

The discovery code assumes minute-start labels.

- `prepare_oct2022_discovery.py` line 78 records `bar_semantics` as assumed start-of-minute timestamps, and says to verify the exporter.
- `experiment.aggregate` (lines 63–88) says timestamps denote the bar open. `available_at` is the open plus the bar length. A completed bar is one whose first minute equals the open stamp and whose last minute equals the open plus horizon minus one minute.
- `fixed_hold_outcomes` (`fixed_hold.py` lines 34–59) buys the open at the entry stamp and treats a 5-minute bucket as complete when the first stamp equals the bucket start and the last stamp equals the bucket start plus 4 minutes.

If the vendor stamp is the minute the bar opens, that pattern is the right one: features wait until `available_at`, and the fill is the open of a bar that has not yet printed its high, low, and close. Using the high or close of that same stamp as if it were already known would be look-ahead. The atom builder’s own note says the current minute contributes its open to eligibility, and completed bars supply the features.

If the vendor stamp is the minute the bar closes, the same code is early by one minute. The open at the stamp is the open of a minute that has already finished, so the high, low, and close of that minute are knowable before the simulated buy. Five-minute buckets grouped with `floor("5min")` would also be shifted relative to the exchange’s 5-minute grid. Features marked available at stamp plus horizon would be late rather than early, which is a lag, not look-ahead. The dangerous side is the fill and any signal that reads the labeled bar’s range at the stamp.

This audit does not decide the vendor convention. The repo’s later alignment is evidence, not a vendor certificate. `audit_new_nq_import.py` lines 20–29 compare the old minute file, on the April–August 2026 overlap, with a new 5-minute export under a New York five-minute-end reading. Shifting the old stamps by −1 minute matched 6,849 of 6,849 volumes and relative OHLC shapes. Other shifts did not. Prices differed by a constant +297.25 points, which the note treats as a continuous-contract adjustment. The written interpretation says this is strong evidence of minute-end labels and that a person still needs to confirm it.

`prepare_clock_end_scenario.py` lines 9–12 apply that −1 minute shift and rerun the book. `FULL_VALIDATION_20261007/FINAL_RESULTS.md` says the minute-start book had 13 provisional post-2024 survivors and the minute-end book has one, `NQ_000748828115`. Those survivor counts are hypothetical backtests. The minute-end table in that note shows, for that one rule, discovery 2022–2024 about $5,045, previously used 2025 about $24,390, and previously seen 2026 about $26,630, before costs beyond the stated assumptions. The older hypothetical 2026 figure of about $101,435 belongs to the minute-start clock and is not the minute-end estimate. The clock used for the sensitivity was chosen by matching 2026 bars, so that one-rule result is not a pre-committed holdout.

## 6. What is still clean, and what is contaminated

Still clean, in the narrow sense that 2025 and 2026 labels did not enter the number:

- The original 100 numeric thresholds, and the later per-chart refits of those same quantile ranks, are computed on eligible 2022–2024 rows.
- The ten-million scorer does not add 2026 hits or 2026 dollars into the save/fail decision or the top-100 heap.
- Exit-model coefficients in the extended-hold pilot are fit on labels that mature before 1 January 2025.
- `separate_exit_components.py`, `transfer_frozen_components.py`, and `explore_component_sensitivity.py` clip inputs before 14 August 2025 and raise if a lockout timestamp remains. They do not score the reserved year.
- The October 3 development experiments never open the sealed partition. Their last bar is before 16 August 2025.
- The August daily-session rules were chosen on data ending 15 August 2025. The sealed runner does not refit them.

Contaminated, in the sense that a later reader cannot treat the published choice as if 2026 or 2025 had been unseen:

- Calendar 2026, from the start of the sealed window through 7 October 2026, has been read. The sealed test kept one portfolio and rejected another. The NQ program then replayed 2026, ranked reports by it, refused an exit upgrade because of it, gated survivors on it, screened timeframe variants on it, and chose a clock shift by matching it.
- The 100 names were chosen with 2025 performance in the sort key, after a search that had already seen 2025 hit rates. The authors also say 2026 had already been inspected. Membership does not depend on a 2026 column in `confirm_ten_million.py`. The information set was not blind.
- 2025 was used again to pick the exit variant, the MAE and target models, and the final refit of those models through 13 August 2025. Component “benchmarks” on that same window are comparisons on a used sample.
- The early expanding folds train on part of 2025 when they test a later 2025 block.
- `databento_portfolio_extension.evaluate_portfolio_extension` fits weights through a cut the report places at 12 November 2025, inside the old sealed year, and computes correlations on the full overlap, including 2026.
- There is no unused calendar-2026 slice left in this snapshot. The April–October 2026 5-minute attachment was opened for alignment and coverage. Post-20 August rows were not turned into strategy scores. They are still seen.

The 100 entry definitions can stay frozen as historical objects. Their thresholds do not need to be thrown away to stop further leakage. Any claim that their 2025 or 2026 hypothetical backtests are a first look is not supported by this code.

## Proposal, not applied

No code, config, or result file was changed in this run. The following is the design to implement after approval. It does not delete data or saved strategies, does not rerun the ten-million search, and does not edit the 100 entry definitions.

### One split file

Add a single YAML file, for example `config/research_splits.yaml`, and make every loader take dates from it. Suggested contents for the next clean test, given this audit:

```yaml
timezone: America/New_York
embargo_cash_sessions: 5
train:
  start: 2024-01-01
  end: 2024-12-31
validation:
  start: 2025-01-01
  end: 2025-12-31
holdout:
  # First session after the 7 October 2026 snapshot. Open ended until one final test.
  start: 2026-10-08
  end: null
contaminated_history:
  # Documented, not a fittable sample. Includes the sealed window and this NQ program.
  start: 2025-08-21
  end: 2026-10-07
```

Train and validation here match the intended protocol. They are not a claim that historical 2025 is still fresh. The holdout start is the day after this snapshot because the new export was read through 7 October 2026. A null end means “whatever has arrived since,” collected forward, not a file that already exists in the lab.

Loaders that today hardcode 2022–2024, 2025, and 2026, or hardcode 2023-08-22 and 2025-08-16, would read this file. Direct paths to `sealed_holdout_*` would go through the guard below. `build_oct2022_atoms.py` opening the sealed ES parquet by glob is the pattern to close.

### A guard on fitting, selection, and tuning

Extend the idea already in `research_partitions.require_before_lockout` and `temporal_validation.load_sealed_holdout`. One function, called at the start of any fit, quantile, feature-screen, model selection, or shortlist rank:

- Reject a timezone-naive index.
- Reject the frame if any timestamp falls on a holdout session, or if a label’s end falls on one.
- Reject a read of a sealed partition unless the caller is the one-time unlock path.
- Allow a report or replay to pass only when the spec hash matches an unlock record.

The current helpers are local. `require_before_lockout` is used by the component scripts and not by the search, the atom builder, or the sealed-panel readers. `load_sealed_holdout` is bypassed by every direct `read_parquet` of the sealed file.

### One-time unlock

`unlock_holdout(spec_path) -> record` writes one log line and then allows a single evaluation:

- UTC timestamp
- SHA-256 of the frozen spec (entries, thresholds, exit rules, cost, clock convention)
- SHA-256 of the holdout file actually opened
- the split-file hash

A second call, or a second evaluation with the same spec, raises. This matches `daily_session_sealed.assert_sealed_evaluation_unused` and `sealed_evaluator.assert_holdout_unused`, which already refuse a second sealed attempt, and adds the spec hash so a quiet edit cannot reuse the unlock. The log is append-only.

### Tests

Use a tiny in-memory bar fixture with three labeled sessions, one in each split. No vendor file.

- A quantile and a model fit on train rows succeed.
- The same fit raises when one holdout timestamp is appended.
- A label whose start is in validation and whose end falls in the holdout is rejected for training.
- Ranking or “pick the max profit” raises if the candidate table contains a holdout row.
- After `unlock_holdout`, one replay of the holdout fixture succeeds and the log has a timestamp and both hashes.
- A second unlock, and a fit after unlock, raise.
- A direct read helper for the sealed path raises when the caller is a search or threshold function.

### Forward holdout

Treat every bar with a New York session date on or before 7 October 2026 as contaminated history. Collect new NQ, and the ES context the rules need, only after that date. Run the frozen 100, with thresholds and exits already written, once, through the unlock. Do not refit quantiles, do not drop names, and do not retune exits on that sample. If the vendor clock is still unconfirmed, record the convention in the unlock spec before the test, and do not choose it by matching the new bars to the old ones.

Until that forward sample exists, published 2025 and 2026 hypothetical backtests stay descriptive replays of data the research has already used.
