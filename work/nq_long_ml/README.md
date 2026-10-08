# Current stage: component sensitivity and simple benchmarks (2026-10-07)

Run explore_component_sensitivity.py, package_sensitivity_report.py and finalize_sensitivity.py. Completed12000replays:100entryfamilies×4matchingcharts×15separatecomponentmethods×2costscenarios. Target forecasts are tested withnostop;MAEstop-distance sensitivity keepsoriginaltargetunchanged. Simple historicalnormalizedMFEmean/median/q70benchmarks compete with the already selected learnedforecaster. No new jointtarget-stopgrid, no automatic promotion and no reserved outcomes. Prior/frozenbaseline results reproducedon6400KPIcomparisons;foldPNLreconciled;191repositorytests passed, existingstaticfailuresunchanged.

Outputs: saved_strategies/NQ_RTH_100_LOCKED_20261007/COMPONENT_SENSITIVITY_20261007. Development-onlyfeatures,opportunities,signalsandforecasts are cached in eachduration's inputs folder. Casesfolderholds1198exploratory configurations with exact chart thresholds. They overlap and are not independent edges. The editableDatareportat http://127.0.0.1:4191/?view=1 retainspriorstages andadds componentbenchmark/sensitivity tables. Final blind evaluation is still unavailable; noactualpaper/liveexecution orportfolio hasbegun.

# Previous stage: frozen component transfer (2026-10-07)

Run transfer_frozen_components.py, finalize_transfer.py and package_transfer_report.py. The prior independently chosen component methods are transferred without candidate-specific tuning across all100 locked families on matching5/15/60/240-minute charts. Target-only, MAE-only, combined and original exits are kept separate. Development evaluation remainsJanuary–August13,2025; the reserved period stays excluded. The four former leaders replayed exactly on128 KPI comparisons. All3200 base/stress policy replays and1600 baseline-policy comparisons are complete.

Outputs: saved_strategies/NQ_RTH_100_LOCKED_20261007/COMPONENT_TRANSFER_20261007. The existing editable Data report at http://127.0.0.1:4191/?view=1 includes all100 candidates, cost stress, quarterly outcomes, best-day concentration and paired block-resampled profit-change intervals. Research flags do not establish independent edges; earlier entry/exit selection contamination and correlated rules remain. No actual paper/live testing or portfolio integration has begun.

# Previous stage: separate target and MAE-exit components (2026-10-07)

Run separate_exit_components.py, then finalize_component_artifacts.py and package_component_report.py. Entry rules remain frozen. Target is tested without a stop; MAE exit is tested against the existing mean70 target. Independently frozen choices are combined only for an interaction check. The old train_mae_exits.py joint runner is disabled.

Use all eligible development observations, expanding forward evaluation folds in2025, and final refits through2025-08-13. The saved reserved-year boundary is2025-08-21, with fivecashsession embargo. Do not evaluate the reserved period or call previously seen history untouched. No minimum entry-per-session requirement. Historical simulation only. New5minimport remains unscored without matching minute/ES coverage.

Results and separately saved entry/target/MAE files are in saved_strategies/NQ_RTH_100_LOCKED_20261007/SEPARATE_COMPONENTS_20261007. Data report: http://127.0.0.1:4191/?view=1.

The following sections are prior-stage context and do not override the current stage.

# NQ long-only probability research

## Current task: NQ RTH entry methods

Find long-entry candidates during RTH, 09:30–16:00 America/New_York, and evaluate
5, 15, 60 and 240 minute forward windows capped at session close. Study public
auction/profile, Linda Raschke and Al Brooks concepts as mechanical adaptations.
Robert J. Frey's public academic work informs research/risk analysis rather than
an invented proprietary entry. Targets, forecasting and stop models remain deferred.

Evaluate meaningful favorable moves and adverse movement before those moves, with
matched ordinary entries and session-clustered uncertainty. Report the strict
one-signal-per-session requirement for individual setups and predefined unions.
No forced entries or portfolio is implemented. Minute volume-at-price is approximate.

Run `rth_named_setups.py`, then `audit_rth_named.py`. The current report and figure are
`NQ_RTH_ENTRY_FINDINGS.md` and `NQ_RTH_ENTRY_FINDINGS.png`; corrected complete results
live in `rth_named_runs/20261006T200105Z`. Earlier screens are historical artifacts.
The report's completed dataset IDs are explicit for reproducible source replay.

## Previous task: next-hour high regression

Forecast NQ's highest price over the next complete 60 minutes, once per hour,
using only completed information at the forecast time. The accuracy goal is
absolute error no greater than 20 points. Report MAE, MSE, RMSE, displacement RÂ²,
within-20 frequency, monthly consistency, market-condition breakdowns and quantile coverage.
This task has no classification label, trading setup requirement, target, stop or trade simulator.

Run `hour_high.py` in the existing local environment, then `report_hour_high.py` after
all twelve monthly evaluation folds finish. VS Code tasks named
"NQ ML: forecast next-hour high locally" and "NQ ML: verify next-hour high results"
provide the same workflow. Independent monthly folds may run in isolated directories
with `--folds` and are merged by `combine_hour_high.py`; final reporting rejects incomplete years.

The current result is written to `HOUR_HIGH_RESULTS.md`. Forecasts, research models,
replay audits and detailed accuracy tables live under `hour_high_runs/`.
Regression uses the same development-only data restrictions below. Earlier experiments
and their historical specifications are retained in separate files.

User specification: NQ; matching 15, 60 and 240 minute charts and forecasts; long entries at calibrated probability >=0.66; 50â€“100 quantitative features; random-forest feature screening down to 20 per horizon; up to ten models; two development years and a separate final evaluation year.

This isolated experiment uses ONLY existing development partitions, 2023-08-22 through 2025-08-15. The repository says its prior sealed year was consumed on 2026-08-21. No fresh one-year holdout is currently verified. Existing sealed data must not be described as untouched or used for feature/model selection.

## Preregistered design

- CME sessions anchored at 17:00 America/Chicago. Aggregate complete contiguous minute bars only. Last partial four-hour bar is omitted. Forecast only a following complete bar within the same session.
- At a completed chart bar, predict whether buying at the next bar's open and selling at its close earns a positive return after 2 ticks per side plus $2.50 commission per side, per NQ contract ($20/point). This is an explicit research cost assumption, not a verified broker quote. Stress costs at twice this level.
- Generate 90 raw features: 66 price/volume/risk/trend features, 20 rolling single-factor features versus ES/ZN/GC/CL/6J, and four ARIMA(1,0,0) features. Add three fold-local PCA scores, for 93 candidates. ES beta/alpha are single-factor proxy features, not literal CAPM: no cash-equity market portfolio or risk-free series is available. Factor estimates use up to 64 historical bars with at least 16 matched NQ/benchmark observations. Missing benchmark bars are omitted, never forward-filled; report coverage and fail on completely unavailable estimator-training features.
- ARIMA fitted to trailing observed intrabar log returns at the first observation of each session, then fixed parameters produce one-step forecasts as new bars arrive. Maximum training window 256 bars; minimum 64. No full-sample ARIMA fitting.
- Four expanding outer folds: first validation 2024-08-22; three-month validation blocks; final block ends 2025-08-16. Earlier training data is split chronologically 60% estimator fit, 20% feature selection, 20% calibration. Labels must mature before the next segment begins.
- Fit random forest on the estimator segment; rank features by unseen selection-segment log-loss degradation after block permutations. Examine correlated feature clusters and report ranking stability. Select 20 per outer fold, then refit on fit+selection; calibrate on the calibration segment; evaluate only on the next outer block. PCA/scaling/imputation parameters are fitted using the estimator segment alone. Compare full-feature and top-20 random forest.
- Gauntlet: logistic regression, LDA, Gaussian naive Bayes, random forest, extra trees, histogram gradient boosting, AdaBoost, linear SVM, nearest neighbors, multilayer perceptron. Equal fixed configurations initially; no claims that surveys establish an intraday NQ winner. ARIMA is a component and baseline family, rather than an additional classifier in this gauntlet.
- Probability calibration: chronological sigmoid/Platt calibration. Do not apply softmax to random-forest probabilities; softmax itself does not establish calibration. Entry threshold remains fixed at 0.66. Do not promise a 66% realized win rate.
- Compare fixed one-bar exits with a signal exit: after entry, re-evaluate each completed bar; sell at the next observed open when calibrated probability falls below 0.50 or the causal trend filter turns negative. Maximum four chart bars; force close by session end. Run no-filter, positive trailing regression slope, and positive short EMA trend conditions as separate predeclared cases. No hindsight maximum-price exits. Targets/stops and a learned exit model remain a later, separately registered experiment.
- Report all-observation accuracy, balanced accuracy, Brier score/log loss, precision and trade coverage at 0.66, costed profit factor/expectancy, daily Sharpe, maximum drawdown, trade counts and fold stability. Compare against always-up predictions and always-long one-bar trades. EMA/regression filters are compared within each model's signal-exit backtest. Accuracy and profitability winners may differ. Survival means robustness to costs, folds and block-resampled daily dollar P&L; a numerical ruin probability additionally requires starting equity, leverage and a ruin definition. Circular 10-session block bootstrap intervals are exploratory and do not correct for model selection.
- Never discard a failed model or a zero-trade result. No model selection based on the consumed sealed year. Select finalists on development evidence, freeze features/parameters/exits, and then evaluate on a genuinely new one-year dataset or prospective data.

## Research basis

- [Kumbure et al., 2022 literature review](https://doi.org/10.1016/j.eswa.2022.116659): broad forecasting techniques and feature inputs; does not identify the best model for these NQ horizons.
- [Sezer et al., systematic deep-learning review](https://arxiv.org/abs/1911.13288): sequence architectures are relevant candidates but performance across other markets/horizons does not transfer automatically.
- [Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning](https://www.nber.org/papers/w25398): nonlinear interactions motivate tree ensembles; monthly equity asset-pricing evidence is not intraday futures validation.
- [Guo et al., calibration](https://proceedings.mlr.press/v70/guo17a.html): classifier confidence and calibrated event frequency differ.
- [Scikit-learn correlated-feature importance example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): correlated predictors complicate attribution; report clusters and importance stability.
- [Bailey et al., backtest overfitting](https://www.davidhbailey.com/dhbpapers/overfitting.pdf): record every trial and preserve independent evaluation.

## Execution

From the workspace root, use the existing environment:

`uv run python work/nq_long_ml/experiment.py --models rf`

`uv run python work/nq_long_ml/experiment.py --models all`

Outputs are timestamped under this experiment folder. Development results are exploratory evidence, never a live-trading recommendation. Missing benchmark bars are not forward-filled. Contract identifiers are unavailable, so cross-roll position holding is not supported. Portfolio capital, margin constraints and live order execution are not implemented.

The workspace's `uv run pytest` currently encounters its documented Windows trampoline error. If a command fails this way, invoke the corresponding module with the existing `.venv/Scripts/python.exe`; no additional packages are installed. The final-year lockout cannot be completed until genuinely untouched data is identified. A broker-realistic target/stop simulator and a learned optimal-stopping exit are outside this initial experiment.



## Latest: 1,000 RTH entry rules with causal excursion references

See `NQ_RTH_1000_ENTRY_RESULTS.md` and `clock_entry_runs_1000/20261007T001139Z`. This supersedes fixed +20-point entry scoring: targets are P70 and 70% of the mean of qualified high-minus-open history over the previous 20 sessions. Both raw and IQR histories are included. 25 exploratory candidate rules; no rule met every-session coverage. No high/stop model or sealed-year testing. Local checks: 167 tests passed; new core lint/type checks passed.


## Latest chronological backtest

`NQ_RTH_FULL_BACKTEST.md`; frozen E0044 versus ordinary higher-opening entries, one contract and one position at a time, 70%-mean IQR reference, no stop. Run `reference_backtests/20261007T003526Z`: candidate 392 resolved/0 unknown trades, net +$65,970 with $25 costs; one-minute delay +$6,575 with a losing earlier period. Comparator -$53,765 on 4,333 resolved trades, 12 unresolved. Every-session requirement remains unmet for candidate. All 392 candidate base trades independently replayed. No fresh/sealed-year validation.


## Active: October 2022 ten-million-definition entry discovery

Imported user-confirmed Eastern NQ source into `data/processed/nq_entry_discovery_202210`. Local checkpointed run: `ten_million_entry_search/STATUS.md` and `RESEARCH_PROTOCOL.md`. IS 2022–2024, validation 2025, separate 2026 confirmation; five-session embargo. Primary 70%-mean high-minus-open reference, raw >53% save, three-predicate quantitative definitions; saved stream duplicates excluded. Candidate files compressed into ZIP shards. Confirmation is previously seen historical data, not blind fresh proof. Volume-roll provenance remains unresolved. Actual progress is in progress.json, not inferred from the target.
