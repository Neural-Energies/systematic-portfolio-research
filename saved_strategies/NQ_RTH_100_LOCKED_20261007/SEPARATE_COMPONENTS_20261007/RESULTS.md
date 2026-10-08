# Separate NQ target and MAE-exit components

Entry rules are retained. Targets were tested with no stop; MAE exits were tested with the existing target. Each choice was frozen independently before their interaction check. Model and configuration files are stored separately in entry_rules, target_models, mae_exit_models and combined_configs.

The saved reserved-year boundary is August21,2025. A five-cash-session gap leaves outcomes throughAugust13,2025. Expanding2025development folds fit all earlier eligible/scorable rows. Final refits use all permitted development rows, with no extra second embargo. The reserved-year outcomes were not evaluated by this experiment. Earlier entry selection used later history, so this is not unbiased independent strategy validation.

Hourly NQ_001009785391: baseline$7,710/PF1.65/DD$7,632.50 across30trades; ExtraTrees target×0.7 plus separately selected ExtraTreesMAE stop$17,850/PF2.35/DD$5,210. Profitable trades decline80%to66.7%. Combined higher-cost/slippage/penetration replay$16,935/PF2.25/DD$5,240.

15-minute NQ_000502850189: historical75th-percentile normalizedMAE stop with existingtarget$3,970 versus$3,220 baseline, DD$3,927.50 versus$5,647.50 across36trades. Joint use of its independently learnedtarget produces only$2,035 andDD$5,730; do not silently promote that combination.

Four-hour NQ_000054683951: learnedmedian target$24,750 versus$16,115 baseline, butDD$10,885 versus$9,895 andonly15trades. No testedMAE stop beat no-stopbaseline under profitobjective.

Five-minute NQ_001049588361 remains unprofitable: combined exits−$2,930 versus−$9,180 baseline. Do not call reducedloss a profitableedge.

Prediction errors and coverage are held-forward development diagnostics, distinct from trading profitability. Final model choices use these development folds and remain subject to selection bias. No full-search multiple-testing correction or live execution claim is made. Source volume-roll provenance remains unverified; newly imported5min dates remain unscored without matchedES/minute/fullsessioninputs.

191 repository tests passed. Changed calculations/tests passed scoped lint, format and type checks. Repository-wide baseline failures remain40lint errors,89type errors in11files and6format files. Browser preview verified allfourtimeframes, eachcomponent selector, nativeediting/cancel, provenance throughDataWebMCP, paintedcharts and narrow/desktoplayouts.

[Verified local Data report](http://127.0.0.1:4191/?view=1)
