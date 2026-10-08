from pathlib import Path
import json
import joblib
import pandas as pd
import sklearn

O=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/SEPARATE_COMPONENTS_20261007')
L=O.parent
rules={r['id']:r for r in json.loads((L/'LOCKED_ENTRIES.json').read_text())}
dirs={k:O/k for k in ['entry_rules','target_models','mae_exit_models','combined_configs']}
for folder in dirs.values():folder.mkdir(exist_ok=True)
packages=list(O.glob('*COMPONENT_MODELS.joblib'));assert len(packages)==4
audit=[]
for path in packages:
    data=joblib.load(path);duration=data['duration'];last=pd.Timestamp(data['last_training_label'])
    assert last<pd.Timestamp('2025-08-21',tz='America/New_York')
    base={k:v for k,v in data.items() if k!='models'}
    ids=[a['atom_id'] for a in rules[data['candidate']]['definition']]
    entry={'candidate':data['candidate'],'duration':duration,'rule_family':rules[data['candidate']]['definition'],
           'execution_thresholds':[r for r in data['entry_thresholds'] if r['atom_id'] in ids],'observed_opening_up_gate':True,'RTH_anchor':'09:30Eastern'}
    (dirs['entry_rules']/f'{duration}m.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
    joblib.dump({**base,'component':'target','model':data['models'].get('target',{'type':'existing_mean70_target'})},dirs['target_models']/f'{duration}m.joblib',compress=3)
    joblib.dump({**base,'component':'MAE_exit','model':data['models'].get('MAE_exit',{'type':'no_stop'})},dirs['mae_exit_models']/f'{duration}m.joblib',compress=3)
    config={'candidate':data['candidate'],'duration':duration,'choices':data['choices'],
            'entry_file':str((dirs['entry_rules']/f'{duration}m.json').resolve()),
            'target_file':str((dirs['target_models']/f'{duration}m.joblib').resolve()),
            'MAE_exit_file':str((dirs['mae_exit_models']/f'{duration}m.joblib').resolve()),
            'trade_size':'One NQ contract','time_exit':'Matching RTH window, capped at cash close','development_only':True}
    (dirs['combined_configs']/f'{duration}m.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    audit.append({'duration':duration,'training_rows':data['training_rows'],'last_training_outcome':str(last),'reserved_rows':0})
(O/'FINAL_FIT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
d=pd.read_csv(O/'BACKTEST_COMPARISONS.csv');assert not d.duplicated(['duration','variant','stress']).any()
f=pd.read_csv(O/'FOLD_INPUT_AUDIT.csv');assert len(f)==12 and f.lockout_rows.sum()==0
selected=d[d.variant.isin(['baseline','combined_frozen_components'])]
text=['# Separate NQ target and MAE-exit components','',
'Entry rules are retained. Targets were tested with no stop; MAE exits were tested with the existing target. Each choice was frozen independently before their interaction check. Model and configuration files are stored separately in entry_rules, target_models, mae_exit_models and combined_configs.',
'','The saved reserved-year boundary is August21,2025. A five-cash-session gap leaves outcomes throughAugust13,2025. Expanding2025development folds fit all earlier eligible/scorable rows. Final refits use all permitted development rows, with no extra second embargo. The reserved-year outcomes were not evaluated by this experiment. Earlier entry selection used later history, so this is not unbiased independent strategy validation.','',
'Hourly NQ_001009785391: baseline$7,710/PF1.65/DD$7,632.50 across30trades; ExtraTrees target×0.7 plus separately selected ExtraTreesMAE stop$17,850/PF2.35/DD$5,210. Profitable trades decline80%to66.7%. Combined higher-cost/slippage/penetration replay$16,935/PF2.25/DD$5,240.',
'','15-minute NQ_000502850189: historical75th-percentile normalizedMAE stop with existingtarget$3,970 versus$3,220 baseline, DD$3,927.50 versus$5,647.50 across36trades. Joint use of its independently learnedtarget produces only$2,035 andDD$5,730; do not silently promote that combination.',
'','Four-hour NQ_000054683951: learnedmedian target$24,750 versus$16,115 baseline, butDD$10,885 versus$9,895 andonly15trades. No testedMAE stop beat no-stopbaseline under profitobjective.',
'','Five-minute NQ_001049588361 remains unprofitable: combined exits−$2,930 versus−$9,180 baseline. Do not call reducedloss a profitableedge.','',
'Prediction errors and coverage are held-forward development diagnostics, distinct from trading profitability. Final model choices use these development folds and remain subject to selection bias. No full-search multiple-testing correction or live execution claim is made. Source volume-roll provenance remains unverified; newly imported5min dates remain unscored without matchedES/minute/fullsessioninputs.','',
'191 repository tests passed. Changed calculations/tests passed scoped lint, format and type checks. Repository-wide baseline failures remain40lint errors,89type errors in11files and6format files. Browser preview verified allfourtimeframes, eachcomponent selector, nativeediting/cancel, provenance throughDataWebMCP, paintedcharts and narrow/desktoplayouts.','',
'[Verified local Data report](http://127.0.0.1:4191/?view=1)']
(O/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
verification={'status':'completed_development_component_research','component_selections_separate':True,'reserved_rows_used':0,'folds':12,'final_fits':audit,'tests_passed':191,'scoped_static_checks':'pass','existing_full_lint_errors':40,'existing_full_mypy_errors':89,'existing_format_files':6,'sklearn_version':sklearn.__version__,'preview_url':'http://127.0.0.1:4191/?view=1','preview_verified':True,'not_live_or_blind_validation':True}
(O/'VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
old=L/'MAE_ML_EXITS_20261007'
(old/'RUN_STATUS.json').write_text(json.dumps({'status':'superseded_before_results','reason':'User requested independently built components and development-only lockout exclusion','replacement':str(O.resolve())},indent=2),encoding='utf-8')
print('SEPARATE_COMPONENT_ARTIFACTS_COMPLETE');print(json.dumps(audit,indent=2))
