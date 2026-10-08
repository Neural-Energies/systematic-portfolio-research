from pathlib import Path
import json
import pandas as pd

root=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
out=root/'COMPONENT_TRANSFER_20261007'
d=pd.read_csv(out/'RESEARCH_SCREEN.csv')
assert len(d)==1600 and not d.duplicated(['duration','candidate','policy']).any()
definitions={s['id']:s for s in json.loads((root/'LOCKED_ENTRIES.json').read_text())}
thresholds=pd.read_csv(root/'FOUR_TIMEFRAMES_20261007/IS_ONLY_THRESHOLDS.csv').set_index(['duration','atom_id'])
configs=out/'research_candidates';configs.mkdir(exist_ok=True)
for row in d[d.worth_further_research].to_dict('records'):
    duration=int(row['duration'])
    atoms=[]
    for atom in definitions[row['candidate']]['definition']:
        exact=thresholds.loc[(duration,atom['atom_id'])].to_dict()
        atoms.append(dict(atom_id=atom['atom_id'],**exact))
    row.update(entry_definition={'id':row['candidate'],'conjunction':'AND','matching_chart_minutes':duration,'opening_up_required':True,'atoms':atoms},
        thresholds_file=str((root/'FOUR_TIMEFRAMES_20261007/IS_ONLY_THRESHOLDS.csv').resolve()),
        separately_saved_components=str((root/'SEPARATE_COMPONENTS_20261007/combined_configs'/f'{duration}m.json').resolve()),
        target_component_active=row['policy'] in ['target_only','combined'],
        mae_component_active=row['policy'] in ['MAE_only','combined'],
        target_model_file=str((root/'SEPARATE_COMPONENTS_20261007/target_models'/f'{duration}m.joblib').resolve()),
        mae_model_file=str((root/'SEPARATE_COMPONENTS_20261007/mae_exit_models'/f'{duration}m.joblib').resolve()),
        status='Development research candidate; policy controls which components are active; no live or reserved-period validation')
    (configs/f"{row['candidate']}_{duration}m_{row['policy']}.json").write_text(json.dumps(row,indent=2))
summary=pd.read_csv(out/'TRANSFER_SUMMARY.csv')
text=['# Frozen component transfer across the NQ shortlist','',
 '100 entry families on each matching 5-,15-,60-,240-minute RTH chart. Existing component methods were transferred without candidate-specific tuning. Each candidate is simulated independently, one NQ contract, one position at a time. This is not a portfolio.',
 '', 'Evaluation: January1–August13,2025; expanding models fit only earlier permitted labels with five cash sessions purged. No reserved outcomes evaluated. Entries and exits were previously selected, so these are development diagnostics, not blind confirmation.', '',summary.to_markdown(index=False),'',
 'Research flags require at least10 trades, positive base/stress profit, basePF>=1.05, net win>=53%, no unresolved trades. Drawdown, quarters and best-day concentration are shown rather than vetoes. Related rules and policy configurations overlap; do not add counts as independent edges.', '',
 'Ten-session block resampling gives a conditional paired profit-change interval,2000 draws. It does not correct the original search or subsequent exit selection. A zero-crossing interval means improvement remains uncertain.']
for duration in [5,15,60,240]:
    part=d[(d.duration==duration)&d.worth_further_research].sort_values('net_profit',ascending=False).head(10)
    columns=['candidate','policy','trades','net_profit','stress_profit','profit_factor','net_win_rate','minute_close_dd','profit_delta_to_baseline','net_without_best_day','delta_ci_low','delta_ci_high']
    text.extend(['',f'## {duration}-minute research leaders','',part[columns].to_markdown(index=False)])
(out/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8')
v=json.loads((out/'VERIFICATION.json').read_text())
v.update(repository_tests='191 passed',baseline_static_failures='40ruff errors;89mypy errors in11files;6format files,unchanged',saved_candidate_configs=int(d.worth_further_research.sum()))
(out/'VERIFICATION.json').write_text(json.dumps(v,indent=2))
print(summary.to_string(index=False)); print('SAVED_CONFIGS',int(d.worth_further_research.sum()))
