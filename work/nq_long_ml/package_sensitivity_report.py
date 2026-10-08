from pathlib import Path
import json
from datetime import datetime,UTC
import pandas as pd

root=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
out=root/'COMPONENT_SENSITIVITY_20261007'
p=Path('work/nq_long_ml/component_report_app/src/data.json')
snapshot=json.loads(p.read_text(encoding='utf-8'))
for key,file in [('sensitivity','COMPARISON.csv'),('sensitivity_summary','SUMMARY.csv'),('sensitivity_errors','FORECAST_COMPARISON.csv')]:
    snapshot['queries'][key]={'rows':json.loads(pd.read_csv(out/file).to_json(orient='records')),
        'source':{'type':'file','name':file,'files':[str((out/file).resolve()),str((out/'PROTOCOL.json').resolve())],
        'description':'100 entry families per chart; January–August13,2025 forward development replay. Target multipliers and historical baselines tested without stops. MAE distance varied with original target unchanged. No joint optimization, no reserved outcomes, no automatic promotion.',
        'metricDefinitions':[
            {'label':'Forecast errors','definition':'Observed full-window high minus entry open compared to unscaled full-window forecasts on all eligible complete forward opportunities. MAE inpoints,MSE inpoints squared,RMSE inpoints. Bias=forecast minusactual. Original MFE forecaster is fixed from the prior component selection.'},
            {'label':'Historical forecast','definition':'Fold-training-only normalized MFE mean/median/q70 multiplied by the current prior completed14barEWMA range; quantile and mean are different objectives. Historical targets use0.7of roundedforecast.'},
            {'label':'Risk distances','definition':'Prior chosen MAE reference multiplied by0.75,1,1.25,1.5. Fourhour reference is training-only normalizedMAEq75 because the previously selected exit was no stop. Original target remains fixed.'},
            {'label':'Profit and drawdown','definition':'OneNQcontract,$20/point,base$25roundtrip;stress$50plus1tickslippage/targetpenetration. Targets/stopsfixedatentry,stop-firstintraminuteambiguity. DDminute-close marked.'},
            {'label':'Research flag','definition':'Atleast10trades,positivebase/stressprofit,basePF>=1.05,netwin>=53%,unknowntradeszero. Exploratoryreviewonly,notindependentvalidation.'}]}}
snapshot['buildStatus']='complete';snapshot['generatedAt']=datetime.now(UTC).isoformat()
p.write_text(json.dumps(snapshot,separators=(',',':')),encoding='utf-8')
screen=pd.read_csv(out/'COMPARISON.csv');definitions={s['id']:s for s in json.loads((root/'LOCKED_ENTRIES.json').read_text())}
thresholds=pd.read_csv(root/'FOUR_TIMEFRAMES_20261007/IS_ONLY_THRESHOLDS.csv').set_index(['duration','atom_id'])
folder=out/'cases';folder.mkdir(exist_ok=True)
policy_codes={name:f'C{j:02d}' for j,name in enumerate(sorted(screen.policy.unique()))}
for row in screen[screen.research_flag].to_dict('records'):
    h=int(row['duration'])
    row.update(status='Exploratory sensitivity result; not promoted',
        entry_atoms=[dict(atom_id=a['atom_id'],**thresholds.loc[(h,a['atom_id'])].to_dict()) for a in definitions[row['candidate']]['definition']],
        original_target='Past20same-clockhigher-openingIQRmeanMFE*0.7;positivequartertickrounding',
        forecast_inputs=str((out/f'{h}m_inputs/forecasts.parquet').resolve()),
        component_model_package=str((root/'SEPARATE_COMPONENTS_20261007'/f'{h}m_COMPONENT_MODELS.joblib').resolve()))
    (folder/f"{row['candidate']}_{h}m_{policy_codes[row['policy']]}.json").write_text(json.dumps(row,indent=2))
print('SENSITIVITY_PACKAGED',len(screen),'SAVED_RESEARCH_CONFIGS',int(screen.research_flag.sum()))
