"""Append reviewed transfer evidence without replacing the existing report."""
from pathlib import Path
import json
from datetime import datetime, UTC
import pandas as pd

out=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/COMPONENT_TRANSFER_20261007')
app=Path('work/nq_long_ml/component_report_app/src/data.json')
snapshot=json.loads(app.read_text(encoding='utf-8'))
for key,file in [('transfer','RESEARCH_SCREEN.csv'),('transfer_summary','TRANSFER_SUMMARY.csv'),('transfer_folds','FOLD_METRICS.csv')]:
    source=out/file
    snapshot['queries'][key]={'rows':json.loads(pd.read_csv(source).to_json(orient='records')),'source':{
        'type':'file','name':file,'files':[str(source.resolve()),str((out/'PROTOCOL.json').resolve())],
        'description':'100 frozen entry families per matching chart. Existing separately chosen component methods, expanding development fits, January through August13 2025 only. Prior selection contamination remains. No reserved outcomes or live execution.',
        'metricDefinitions':[
            {'label':'Research flag','definition':'At least10 resolved trades, positive base and cost/slippage stress profit, base profit factor>=1.05, profitable trades>=53%, no unknown trades. Exploratory shortlist, not independent validation.'},
            {'label':'Profit delta','definition':'Policy net profit minus the existing mean70-target/no-stop baseline for the same entry family and duration.'},
            {'label':'Paired uncertainty','definition':'2.5th and97.5th percentiles of2000 circular ten-session block resamples of paired daily PNL differences; conditional on the existing selected entry/exit choices, no full-search multiplicity correction.'},
            {'label':'Profit without best day','definition':'Total net dollars minus the largest positive session PNL, or no deduction when no session was profitable.'}]}}
snapshot['generatedAt']=datetime.now(UTC).isoformat()
snapshot['buildStatus']='complete'
app.write_text(json.dumps(snapshot,separators=(',',':')),encoding='utf-8')
print('TRANSFER_REPORT_PACKAGED',len(snapshot['queries']['transfer']['rows']))
