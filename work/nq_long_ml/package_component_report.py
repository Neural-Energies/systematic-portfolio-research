from pathlib import Path
import json
import pandas as pd
from datetime import datetime,UTC

O=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/SEPARATE_COMPONENTS_20261007')
A=Path('work/nq_long_ml/component_report_app')
snapshot={'surface':'report','title':'NQ targets and MAE exits: separate component research','status':'reviewed','buildStatus':'complete' if (O/'COMPONENT_SELECTIONS.csv').exists() else 'creating','generatedAt':datetime.now(UTC).isoformat(),'report':{'asOf':'2025-08-13'},'queries':{}}
if (A/'src/data.json').exists():snapshot['id']=json.loads((A/'src/data.json').read_text(encoding='utf-8'))['id']
sources={'metrics':'BACKTEST_COMPARISONS.csv','errors':'FORECAST_ERRORS.csv','audit':'FOLD_INPUT_AUDIT.csv','quality':'DATA_QUALITY.csv','choices':'COMPONENT_SELECTIONS.csv'}
for key,file in sources.items():
    p=O/file
    if p.exists():rows=json.loads(pd.read_csv(p).to_json(orient='records'))
    elif key=='choices':rows=[json.loads(p.read_text()) for p in sorted(O.glob('*_FROZEN.json'))]
    else:rows=[]
    snapshot['queries'][key]={'rows':rows,'source':{'type':'file','name':file,'files':[str(p.resolve())],'description':'Locally calculated historical development research; no reserved-period evaluation. All values simulated, one NQ contract.','metricDefinitions':[]}}
definitions=[{'label':'Net profit','definition':'Sum of individual trade net dollars, with $20 per point, stated entry/exit slippage and round-trip costs. Constant one-contract exposure.'},
 {'label':'Trade profit factor','definition':'Gross positive net trade dollars divided by absolute negative net trade dollars; target, stop and time exits included.'},
 {'label':'Minute drawdown','definition':'Largest peak-to-trough loss of the minute-close marked cumulative PNL, including open positions. Minute-low bound is separately retained.'},
 {'label':'Win rate','definition':'Fraction of resolved target, stop and time-exit trades with positive net PNL. Separate from target-hit rate.'},
 {'label':'Evaluation','definition':'Stitched expanding development folds January1–August13,2025; training earlier data with five cash sessions purged at each fold. No lockout outcomes accessed.'}]
snapshot['queries']['metrics']['source']['metricDefinitions']=definitions
snapshot['queries']['errors']['source']['metricDefinitions']=[{'label':'MAE / MSE / RMSE','definition':'Errors between forecast and observed full-window high-minus-open (MFE) or open-minus-low (MAE), computed only in held-forward development folds; points or points squared.'},
 {'label':'Coverage','definition':'Observed fraction of full-window excursions at or below forecast; measured calibration, not a guaranteed live probability.'}]
snapshot['queries']['audit']['source']['metricDefinitions']=[{'label':'Training rows','definition':'All earlier eligible opportunities with complete labels, excluding the five cash sessions before the evaluation fold.'}]
snapshot['queries']['choices']['source']['metricDefinitions']=[{'label':'Selection','definition':'Target and MAE-exit chosen independently by maximum stitched development OOS profit; existing baseline included; minimum10trades/80%baseline count, drawdown tie-break. Choices are development selections, not lockout-validated winners.'}]
e=pd.DataFrame(snapshot['queries']['errors']['rows']);summaries=[]
if len(e):
    for (duration,label,model),g in e.groupby(['duration','label','model']):
        n=g.samples.sum();mse=(g.MSE_points*g.samples).sum()/n
        summaries.append({'duration':int(duration),'label':label,'model':model,'samples':int(n),'MAE_points':float((g.MAE_points*g.samples).sum()/n),'RMSE_points':float(mse**.5),'MSE_points':float(mse),'coverage':float((g.coverage_actual_below_forecast*g.samples).sum()/n)})
snapshot['queries']['error_summary']={'rows':summaries,'source':{'type':'file','name':'Pooled forecast errors','files':[str((O/'FORECAST_ERRORS.csv').resolve())],'description':'Sample-count-weighted fold errors. PooledRMSE=sqrt(sum(foldMSE*foldN)/sumN);MAEandcoverageweightedbyN. Forecastopportunities,notonlyselected trades.','metricDefinitions':snapshot['queries']['errors']['source']['metricDefinitions']}}
p=A/'src/data.json' if A.exists() else Path('work/nq_long_ml/component_reviewed_snapshot.json')
p.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
print('SNAPSHOT',p,'STATUS',snapshot['buildStatus'],'ROWS',{k:len(v['rows']) for k,v in snapshot['queries'].items()})
