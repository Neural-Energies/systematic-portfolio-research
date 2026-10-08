"""Four matching RTH chart/holding-duration screens, corrected minute-open clock."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import sys,json,hashlib,html
import numpy as np
import pandas as pd
from systematic_research.rth_bar_grid import rth_bar_grid,completed_rth_bars,minute_start_clock
from systematic_research.fixed_hold import fixed_hold_outcomes
from systematic_research.clock_excursion import same_clock_history
from systematic_research.execution_stress import minute_outcomes,schedule_events,marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis
sys.path.insert(0,str(Path('work/nq_long_ml').resolve()))
from experiment import aggregate

R=Path('data/processed/nq_entry_discovery_202210_minute_end_scenario')
L=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
O=L/'FOUR_TIMEFRAMES_20261007';O.mkdir(exist_ok=True)
m=pd.read_parquet(R/'minutes.parquet');schedule=pd.read_parquet(R/'rth_schedule.parquet')
original=pd.read_parquet('data/processed/nq_entry_discovery_202210/minutes.parquet')
assert m.index.equals(minute_start_clock(original.index,labels_at_end=True))
assert np.array_equal(m.to_numpy(),original.to_numpy());del original
short=json.loads((L/'LOCKED_ENTRIES.json').read_text());atoms=pd.read_csv(L/'FULL_VALIDATION_20261007/exact_frozen_atoms.csv')
old_f=pd.read_parquet('data/processed/nq_entry_discovery_202210/opportunities.parquet')
old_features=pd.read_parquet('data/processed/nq_entry_discovery_202210/features.parquet')
train=old_f.split.eq('in_sample')&old_f.eligible
quantiles={}
for a in atoms.itertuples():
    q=old_features.loc[train,a.feature].replace([np.inf,-np.inf],np.nan).dropna().quantile(np.arange(.1,1,.1))
    quantiles[a.atom_id]=float(q.index[np.argmin(np.abs(q.to_numpy()-a.threshold_full_precision))])
del old_f,old_features
feature_source=Path('work/nq_long_ml/build_oct2022_atoms.py').read_text().split("train=f.split.eq('in_sample')")[0]
feature_source=feature_source.replace("f=pd.read_parquet(R/'opportunities.parquet');m=pd.read_parquet(R/'minutes.parquet');b=pd.read_parquet(R/'completed_5min.parquet')",'')
feature_source=feature_source.replace(".reindex(f.index)\n    families", ".reindex(f.index,method='ffill')\n    families")
feature_source=feature_source.replace('eb=aggregate(es,5)','eb=completed_rth_bars(es,grid)')
clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float)
scenarios={'base':(0,25.,0.,0.,0.),'one_tick':(0,25.,.25,.25,.25),'cost50':(0,50.,0.,0.,0.)}
protocol={'source_clock':'Minute-end export labels normalized once by minus1minute; OHLC unchanged; overlap audit6849exactbars supports correction','source_label_confirmation':'Exporter endpoint semantics inferred from overlap, not separately vendor-confirmed',
 'durations_minutes':[5,15,60,240],'chart_anchor':'09:30America/New_York; half-open windows, partial last bar capped at cash close; half-days respected',
 'entry':'Buy actual window open, using prior completed chart bars and observed opening-up check; no current bar high/close/volume used as signal',
 'signals':'100 locked rule families mapped to each matching RTH chart; original feature/operator/quantile rank retained, numeric thresholds refit only on that chart2022-2024; these are new timeframe variants, not exact old100 frozen numeric rules',
 'opening_up':'Current interval open > previous completed RTH interval open, including previous-session final bar at09:30',
 'targets':['70% of prior20same-clock conditional IQR-trimmed mean high-open','70th percentile of that same historical distribution'],
 'exit':'No stop; target or final minute close at duration/session end','scenarios':scenarios,'cost_units':'USD per1NQcontract;$20/indexpoint',
 'paper_candidate_screen':'30resolved post2024trades; combinedPF>=1.05; combinedprofit>0; combinednetwin>=53%; latest2026profit>0 andPF>1; no unresolved selected paths. No dailyminimum, quarter-profit requirement or severe-stress gate.',
 'limitations':'Previouslyseen2025/2026;10moriginalselection bias and400timeframevariants; no live performance; volume-rollprovenanceunverified; fresh5minimportnot scored without matchingES/minute inputs; no stop or sizing model',
 'locked_rule_sha256':hashlib.sha256((L/'LOCKED_ENTRIES.json').read_bytes()).hexdigest()}
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
metrics=[];thresholds=[];profiles=None
for duration in [5,15,60,240]:
    print('START_DURATION',duration,flush=True)
    grid=rth_bar_grid(schedule,duration)
    f=fixed_hold_outcomes(m,grid.index,pd.DatetimeIndex(grid.planned_exit));f['anchor']=grid.anchor
    local=f.index.tz_convert('America/New_York');f['slot']=local.hour*60+local.minute
    f['opening_up']=f.entry_open.gt(f.entry_open.shift());f['excursion']=f.mfe_points
    history=same_clock_history(f,pd.DatetimeIndex(schedule.open))
    f['target_points']=np.ceil(history.mean70*4)/4;f['percentile70_points']=np.ceil(history.q70*4)/4
    f['eligible']=f.opening_up&f.target_points.gt(0)&f.entry_open.notna()
    f['split']=np.where(local.year<=2024,'in_sample',np.where(local.year==2025,'validation','final_confirmation'))
    embargo=[]
    for year in [2025,2026]:embargo+=schedule[schedule.index.year==year].head(5).open.tolist()
    f.loc[f.anchor.isin(embargo),'split']='embargo'
    b=completed_rth_bars(m,grid)
    src=feature_source
    if profiles is not None:
        start=src.index('# Profile uses');end=src.index("for column in ['poc','val','vah']:",start)
        src=src[:start]+src[end:]
    namespace={'f':f,'m':m,'b':b,'grid':grid,'completed_rth_bars':completed_rth_bars,'prof':profiles}
    exec(compile(src,'<matching-chart-features>','exec'),namespace)
    if profiles is None:profiles=namespace['prof']
    features=pd.DataFrame(namespace['features'],index=f.index)
    train=f.split.eq('in_sample')&f.eligible
    masks={}
    for a in atoms.itertuples():
        threshold=features.loc[train,a.feature].replace([np.inf,-np.inf],np.nan).dropna().quantile(quantiles[a.atom_id])
        thresholds.append({'duration':duration,'atom_id':a.atom_id,'feature':a.feature,'operator':a.operator,'IS_quantile':quantiles[a.atom_id],'threshold':threshold})
        masks[a.atom_id]=(features[a.feature].gt(threshold) if a.operator=='>' else features[a.feature].lt(threshold)).to_numpy()
    signals={s['id']:np.logical_and.reduce([masks[a['atom_id']] for a in s['definition']])&f.eligible.to_numpy() for s in short}
    signals['ELIGIBLE_ENTRY_BASELINE']=f.eligible.to_numpy()
    starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8
    for target_type,column in [('mean70','target_points'),('percentile70','percentile70_points')]:
        targets=(f.entry_open+f[column]).to_numpy(float)
        for variant,args in scenarios.items():
            outcomes=minute_outcomes(clock,ohlc,starts,ends,targets,*args)
            for cid,signal in signals.items():
                for period in ['in_sample','validation','final_confirmation','post2024']:
                    scope=f.split.isin(['validation','final_confirmation']) if period=='post2024' else f.split.eq(period)
                    selected=schedule_events(signal&scope.to_numpy()&f[column].gt(0).to_numpy(),starts,outcomes)
                    a=outcomes[selected]
                    t=f.iloc[selected][['anchor','planned_exit']].copy();t['target_points']=f[column].iloc[selected]
                    t['entry_time']=f.index[selected];t['entry_price']=a[:,4];t['exit_price']=a[:,5]
                    t['exit_time']=pd.to_datetime(a[:,0].astype(np.int64),utc=True);t['net_dollars']=a[:,1]
                    t['status']=np.where(a[:,3]==0,'unresolved',np.where(a[:,2]==1,'target','timeout'))
                    t['holding_minutes']=a[:,8];t['mae_upper_points']=a[:,6];t['full_window_mfe_points']=f.mfe_points.iloc[selected].to_numpy()
                    sessions=pd.DatetimeIndex(f.loc[scope,'anchor'].unique()).sort_values()
                    kp,curve=trade_kpis(t,sessions);dd,lowdd=marked_drawdowns(clock,ohlc,starts,outcomes,selected,args[0],args[1])
                    kp.update(candidate=cid,duration_minutes=duration,target_type=target_type,period=period,scenario=variant,minute_close_dd=dd,minute_low_dd_bound=lowdd,
                        mean_mfe_to_exit=float(np.nanmean(a[:,7])) if len(a) else np.nan)
                    metrics.append(kp)
    pd.DataFrame(metrics).to_csv(O/'ALL_METRICS.csv',index=False)
    pd.DataFrame(thresholds).to_csv(O/'IS_ONLY_THRESHOLDS.csv',index=False)
    print('COMPLETE_DURATION',duration,'cases',len(metrics),flush=True)

d=pd.DataFrame(metrics);post=d[(d.period=='post2024')&(d.scenario=='base')].copy()
latest=d[(d.period=='final_confirmation')&(d.scenario=='base')].set_index(['candidate','duration_minutes','target_type'])
rows=[]
for r in post.itertuples():
    key=(r.candidate,r.duration_minutes,r.target_type);recent=latest.loc[key]
    passed=r.trades>=30 and r.net_profit>0 and r.profit_factor>=1.05 and r.net_win_rate>=.53 and recent.net_profit>0 and recent.profit_factor>1 and r.unknown_trades==0
    item={'candidate':r.candidate,'duration_minutes':r.duration_minutes,'target_type':r.target_type,'status':'paper_candidate' if passed else 'watch_or_reject',
          'post_trades':r.trades,'post_net_profit':r.net_profit,'post_PF':r.profit_factor,'post_win_pct':100*r.net_win_rate,'post_sharpe':r.sharpe,'minute_DD':r.minute_close_dd,
          'mean_MAE_points':r.mean_mae_points,'target_hit_pct':100*r.target_hit_rate,'latest_trades':recent.trades,'latest_profit':recent.net_profit,'latest_PF':recent.profit_factor}
    rows.append(item)
screen=pd.DataFrame(rows);screen.to_csv(O/'PAPER_SCREEN.csv',index=False)
for r in screen.query("status=='paper_candidate' and candidate!='ELIGIBLE_ENTRY_BASELINE'").to_dict('records'):
    (O/f"{r['candidate']}_{r['duration_minutes']}m_{r['target_type']}.json").write_text(json.dumps(r,indent=2),encoding='utf-8')
counts=screen.query("candidate!='ELIGIBLE_ENTRY_BASELINE'").groupby(['duration_minutes','status']).size().unstack(fill_value=0)
counts.to_csv(O/'COUNTS.csv');print(counts.to_string(),flush=True)
view=screen.sort_values(['duration_minutes','status','post_sharpe'],ascending=[True,True,False])
body='<h1>NQ entries: 5, 15, 60 and 240 minutes</h1><p>Corrected minute-end clock; buy each matching RTH bar open using completed data. Latest data through August20,2026. These are historical paper-candidate screens, not live trades.</p>'
body+='<p>100 rule families per chart, IS-only chart thresholds. Two target definitions:70%mean and70thpercentile; prior20same-clock cash sessions, opening-up conditional, past-only IQR. Cash-open09:30anchor; finalwindow capped atRTHclose. No stop or dailyfrequency minimum. Base$25 round-trip cost,1NQcontract. All numbers inUSD unless labeledpoints.</p>'
body+='<p>Prior reports confused five-minute entry checks with a one-hour exit. This report explicitly rebuilds chart features and excursion targets for each duration. The new five-minute import supports the corrected minute-end alignment, but exporter confirmation and volume-roll provenance remain unverified.</p>'
body+=counts.to_html()+'<p>Paper screen:30post2024trades, positive combinedprofit,PF≥1.05,netwin≥53%,latest2026positive/PF>1,no unknown selected paths. Stronger cost/slippage results are reported for review, not automatic vetoes. Previously seen history and original broad search can inflate performance.</p>'
for h in [5,15,60,240]:
    body+=f'<h2>{h}-minute chart and holding window</h2>'
    part=view[view.duration_minutes.eq(h)]
    body+=part.to_html(index=False,float_format=lambda x:f'{x:,.2f}')
body+='<p><a href="ALL_METRICS.csv">All periods and cost/slippage scenarios</a> · <a href="PAPER_SCREEN.csv">All paper-screen decisions</a> · <a href="PROTOCOL.json">Definitions and assumptions</a></p>'
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>NQ four timeframe entry research</title><style>body{font:16px system-ui;margin:30px;line-height:1.5}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border:1px solid #ddd}th{background:#eee}h2{margin-top:40px}</style>'+body,encoding='utf-8')
print('FINISHED',flush=True)
