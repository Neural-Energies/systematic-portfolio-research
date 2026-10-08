"""Separate target benchmarks and MAE stop-distance sensitivity on development."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from systematic_research.mae_brackets import bracket_outcomes
from systematic_research.execution_stress import schedule_events, marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis

ROOT=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
PRIOR=ROOT/'COMPONENT_TRANSFER_20261007'
OUT=ROOT/'COMPONENT_SENSITIVITY_20261007';OUT.mkdir(exist_ok=True)
LOCKOUT=pd.Timestamp('2025-08-21',tz='America/New_York').tz_convert('UTC')
LABEL_END=pd.Timestamp('2025-08-14 09:30',tz='America/New_York').tz_convert('UTC')
FOLDS=[pd.Timestamp(t,tz='America/New_York').tz_convert('UTC') for t in ['2025-01-01','2025-04-01','2025-07-01','2025-08-14 09:30']]
choices=pd.read_csv(ROOT/'SEPARATE_COMPONENTS_20261007/COMPONENT_SELECTIONS.csv')
metrics=[];audits=[];forecast_rows=[];fold_results=[]
old=Path('work/nq_long_ml/separate_exit_components.py').read_text()
exec(old[old.index('from sklearn.pipeline'):old.index('OUT=Path')],globals())
exec(old[old.index('def make_model'):old.index('def research')],globals())
# Preserve the exact earlier fitting, purging and rounding sequence.
source=Path('work/nq_long_ml/transfer_frozen_components.py').read_text()
fit=source[source.index('    require_before_lockout(m.index'):source.index('    target=f.target_points')]
fit=fit.replace('    for start,end in zip(FOLDS[:-1],FOLDS[1:]):',
    "    simple={key:np.full(len(f),np.nan) for key in ['median','mean','q70','MAE_q75']}\n    unscaled=np.full(len(f),np.nan)\n    for start,end in zip(FOLDS[:-1],FOLDS[1:]):")
fit=fit.replace("        for component,winner in names.items():",
    "        mfe=f.mfe_points/scale;mae=f.mae_points/scale\n        for key,value in [('median',mfe[train].median()),('mean',mfe[train].mean()),('q70',mfe[train].quantile(.7)),('MAE_q75',mae[train].quantile(.75))]:\n            simple[key][test]=np.ceil(np.maximum(value*scale[test].to_numpy(),.25)*4)/4\n        for component,winner in names.items():")
fit=fit.replace("                if component=='target':raw=raw*float(winner.split('_x')[-1])",
    "                if component=='target':\n                    unscaled[test]=raw\n                    raw=raw*float(winner.split('_x')[-1])")
fit=fit.replace("print('TRANSFER_FOLD'","print('SENSITIVITY_FOLD'")
exec('def prepare(duration,m,f,b,features,signals):\n'+fit+'\n    return scale,x,names,predicted,evaluation,simple,unscaled\n',globals())

protocol=json.loads((PRIOR/'PROTOCOL.json').read_text())
protocol.update(phase='Separate target benchmark and risk-distance sensitivity',
    target_test='No stop; frozen learned forecast with original multiplier; unscaled learnedMFE forecast times0.5,0.6,0.7,0.8,0.9,1.0; training-only normalized historicalmedian/mean/q70 MFE times0.7; originalsame-clockmean70baseline.',
    risk_test='Existing originaltarget fixed; chosen MAE distance times0.75,1,1.25,1.5. Fourhour has no previously chosenstop, so historical normalizedMAEq75 is the explicit risk-distance benchmark. No target-stop jointgrid.',
    selection='Exploratory comparisons only. All tested configurations and failures retained; no automatic replacement of frozen component choices. Same development data, not a fresh test.',
    forecast_comparison='Unscaled full-window MFE predictions on all scorable eligible forward opportunities; MAE/MSE/RMSE separate from trading PNL.',
    limitation='Many entry families and sensitivity configurations overlap. Prior selection contamination persists. No claim of independent significance or profit optimum.')
(OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))

def explore(duration,m,f,b,features,signals):
    scale,x,names,pred,evaluation,simple,unscaled=prepare(duration,m,f,b,features,signals)
    no_stop=np.full(len(f),np.inf); original=f.target_points.to_numpy()
    settings={'original_baseline':(original,no_stop,'baseline'),
        'frozen_target':(pred['target'],no_stop,'target')}
    for factor in [.5,.6,.7,.8,.9,1.]:
        settings[f'learned_target_x{factor}']=(np.ceil(unscaled*factor*4)/4,no_stop,'target')
    for key in ['median','mean','q70']:
        settings[f'historical_{key}_target_x0.7']=(np.ceil(simple[key]*.7*4)/4,no_stop,'target')
    risk=pred['MAE_exit'] if names['MAE_exit']!='baseline' else simple['MAE_q75']
    for factor in [.75,1.,1.25,1.5]:
        settings[f'MAE_distance_x{factor}']=(original,np.ceil(risk*factor*4)/4,'risk')
    assert len(settings)==15
    # Cache development-only preparation for subsequent research without rereading the lockout.
    cache=OUT/f'{duration}m_inputs';cache.mkdir(exist_ok=True)
    f.to_parquet(cache/'opportunities.parquet');x.to_parquet(cache/'features.parquet')
    pd.DataFrame({cid:s for cid,s in signals.items()},index=f.index).to_parquet(cache/'signals.parquet')
    pd.DataFrame({'learned_MFE':unscaled,'frozen_target':pred['target'],'risk_reference':risk,
        **simple,'evaluation':evaluation},index=f.index).to_parquet(cache/'forecasts.parquet')
    good=evaluation&f.scorable.to_numpy()
    actual=f.mfe_points.to_numpy()[good]
    for name,values in [('learned',unscaled),('historical_median',simple['median']),('historical_mean',simple['mean']),('historical_q70',simple['q70'])]:
        error=values[good]-actual
        forecast_rows.append(dict(duration=duration,method=name,forecasts=len(actual),MAE_points=float(np.abs(error).mean()),MSE_points=float((error**2).mean()),RMSE_points=float(np.sqrt((error**2).mean())),bias_points=float(error.mean()),actual_below_forecast=float((actual<=values[good]).mean())))
    clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float)
    starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8
    sessions=pd.DatetimeIndex(f.loc[(f.index>=FOLDS[0])&(f.index<FOLDS[-1]),'anchor'].unique()).sort_values()
    for policy,(target,stop,component) in settings.items():
        for stress in [False,True]:
            cost=50. if stress else 25.;slip=.25 if stress else 0.
            a=bracket_outcomes(clock,ohlc,starts,ends,f.entry_open.to_numpy()+target,stop,0,cost,slip,slip,slip)
            for cid,signal in signals.items():
                if cid=='ELIGIBLE_ENTRY_BASELINE':continue
                selected=schedule_events(signal&evaluation&np.isfinite(target)&(target>0)&(stop>0),starts,a)
                r=a[selected];t=f.iloc[selected][['anchor']].copy()
                t['entry_time']=f.index[selected];t['net_dollars']=r[:,1]
                t['status']=np.where(r[:,3]==0,'unresolved',np.where(r[:,11]==-1,'stop',np.where(r[:,2]==1,'target','timeout')))
                t['target_points']=target[selected];t['holding_minutes']=r[:,8];t['mae_upper_points']=r[:,6]
                kp,curve=trade_kpis(t,sessions);dd,low=marked_drawdowns(clock,ohlc,starts,a,selected,0,cost)
                kp.update(duration=duration,candidate=cid,policy=policy,component=component,stress=stress,minute_close_dd=dd,minute_low_dd_bound=low,
                    net_without_best_day=float(curve.pnl_dollars.sum()-max(curve.pnl_dollars.max(),0)),stop_rate=float(t.status.eq('stop').mean()))
                metrics.append(kp)
                for j,(start,end) in enumerate(zip(FOLDS[:-1],FOLDS[1:])):
                    part=t[(t.entry_time>=start)&(t.entry_time<end)&t.status.ne('unresolved')]
                    fold_results.append(dict(duration=duration,candidate=cid,policy=policy,stress=stress,fold=j+1,trades=len(part),net_profit=float(part.net_dollars.sum())))
            print('SENSITIVITY_POLICY',duration,policy,stress,flush=True)
    d=pd.DataFrame(metrics)
    previous=pd.read_csv(PRIOR/'ALL_METRICS.csv')
    for new_name,old_name in [('original_baseline','baseline'),('frozen_target','target_only')]:
        l=d[(d.duration==duration)&d.policy.eq(new_name)].set_index(['candidate','stress'])
        r=previous[(previous.duration==duration)&previous.policy.eq(old_name)].set_index(['candidate','stress'])
        for field in ['trades','net_profit','minute_close_dd','profit_factor']:
            assert np.allclose(l[field].sort_index(),r[field].sort_index(),equal_nan=True),(duration,new_name,field)
    d.to_csv(OUT/'ALL_METRICS.csv',index=False)
    pd.DataFrame(forecast_rows).to_csv(OUT/'FORECAST_COMPARISON.csv',index=False)
    pd.DataFrame(fold_results).to_csv(OUT/'FOLD_METRICS.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'FOLD_INPUT_AUDIT.csv',index=False)
    print('SENSITIVITY_REPLAY_VERIFIED',duration,flush=True)

source=Path('work/nq_long_ml/run_four_timeframes.py').read_text()
prefix=source[:source.index('    for target_type,column')]
prefix=prefix.replace("O=L/'FOUR_TIMEFRAMES_20261007'","O=L/'COMPONENT_SENSITIVITY_20261007'")
prefix=prefix.replace('del original\n','del original\nm=m.loc[m.index<LABEL_END].copy();schedule=schedule.loc[schedule.open<LABEL_END].copy()\n')
prefix=prefix.replace("(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')",'')
prefix=prefix.replace('metrics=[];thresholds=[];profiles=None','thresholds=[];profiles=None')
prefix+='\n    explore(duration,m,f,b,features,signals)\n'
exec(compile(prefix,'<development-only sensitivity preparation>','exec'),globals())
d=pd.DataFrame(metrics);assert len(d)==12000
assert not d.duplicated(['duration','candidate','policy','stress']).any()
base=d[~d.stress].copy();stressed=d[d.stress].set_index(['duration','candidate','policy'])
baseline=base[base.policy.eq('original_baseline')].set_index(['duration','candidate'])
base['stress_profit']=[stressed.loc[(r.duration,r.candidate,r.policy),'net_profit'] for r in base.itertuples()]
base['profit_delta_to_original']=[r.net_profit-baseline.loc[(r.duration,r.candidate),'net_profit'] for r in base.itertuples()]
base['research_flag']=(base.trades>=10)&(base.net_profit>0)&(base.stress_profit>0)&(base.profit_factor>=1.05)&(base.net_win_rate>=.53)&base.unknown_trades.eq(0)
base.to_csv(OUT/'COMPARISON.csv',index=False)
summary=base.groupby(['duration','policy','component']).agg(positive=('net_profit',lambda s:int((s>0).sum())),research_flags=('research_flag','sum'),median_profit=('net_profit','median'),median_delta=('profit_delta_to_original','median'),median_drawdown=('minute_close_dd','median'))
summary.to_csv(OUT/'SUMMARY.csv')
(OUT/'VERIFICATION.json').write_text(json.dumps(dict(status='complete',replays=12000,comparison_rows=6000,baseline_frozen_replay_kpi_checks=6400,reserved_rows=0,automatic_promotion=False),indent=2))
print(summary.to_string());print('COMPONENT_SENSITIVITY_COMPLETE',flush=True)
