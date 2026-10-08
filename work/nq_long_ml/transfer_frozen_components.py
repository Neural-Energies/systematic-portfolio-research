"""Replay frozen component methods across all shortlisted entry families.

Development transfer diagnostics, not independent validation or new model search.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from systematic_research.mae_brackets import bracket_outcomes
from systematic_research.execution_stress import schedule_events, marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis
from systematic_research.research_partitions import purged_training_rows, require_before_lockout

ROOT = Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
PREVIOUS = ROOT / 'SEPARATE_COMPONENTS_20261007'
OUT = ROOT / 'COMPONENT_TRANSFER_20261007'
OUT.mkdir(exist_ok=True)
LOCKOUT = pd.Timestamp('2025-08-21', tz='America/New_York').tz_convert('UTC')
LABEL_END = pd.Timestamp('2025-08-14 09:30', tz='America/New_York').tz_convert('UTC')
FOLDS = [pd.Timestamp(t, tz='America/New_York').tz_convert('UTC') for t in
         ['2025-01-01', '2025-04-01', '2025-07-01', '2025-08-14 09:30']]
LEADERS = {5:'NQ_001049588361',15:'NQ_000502850189',60:'NQ_001009785391',240:'NQ_000054683951'}
metrics, fold_rows, trade_rows, audits = [], [], [], []
# Reuse the exact fixed estimator definitions; do not execute the old runner.
old = Path('work/nq_long_ml/separate_exit_components.py').read_text()
exec(old[old.index('from sklearn.pipeline'):old.index('OUT=Path')], globals())
exec(old[old.index('def make_model'):old.index('def research')], globals())
choices = pd.read_csv(PREVIOUS / 'COMPONENT_SELECTIONS.csv')
protocol = json.loads((PREVIOUS / 'PROTOCOL.json').read_text())
protocol.update(phase='Frozen component transfer across 100 entry families per timeframe',
    selection='No new estimator, feature, factor or exit tuning. Prior separate component choices transferred unchanged. Existing baseline remains visible. Candidate flags are descriptive development screens.',
    scope='100 shortlisted families x four matching charts x baseline/target-only/MAE-only/combined x base/stress. One position at a time per candidate, not a 100-position portfolio.',
    uncertainty='Ten-session circular block resampling of paired daily PNL, 2000 draws, seed20261007. Conditional on selected entries/exits; no full-search multiplicity correction. Earlier selection contamination remains.',
    screen='At least10 resolved trades, positive base and stress profit, basePF>=1.05 and positive net-win-rate>=53%; quarters, drawdown and best-day concentration are reported rather than vetoes.')
(OUT / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))

def transfer(duration, m, f, b, features, signals):
    require_before_lockout(m.index, LOCKOUT)
    require_before_lockout(pd.DatetimeIndex(f.planned_exit), LOCKOUT)
    scale = pd.Series((b.high-b.low).ewm(span=14,adjust=False,min_periods=14).mean().to_numpy(),
        index=pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill').clip(lower=.25)
    x = features.replace([np.inf,-np.inf],np.nan).copy()
    x['clock_minutes']=f.slot; x['prior_range_points']=scale
    x['open_gap_scaled']=(f.entry_open-b.close.set_axis(pd.DatetimeIndex(b.available_at)).reindex(f.index,method='ffill'))/scale
    names = {r.component:r.selected for r in choices.query('duration==@duration').itertuples()}
    predicted = {k:np.full(len(f),np.nan) for k in names}
    eligible=f.eligible.to_numpy()&scale.notna().to_numpy()
    evaluation=np.zeros(len(f),dtype=bool)
    for start,end in zip(FOLDS[:-1],FOLDS[1:]):
        train=purged_training_rows(f.index,pd.DatetimeIndex(f.planned_exit),pd.DatetimeIndex(schedule.open),start)&eligible&f.scorable.to_numpy()
        test=np.asarray((f.index>=start)&(f.index<end))&eligible; evaluation|=test
        assert not np.any(train&test)
        audits.append(dict(duration=duration,fold_start=str(start),training_rows=int(train.sum()),test_rows=int(test.sum()),training_last_label=str(pd.DatetimeIndex(f.planned_exit)[train].max()),reserved_rows=0))
        for component,winner in names.items():
            if winner=='baseline':continue
            label='MFE' if component=='target' else 'MAE'
            values=(f.mfe_points if label=='MFE' else f.mae_points)/scale
            if winner.startswith('stop_historical'):
                raw=values[train].quantile(float(winner.split('_')[-1]))*scale[test].to_numpy()
            else:
                name=winner.removeprefix('target_').split('_x')[0] if component=='target' else winner.removeprefix('stop_')
                model=make_model(label,name); log=name in ['ridge','extra_trees']
                model.fit(x.loc[train],np.log1p(values[train]) if log else values[train])
                raw=model.predict(x.loc[test]); raw=np.expm1(np.clip(raw,0,10)) if log else np.maximum(raw,0)
                # Preserve original forecast-rounding then factor-rounding order.
                raw=np.ceil(np.maximum(raw*scale[test].to_numpy(),.25)*4)/4
                if component=='target':raw=raw*float(winner.split('_x')[-1])
            predicted[component][test]=np.ceil(raw*4)/4
        print('TRANSFER_FOLD',duration,str(start),flush=True)
    target=f.target_points.to_numpy(); stop=np.full(len(f),np.inf)
    if names['target']!='baseline':target=predicted['target']
    if names['MAE_exit']!='baseline':stop=predicted['MAE_exit']
    settings={'baseline':(f.target_points.to_numpy(),np.full(len(f),np.inf)),
        'target_only':(target,np.full(len(f),np.inf)),
        'MAE_only':(f.target_points.to_numpy(),stop),'combined':(target,stop)}
    clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float)
    starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8
    sessions=pd.DatetimeIndex(f.loc[(f.index>=FOLDS[0])&(f.index<FOLDS[-1]),'anchor'].unique()).sort_values()
    rng=np.random.default_rng(20261007+duration)
    n=len(sessions); block=10; draws=2000
    sample=((rng.integers(0,n,(draws,int(np.ceil(n/block))))[:,:,None]+np.arange(block))%n).reshape(draws,-1)[:,:n]
    daily_cache={}
    for policy,(targets,stops) in settings.items():
        for stress in [False,True]:
            cost=50. if stress else 25.;slip=.25 if stress else 0.
            outcomes=bracket_outcomes(clock,ohlc,starts,ends,f.entry_open.to_numpy()+targets,stops,0,cost,slip,slip,slip)
            for cid,signal in signals.items():
                if cid=='ELIGIBLE_ENTRY_BASELINE':continue
                selected=schedule_events(signal&evaluation&np.isfinite(targets)&(targets>0)&(stops>0),starts,outcomes)
                a=outcomes[selected]
                t=f.iloc[selected][['anchor','planned_exit']].copy()
                t['entry_time']=f.index[selected];t['entry_price']=a[:,4];t['exit_price']=a[:,5]
                t['target_points']=targets[selected];t['stop_points']=stops[selected]
                t['exit_time']=pd.to_datetime(a[:,0].astype(np.int64),utc=True);t['net_dollars']=a[:,1]
                t['status']=np.where(a[:,3]==0,'unresolved',np.where(a[:,11]==-1,'stop',np.where(a[:,2]==1,'target','timeout')))
                t['holding_minutes']=a[:,8];t['mae_upper_points']=a[:,6]
                kp,curve=trade_kpis(t,sessions)
                dd,low=marked_drawdowns(clock,ohlc,starts,outcomes,selected,0,cost)
                daily=curve.pnl_dollars.to_numpy(); key=(cid,stress)
                if policy=='baseline':daily_cache[key]=daily.copy()
                delta=daily-daily_cache[key]
                boot=delta[sample].sum(axis=1)
                kp.update(duration=duration,candidate=cid,policy=policy,stress=stress,minute_close_dd=dd,minute_low_dd_bound=low,
                    stop_rate=float(t.status.eq('stop').mean()),net_without_best_day=float(daily.sum()-max(daily.max(),0)),
                    profit_delta_to_baseline=float(delta.sum()),delta_ci_low=float(np.quantile(boot,.025)),delta_ci_high=float(np.quantile(boot,.975)),
                    bootstrap_positive_delta_fraction=float((boot>0).mean()))
                metrics.append(kp)
                for j,(start,end) in enumerate(zip(FOLDS[:-1],FOLDS[1:])):
                    part=t[(t.entry_time>=start)&(t.entry_time<end)]
                    resolved=part[part.status.ne('unresolved')]
                    fold_rows.append(dict(duration=duration,candidate=cid,policy=policy,stress=stress,fold=j+1,
                        trades=len(resolved),net_profit=float(resolved.net_dollars.sum())))
                t['duration']=duration;t['candidate']=cid;t['policy']=policy;t['stress']=stress
                trade_rows.append(t)
            print('TRANSFER_POLICY',duration,policy,stress,flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'ALL_METRICS.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(OUT/'FOLD_METRICS.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'FOLD_INPUT_AUDIT.csv',index=False)
    pd.concat(trade_rows,ignore_index=True).to_parquet(OUT/'ALL_TRADES.parquet',index=False)
    # Independent regression check: old leader must replay identically.
    expected=pd.read_csv(PREVIOUS/'BACKTEST_COMPARISONS.csv')
    actual=pd.DataFrame(metrics)
    for policy,old_name in [('baseline','baseline'),('target_only',names['target']),('MAE_only',names['MAE_exit']),('combined','combined_frozen_components')]:
        for stress in [False,True]:
            e=expected[(expected.duration==duration)&(expected.variant==old_name)&(expected.stress==stress)].iloc[0]
            r=actual[(actual.duration==duration)&(actual.candidate==LEADERS[duration])&(actual.policy==policy)&(actual.stress==stress)].iloc[0]
            for field in ['trades','net_profit','minute_close_dd','profit_factor']:
                assert np.isclose(e[field],r[field],equal_nan=True), (duration,policy,stress,field,e[field],r[field])
    print('LEADER_REPLAY_VERIFIED',duration,flush=True)

source=Path('work/nq_long_ml/run_four_timeframes.py').read_text()
prefix=source[:source.index('    for target_type,column')]
prefix=prefix.replace("O=L/'FOUR_TIMEFRAMES_20261007'","O=L/'COMPONENT_TRANSFER_20261007'")
prefix=prefix.replace('del original\n','del original\nm=m.loc[m.index<LABEL_END].copy();schedule=schedule.loc[schedule.open<LABEL_END].copy()\n')
prefix=prefix.replace("(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')",'')
prefix=prefix.replace('metrics=[];thresholds=[];profiles=None','thresholds=[];profiles=None')
prefix+='\n    transfer(duration,m,f,b,features,signals)\n'
exec(compile(prefix,'<development-only transfer preparation>','exec'),globals())
d=pd.DataFrame(metrics)
assert len(d)==3200 and not d.duplicated(['duration','candidate','policy','stress']).any()
base=d[~d.stress].copy();stress=d[d.stress].set_index(['duration','candidate','policy'])
base['stress_profit']=[stress.loc[(r.duration,r.candidate,r.policy),'net_profit'] for r in base.itertuples()]
base['worth_further_research']=(base.trades>=10)&(base.net_profit>0)&(base.stress_profit>0)&(base.profit_factor>=1.05)&(base.net_win_rate>=.53)&base.unknown_trades.eq(0)
base.to_csv(OUT/'RESEARCH_SCREEN.csv',index=False)
counts=base.groupby(['duration','policy']).agg(positive=('net_profit',lambda s:int((s>0).sum())),research_candidates=('worth_further_research','sum'),median_profit=('net_profit','median'),median_improvement=('profit_delta_to_baseline','median'))
counts.to_csv(OUT/'TRANSFER_SUMMARY.csv')
(OUT/'VERIFICATION.json').write_text(json.dumps({'metrics_rows':len(d),'unique_candidates_per_timeframe':100,'leader_replay_checks':128,'reserved_rows':0,'seed':20261007,'status':'complete'},indent=2))
print(counts.to_string());print('COMPONENT_TRANSFER_COMPLETE',flush=True)
