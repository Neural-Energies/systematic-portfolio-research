"""Frozen-rule historical forward execution, stress, clustering and uncertainty."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib,itertools,sys
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm
from systematic_research.execution_stress import minute_outcomes,schedule_events,marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis
from systematic_research.reference_backtest import backtest_reference

P=Path('data/processed/nq_entry_discovery_202210_minute_end_scenario');L=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007')
O=L/'FULL_VALIDATION_20261007'/'MINUTE_END_SCENARIO';O.mkdir(exist_ok=True)
raw=(L/'LOCKED_ENTRIES.json').read_bytes();short=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
assert sha=='cfe8d4f2b518a339acfaa7a6a4c85f0fb08194bafd3a0862b00fe23afdd6fce8' and len(short)==100
def progress(phase,**kwargs):
    (O/'STATUS.json').write_text(json.dumps({'phase':phase,'scope':'Historical forward simulation only; no live/paper feed or orders','daily_entry_requirement':False,**kwargs},indent=2));print(phase,kwargs,flush=True)
scenarios={'base':(0,25.,0.,0.,0.),'double_cost':(0,50.,0.,0.,0.),'quadruple_cost':(0,100.,0.,0.,0.),'delay_1m':(1,25.,0.,0.,0.),'delay_2m':(2,25.,0.,0.,0.),'penetration_1tick':(0,25.,0.,0.,.25),'penetration_4ticks':(0,25.,0.,0.,1.),'slip_1tick':(0,25.,.25,.25,0.),'combined_stress':(1,50.,.25,.25,.25)}
protocol={'locked_sha256':sha,'daily_entry_requirement':False,'entries':'Same100 locked rules; RTH checks every5m; causal completed features; observed current open used for eligibility','exit':'Original70%-mean high-minus-open target, past20 same-clock sessions, higher-open conditional IQR; otherwise cash-capped one-hour close; no stop','scenarios':scenarios,'cost_units':'USD per one NQ contract round trip; slippage and penetration in index points; $20 per point','holding_extension':'ML pilot retained as separate research; not silently replacing original exits','simulation_survivor_gates':['50 resolved trades in each2025 and2026 base period','No unresolved selected paths in any post-2024 execution scenario','Positive net PNL and PF>=1.10 in every scenario in both post-2024 periods','At least10 base trades and positive net profit in every post-2024 calendar quarter (2026Q3 partial)','Positive fifth-percentile mean daily PNL under five-session block bootstrap (2000 draws)','Base expectancy exceeds same-time/weekday/year eligible-entry randomized placebos with raw p<=.05 (2000 draws); Holm100-trial diagnostic reported separately'],
'provenance_gate':'Simulation survivors are not institutionally certified. Underlying per-contract volume-roll series and timestamp/exporter semantics are unverified. NQ under Databento folder is same export, not independent vendor replication.2025 and2026 previously used; no fresh blind holdout exists.',
'selection_caveat':'Quarter replay of rules fixed by2024 is chronological but selected candidate universe used2025 and previously seen2026. It is not an unbiased new walk-forward strategy-selection test. Neither100-trial Holm nor weekly bootstrap corrects the original ten-million-definition selection.',
'random_seed':20261007,'reference_capital':100000,'max_drawdown_gate':'No invented capital risk tolerance; risk KPIs reported for decision.'}
(O/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2))
(O/'CURRENT_REQUIREMENTS.json').write_text(json.dumps({'effective_date_eastern':'2026-10-07','minimum_one_entry_per_session':False,'scope':'Simulate trading using existing data; no live paper account requested','entries_frozen':100},indent=2))
progress('data_audit')
m=pd.read_parquet(P/'minutes.parquet');f=pd.read_parquet(P/'opportunities.parquet');features=pd.read_parquet(P/'features.parquet');bits=np.load(P/'atom_bits.npy',mmap_mode='r')
clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float);starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8;targets=(f.entry_open+f.target_points).to_numpy(float)
bad=(~np.isfinite(m.to_numpy()).all(axis=1))|(m.high<m[['open','low','close']].max(axis=1)).to_numpy()|(m.low>m[['open','high','close']].min(axis=1)).to_numpy()
if m.index.duplicated().any() or not m.index.is_monotonic_increasing or bad.any():raise RuntimeError('Critical source OHLC/key defect')
schedule=pd.read_parquet(P/'rth_schedule.parquet');coverage=[]
for date,s in schedule.iterrows():
    n=m.index.searchsorted(s.close)-m.index.searchsorted(s.open);expected=int((s.close-s.open).total_seconds()/60)
    coverage.append({'session':str(date.date()),'rows':int(n),'expected':expected,'missing':expected-int(n)})
pd.DataFrame(coverage).to_csv(O/'rth_minute_coverage.csv',index=False)
needed=sorted({q['atom_id'] for item in short for q in item['definition']});registry=pd.read_csv(P/'atoms.csv').set_index('atom_id');train=f.split.eq('in_sample')&f.eligible
recovered=[];bitmismatches=[]
for atom in needed:
    definition=registry.loc[atom,'definition'];field,operator,value=definition.split(';')[0].split();printed=float(value);original_atoms=pd.read_csv(L/'FULL_VALIDATION_20261007'/'exact_frozen_atoms.csv').set_index('atom_id')
    threshold=float(original_atoms.loc[atom,'threshold_full_precision'])
    if not np.isclose(threshold,printed,rtol=1e-10,atol=1e-10):raise RuntimeError('Cannot recover original threshold')
    comparison=features[field].gt(threshold) if operator=='>' else features[field].lt(threshold)
    actual=np.unpackbits(bits[atom].view(np.uint8),bitorder='little')[:len(f)].astype(bool)
    differences=np.flatnonzero(actual!=(comparison&f.eligible).to_numpy())
    bitmismatches.append(len(differences));recovered.append({'atom_id':atom,'feature':field,'operator':operator,'threshold_full_precision':threshold,'original_definition':definition,'bitstream_mismatches':len(differences)})
pd.DataFrame(recovered).to_csv(O/'exact_frozen_atoms.csv',index=False)
if sum(bitmismatches):raise RuntimeError('Locked signal cannot be exactly reconstructed')
canonical=pd.read_parquet('data/processed/databento_canonical/symbol=NQ/bars.parquet',columns=['timestamp_utc','open','high','low','close','volume','source_file'])
same=canonical.assign(timestamp_utc=lambda d:d.timestamp_utc-pd.Timedelta(minutes=1)).set_index('timestamp_utc')[m.columns].reindex(m.index).eq(m).all().all()
audit={'minutes':len(m),'first':str(m.index.min()),'last':str(m.index.max()),'duplicate_keys':int(m.index.duplicated().sum()),'invalid_ohlc':int(bad.sum()),'cash_sessions':len(coverage),'incomplete_cash_sessions':int(sum(x['missing']!=0 for x in coverage)),'missing_cash_minutes':int(sum(max(0,x['missing']) for x in coverage)),'exact_atom_reconstruction':len(needed),'original_thresholds_refitted':False,'atom_bitstream_mismatches':sum(bitmismatches),'independent_NQ_vendor_replication':False,'canonical_NQ_same_export':bool(same),'canonical_source_files':canonical.source_file.unique().tolist(),'volume_roll_provenance':'Unverified; no underlying multiple-contract volume history supplied','bar_timestamp_semantics':'Source minute-end sensitivity: UTC labels minus1minute; new5m end-label interpretation awaiting human confirmation','source_timezone':'America/New_York human confirmed, feature timestamps UTC','data_freshness':'LatestNQ2026-08-20; historical simulation, no live evidence'}
(O/'DATA_AUDIT.json').write_text(json.dumps(audit,indent=2))
signals=[]
for item in short:
    a=[q['atom_id'] for q in item['definition']];words=bits[a[0]]&bits[a[1]]&bits[a[2]]
    signals.append(np.unpackbits(words.view(np.uint8),bitorder='little')[:len(f)].astype(bool))
signals=np.array(signals);ids=[x['id'] for x in short]
progress('execution_scenarios',audited_minutes=len(m),reconstructed_atoms=len(needed))
outcomes={}
for name,(delay,cost,slip,exit_slip,penetration) in scenarios.items():
    print('PRECOMPUTE',name,flush=True)
    outcomes[name]=minute_outcomes(clock,ohlc,starts,ends,targets,delay,cost,slip,exit_slip,penetration)
periods=['in_sample','validation','final_confirmation'];metrics=[];quarter_rows=[];base_logs={};curves={};checks=[]
original=pd.read_csv('work/nq_long_ml/shortlist_quant_reports/all_100_metrics.csv')
for k,cid in enumerate(ids):
    for stage in periods:
        scope=f.split.eq(stage).to_numpy();sessions=pd.DatetimeIndex(f.loc[scope,'anchor'].unique()).sort_values()
        for name,args in scenarios.items():
            a=outcomes[name];chosen=schedule_events(signals[k]&scope,starts,a)
            t=f.iloc[chosen][['anchor','target_points','planned_exit']].copy();t['entry_time']=f.index[chosen]+pd.Timedelta(minutes=args[0]);t['entry_price']=a[chosen,4];t['exit_price']=a[chosen,5]
            t['exit_time']=pd.to_datetime(a[chosen,0].astype(np.int64),utc=True);t['net_dollars']=a[chosen,1];t['status']=np.where(a[chosen,3]==0,'unresolved',np.where(a[chosen,2]==1,'target','timeout'));t['mae_upper_points']=a[chosen,6];t['holding_minutes']=a[chosen,8];t['full_window_mfe_points']=f.mfe_points.iloc[chosen].to_numpy()
            kp,curve=trade_kpis(t,sessions);dd,lowdd=marked_drawdowns(clock,ohlc,starts,a,chosen,args[0],args[1]);kp.update({'candidate':cid,'period':stage,'variant':name,'minute_close_dd':dd,'minute_low_dd_bound':lowdd,'skipped_already_target':int(((signals[k]&scope)&(a[:,3]==2)).sum()),'skipped_expired_window':int(((signals[k]&scope)&(a[:,3]==3)).sum())})
            metrics.append(kp)
            if name=='base':
                previous=original[(original.candidate==cid)&(original.period==stage)].iloc[0]
                checks.append({'candidate':cid,'period':stage,'original_net_profit':previous.net_profit,'minute_end_net_profit':kp['net_profit'],'original_pf':previous.profit_factor,'minute_end_pf':kp['profit_factor'],'original_trades':previous.trades,'minute_end_trades':kp['trades']})
                base_logs[(cid,stage)]=t;curves[(cid,stage)]=curve
                t.to_csv(O/f'{cid}_{stage}_trades.csv',index=False)
                curve.to_csv(O/f'{cid}_{stage}_daily.csv',index_label='cash_session_open')
                quarter=pd.DatetimeIndex(t.entry_time).tz_convert('America/New_York').tz_localize(None).to_period('Q')
                for q in pd.PeriodIndex(pd.DatetimeIndex(sessions).tz_convert('America/New_York').tz_localize(None),freq='Q').unique():
                    part=t.loc[quarter==q];valid=part[part.status.ne('unresolved')];loss=valid.net_dollars[valid.net_dollars<0].sum()
                    quarter_rows.append({'candidate':cid,'period':stage,'quarter':str(q),'trades':len(valid),'net_profit':valid.net_dollars.sum(),'profit_factor':valid.net_dollars[valid.net_dollars>0].sum()/-loss if loss<0 else np.nan,'net_win_rate':valid.net_dollars.gt(0).mean(),'unresolved':part.status.eq('unresolved').sum()})
    if (k+1)%10==0:progress('execution_scenarios',candidates_complete=k+1)
    pd.DataFrame(metrics).to_csv(O/'all_execution_metrics.csv',index=False)
pd.DataFrame(quarter_rows).to_csv(O/'quarterly_results.csv',index=False)
pd.DataFrame(checks).to_csv(O/'base_reconciliation.csv',index=False)
# Independent slow engine verifies trade paths and minute marks for three rules.
for cid in [ids[0],'NQ_000926441771',ids[-1]]:
    k=ids.index(cid);scope=f.split.eq('final_confirmation');op=f.loc[scope&f.eligible]
    for scenario in ['base','delay_1m']:
        delay,cost,slip,exit_slip,penetration=scenarios[scenario]
        reference,marks=backtest_reference(m,op,pd.Series(signals[k],index=f.index),cost=cost,latency_minutes=delay,penetration=penetration)
        valid=reference[reference.status.isin(['target','timeout'])];target=pd.DataFrame(metrics).query('candidate==@cid and period=="final_confirmation" and variant==@scenario').iloc[0]
        eq=np.r_[0.,marks.equity.to_numpy()];dd=np.max(np.maximum.accumulate(eq)-eq)
        assert len(valid)==target.trades and np.isclose(valid.net_dollars.sum(),target.net_profit) and np.isclose(dd,target.minute_close_dd)
(O/'EXECUTION_VERIFICATION.json').write_text(json.dumps({'all300_base_period_results_reconciled':False,'reason':'New clock-end scenario compared separately with archived original clock results; all original numeric thresholds remain fixed','independent_slow_minute_engine_cases':6,'passed':True},indent=2))
progress('grouping_and_statistical_diagnostics')
# Group only by validation signal similarity, before inspecting2026 survivor gates.
val=f.split.eq('validation').to_numpy();a=signals[:,val].astype(np.int32);intersection=a@a.T;counts=a.sum(axis=1);union=counts[:,None]+counts[None,:]-intersection;jaccard=np.divide(intersection,union,out=np.zeros_like(intersection,dtype=float),where=union>0)
pd.DataFrame(jaccard,index=ids,columns=ids).to_csv(O/'validation_signal_jaccard.csv')
parent=np.arange(100)
def find(i):
    while parent[i]!=i:i=int(parent[i])
    return i
for i,j in itertools.combinations(range(100),2):
    if jaccard[i,j]>=.8:parent[find(j)]=find(i)
groups={};grows=[]
metric=pd.DataFrame(metrics)
for i,cid in enumerate(ids):groups.setdefault(find(i),[]).append(cid)
for group,members in enumerate(groups.values(),1):
    v=metric[(metric.period=='validation')&metric.candidate.isin(members)]
    rank=v.groupby('candidate').profit_factor.min().sort_values(ascending=False);representative=str(rank.index[0])
    for cid in members:grows.append({'candidate':cid,'group':group,'group_size':len(members),'representative':representative,'is_representative':cid==representative})
pd.DataFrame(grows).to_csv(O/'candidate_groups.csv',index=False)
(O/'FROZEN_GROUP_REPRESENTATIVES.json').write_text(json.dumps({'selection':'Worst validation scenario PF; no2026 re-ranking','groups':len(groups),'representatives':sorted({r['representative'] for r in grows}),'warning':'Validation was used in original selection; not a fresh selection period'},indent=2))
post=f.split.isin(['validation','final_confirmation']).to_numpy();post_sessions=pd.DatetimeIndex(f.loc[post,'anchor'].unique()).sort_values();daily=np.column_stack([pd.concat([curves[(cid,'validation')].pnl_dollars,curves[(cid,'final_confirmation')].pnl_dollars]).reindex(post_sessions,fill_value=0).to_numpy() for cid in ids])
rng=np.random.default_rng(20261007);N=len(post_sessions);weights=np.zeros((2000,N))
for i in range(2000):
    block=rng.integers(0,N,size=int(np.ceil(N/5)));idx=((block[:,None]+np.arange(5))%N).ravel()[:N];weights[i]=np.bincount(idx,minlength=N)/N
boot=weights@daily;ci=np.quantile(boot,[.05,.5,.95],axis=0)

@njit
def placebo_test(pool,group_bounds,group_counts,starts,ends,pnl,actual,seed):
    np.random.seed(seed);n=len(starts);beats=0;means=np.empty(2000)
    for trial in range(2000):
        mask=np.zeros(n,dtype=np.bool_)
        for g in range(len(group_counts)):
            need=group_counts[g];done=0
            while done<need:
                p=pool[np.random.randint(group_bounds[g],group_bounds[g+1])]
                if not mask[p]:mask[p]=True;done+=1
        busy=np.iinfo(np.int64).min;total=0.;count=0
        for p in range(n):
            if mask[p] and starts[p]>=busy:
                busy=ends[p]
                if np.isfinite(pnl[p]):total+=pnl[p];count+=1
        value=total/count if count else 0.;means[trial]=value
        if value>=actual:beats+=1
    return (beats+1)/2001,np.mean(means)

local=f.index.tz_convert('America/New_York');keys=pd.Series((local.year*10000+local.dayofweek*2000+f.slot.to_numpy()).to_numpy(),index=f.index)
eligible=f.eligible.to_numpy()&post;values=keys.to_numpy();allgroups=np.unique(values[eligible]);pool=np.concatenate([np.flatnonzero(eligible&(values==g)) for g in allgroups]);bounds=np.r_[0,np.cumsum([np.count_nonzero(eligible&(values==g)) for g in allgroups])]
stats=[];base=outcomes['base']
for i,cid in enumerate(ids):
    gc=np.array([np.count_nonzero(signals[i]&post&(values==g)) for g in allgroups],dtype=np.int64)
    closed=pd.concat([base_logs[(cid,'validation')],base_logs[(cid,'final_confirmation')]])
    observed=closed.net_dollars.mean();pv,nullmean=placebo_test(pool,bounds,gc,starts,base[:,0].astype(np.int64),base[:,1],observed,20261007+i)
    stats.append({'candidate':cid,'mean_daily_pnl':daily[:,i].mean(),'weekly_bootstrap_mean_pnl_05':ci[0,i],'weekly_bootstrap_mean_pnl_95':ci[2,i],'mean_trade_pnl':observed,'placebo_mean_trade_pnl':nullmean,'placebo_p_raw':pv,'placebo_draws':2000})
    if (i+1)%10==0:progress('statistical_diagnostics',candidates_complete=i+1,groups=len(groups))
st=pd.DataFrame(stats);order=np.argsort(st.placebo_p_raw.to_numpy());adjusted=np.empty(100);adjusted[order]=np.minimum(1,np.maximum.accumulate(st.placebo_p_raw.to_numpy()[order]*(100-np.arange(100))));st['placebo_p_holm100']=adjusted;st.to_csv(O/'statistical_diagnostics.csv',index=False)
verdicts=[];quarter=pd.DataFrame(quarter_rows)
for cid in ids:
    r=metric[(metric.candidate==cid)&metric.period.isin(['validation','final_confirmation'])];b=r[r.variant.eq('base')];q=quarter[(quarter.candidate==cid)&quarter.period.isin(['validation','final_confirmation'])];s=st[st.candidate.eq(cid)].iloc[0];reasons=[]
    if b.trades.min()<50:reasons.append('insufficient_period_trades')
    if r.unknown_trades.sum()>0:reasons.append('unresolved_execution')
    if (r.net_profit<=0).any() or (r.profit_factor<1.1).any() or r.profit_factor.isna().any():reasons.append('cost_or_fill_fragility')
    if (q.trades<10).any():reasons.append('insufficient_quarter_support')
    if (q.net_profit<=0).any():reasons.append('negative_calendar_quarter')
    if s.weekly_bootstrap_mean_pnl_05<=0:reasons.append('weekly_bootstrap_uncertain')
    if s.placebo_p_raw>.05:reasons.append('not_better_than_random_timing')
    group=next(g for g in grows if g['candidate']==cid)
    verdicts.append({'candidate':cid,'simulation_survivor':not reasons,'failure_reasons':'; '.join(reasons),'group':group['group'],'group_representative':group['is_representative'],'min_post_stress_pf':r.profit_factor.min(),'min_post_stress_net_profit':r.net_profit.min(),'worst_post_minute_dd':r.minute_close_dd.max(),'placebo_p_raw':s.placebo_p_raw,'placebo_p_holm100':s.placebo_p_holm100,'bootstrap_daily_mean_05':s.weekly_bootstrap_mean_pnl_05,'institutional_certified':False,'min_entry_frequency_required':False})
verdict=pd.DataFrame(verdicts);verdict.to_csv(O/'candidate_verdicts.csv',index=False)
survivors=verdict[verdict.simulation_survivor];survivors.to_csv(O/'simulation_survivors.csv',index=False)
progress('historical_simulation_complete',candidates=100,execution_cases=len(metrics),quarter_rows=len(quarter),similarity_groups=len(groups),simulation_survivors=len(survivors),representative_survivors=int((verdict.simulation_survivor&verdict.group_representative).sum()),holm100_diagnostic_pass=int((verdict.simulation_survivor&verdict.placebo_p_holm100.le(.05)).sum()),institutional_certified=0,data_audit=audit)
