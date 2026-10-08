"""Frozen-entry, observed-quote continuation pilot; not a calendar-certified backtest."""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
from pathlib import Path
import hashlib,json,shutil
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from numba import njit

P=Path('data/processed/nq_entry_discovery_202210')
S=Path('work/nq_long_ml/ten_million_entry_search/final_confirmation')
O=Path('work/nq_long_ml/extended_hold_pilot');O.mkdir(exist_ok=True)
raw=(S/'frozen_shortlist.json').read_bytes(); digest=hashlib.sha256(raw).hexdigest()
lock=O/'LOCKED_ENTRIES.json'
if lock.exists() and hashlib.sha256(lock.read_bytes()).hexdigest()!=digest:raise RuntimeError('Existing lock differs')
if not lock.exists():shutil.copyfile(S/'frozen_shortlist.json',lock)
short=json.loads(raw)
assert len(short)==100 and len({x['id'] for x in short})==100
contract={'entry_sha256':digest,'count':100,'entry_rules':'Original three atoms, original eligibility, RTH only, one contract and one position per candidate; no changes to entry thresholds. Longer holds suppress later entries.',
 'exit_only_research':True,'overnight':'Retain position through Globex; no exit forced at 16:00. No stop added.',
 'maximums':'Test both 120 elapsed hours and five cash trading sessions counting entry session as session one; latter cap at fifth session cash close. Fill at first observed executable quote at/after deadline.',
 'baseline':'Original target/hour exit plus fixed 60m,4h,24h and 120h holds. Continuation models first evaluated after60m, then every5m.',
 'cost':25,'point_value':20,'training':'2022-2024 only, all-session completed five-minute features; purge forward labels crossing2025. Deterministic training thinning every15m. Validation2025 used to choose exit variant before separate2026 replay.',
 'model_target':'Future open minus current executable open, divided by causal ATR. Horizon60m or240m. Hold if estimated remaining displacement exceeds threshold; regression values are not calibrated probabilities.',
 'limits':['Previously seen2026','Ten-million-entry selection and correlated candidates','Volume-roll contract provenance unverified','Observed-quote preliminary replay; exchange closure and missing-quote calendar not certified','No overnight roll execution, financing, changing spread or margin simulation','No guarantee of at least one entry per session'],
 'research_sources':['https://arxiv.org/abs/math/0408276','https://epubs.siam.org/doi/10.1137/21M1460648'],
 'method_note':'Rolling horizon regression is an initial continuation screen, not a claimed implementation of dynamic-programming optimal stopping.'}
(O/'PROTOCOL.json').write_text(json.dumps(contract,indent=2))
(O/'STATUS.json').write_text(json.dumps({'status':'building_features','locked100':True,'sha256':digest}))
print('LOCKED',len(short),digest,flush=True)
b=pd.read_parquet(P/'completed_5min.parquet');m=pd.read_parquet(P/'minutes.parquet');f=pd.read_parquet(P/'opportunities.parquet')
t=pd.DatetimeIndex(b.available_at); ns=t.asi8
pos=m.index.get_indexer(t); available=pos>=0
price=np.full(len(b),np.nan);price[available]=m.open.to_numpy()[pos[available]]
c=b.close;v=b.volume
tr=pd.concat([b.high-b.low,(b.high-c.shift()).abs(),(b.low-c.shift()).abs()],axis=1).max(axis=1)
atr=tr.ewm(alpha=1/14,adjust=False,min_periods=48).mean().clip(lower=.25)
x=pd.DataFrame(index=b.index)
for w in [1,3,6,12,24,48,144,288]:
    x[f'momentum_{w}']=(c-c.shift(w))/atr
    x[f'ema_{w}']=(c-c.ewm(span=max(w,2),adjust=False).mean())/atr
    x[f'volume_{w}']=v/v.shift().rolling(max(w,3)).mean().replace(0,np.nan)
    x[f'volatility_{w}']=c.diff().rolling(max(w,3)).std()/atr
    x[f'location_{w}']=(c-b.low.rolling(max(w,3)).min())/(b.high.rolling(max(w,3)).max()-b.low.rolling(max(w,3)).min()).replace(0,np.nan)
x['body']=(c-b.open)/atr;x['range']=(b.high-b.low)/atr
x['quote_gap_minutes']=pd.Series(ns,index=b.index).diff()/60e9
local=t.tz_convert('America/New_York');clock=local.hour*60+local.minute
x['clock_sin']=np.sin(clock*2*np.pi/1440);x['clock_cos']=np.cos(clock*2*np.pi/1440);x['weekday']=local.dayofweek
x['rth']=((clock>=570)&(clock<960)).astype(int)
# ES features use only the last completed, exactly aligned five-minute bar.
espaths=[p for folder in ['development_minute_returns','sealed_holdout_minute_returns'] for p in Path('data/processed/databento_research',folder,'symbol=ES').glob('*.parquet')]
if espaths:
    es=pd.concat([pd.read_parquet(p,columns=['timestamp_utc','open','close']) for p in espaths]).set_index('timestamp_utc').sort_index();es=es[~es.index.duplicated()]
    esclose=es.close.reindex(t-pd.Timedelta(minutes=1));esclose.index=b.index
    for w in [1,3,12,48]:x[f'es_return_{w}']=np.log(esclose/esclose.shift(w))
x=x.replace([np.inf,-np.inf],np.nan);X=x.to_numpy(float)
atrv=atr.to_numpy();train_end=pd.Timestamp('2025-01-01',tz='UTC').value
year=local.year.to_numpy();training=(ns<train_end)&available&np.isfinite(atrv)&(np.arange(len(b))%3==0)
validation=(year==2025)&available;final=(year==2026)&available
predictions={};diagnostics=[]
for horizon in [60,240]:
    target_time=ns+int(horizon*60e9);j=np.searchsorted(ns,target_time);safe=np.minimum(j,len(b)-1)
    exact=(j<len(b))&(ns[safe]-target_time<=int(5*60e9))&available[safe]
    y=(price[safe]-price)/atrv
    fit=training&exact&(ns[safe]<train_end)&np.isfinite(y)
    models={'ridge':make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),Ridge(alpha=100)),
            'boosted':HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,min_samples_leaf=150,l2_regularization=10,early_stopping=False,random_state=20261007),
            'extra_trees':make_pipeline(SimpleImputer(add_indicator=True),ExtraTreesRegressor(n_estimators=48,max_depth=8,min_samples_leaf=100,n_jobs=2,random_state=20261007))}
    for name,model in models.items():
        print('FIT',name,horizon,int(fit.sum()),flush=True)
        cache=O/f'prediction_{name}_{horizon}.npy'
        if cache.exists():
            pred=np.load(cache)
            if len(pred)!=len(b):raise RuntimeError('Prediction cache length mismatch')
        else:
            model.fit(X[fit],y[fit]);pred=model.predict(X);pred[~available]=np.nan
            np.save(cache,pred)
        predictions[(name,horizon)]=pred
        for stage,mask in [('validation',validation),('final_confirmation',final)]:
            use=mask&exact&np.isfinite(y)
            actual=(price[safe]-price)[use];estimate=(pred*atrv)[use]
            baseline=np.zeros(len(actual));mse=mean_squared_error(actual,estimate)
            diagnostics.append({'model':name,'horizon_minutes':horizon,'period':stage,'samples':len(actual),'mae_points':mean_absolute_error(actual,estimate),'mse_points2':mse,'rmse_points':np.sqrt(mse),'zero_change_rmse':np.sqrt(mean_squared_error(actual,baseline)),'direction_accuracy':float(np.mean((estimate>0)==(actual>0)))})
pd.DataFrame(diagnostics).to_csv(O/'regression_diagnostics.csv',index=False)
print('REGRESSION COMPLETE',flush=True)

@njit
def replay(entry_indices,deadlines,ns,price,closes,lows,highs,score,minimum_ns,period_end):
    out=np.empty((len(entry_indices),10));n=0;busy=-1;realized=0.;peak=0.;dd=0.
    for z in range(len(entry_indices)):
        i=entry_indices[z]
        if i<0 or i<busy or not np.isfinite(price[i]):continue
        cap=deadlines[z]
        k=i+1
        while k<len(ns) and (not np.isfinite(price[k]) or ns[k]<cap and (ns[k]-ns[i]<minimum_ns or not np.isfinite(score[k]) or score[k]>0)):
            k+=1
        if k>=len(ns) or ns[k]>=period_end or ns[k]>cap:
            # Do not turn a missing/closed deadline quote into an overnight fill.
            # The selected trade is unresolved, not retroactively skipped.
            out[n,:]=np.array([i,-1.,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]);n+=1;busy=k;continue
        pnl=(price[k]-price[i])*20-25;mae=0.;mfe=0.;gaps=0
        # Bar q becomes known at ns[q], so i+1 is the first held five-minute bar.
        # Bar i ended before entry and must never enter MAE/MFE or position marks.
        for q in range(i+1,k+1):
            mae=max(mae,price[i]-lows[q]);mfe=max(mfe,highs[q]-price[i])
            equity=realized+(closes[q]-price[i])*20-25;peak=max(peak,equity);dd=max(dd,peak-equity)
            if q>i and ns[q]-ns[q-1]>int(5*60e9):gaps+=1
        realized+=pnl;peak=max(peak,realized);dd=max(dd,peak-realized)
        out[n,:]=np.array([i,k,pnl,(ns[k]-ns[i])/60e9,mae,mfe,gaps,dd,price[i],price[k]]);n+=1;busy=k
    return out[:n]

bits=np.load(P/'atom_bits.npy',mmap_mode='r');schedule=pd.read_parquet(P/'rth_schedule.parquet')
session_opens=pd.DatetimeIndex(schedule.open).asi8;session_closes=pd.DatetimeIndex(schedule.close).asi8
variants={f'fixed_{h}h':(np.zeros(len(b)),h,h*60) for h in [1,4,24,120]}
for name in ['ridge','boosted','extra_trees']:
    for mode in ['60m','multi']:
        base=predictions[(name,60)] if mode=='60m' else np.maximum(predictions[(name,60)],predictions[(name,240)])
        variants[f'{name}_{mode}_120h']=(base,120,60)
        variants[f'{name}_{mode}_5sessions']=(base,None,60)
rows=[];chosen_variant=None
baseline=pd.read_csv('work/nq_long_ml/shortlist_quant_reports/all_100_metrics.csv')
for stage in ['validation','final_confirmation']:
    if stage=='final_confirmation':
        val=pd.DataFrame(rows);ml=val[val.variant.str.contains('ridge|boosted|extra_trees')]
        ranking=ml.groupby('variant').agg(median_expectancy=('expectancy','median'),median_pf=('profit_factor','median'),positive_candidates=('net_dollars',lambda a:int((a>0).sum())))
        ranking=ranking.sort_values(['median_expectancy','median_pf'],ascending=False)
        chosen_variant=str(ranking.index[0]);ranking.to_csv(O/'validation_variant_ranking.csv')
        (O/'FROZEN_EXIT_SELECTION.json').write_text(json.dumps({'selected_variant':chosen_variant,'rule':'Maximum median per-candidate validation expectancy; PF tie-break. No2026 optimization.','candidate_count':100},indent=2))
    for item in short:
        a=[q['atom_id'] for q in item['definition']];words=bits[a[0]]&bits[a[1]]&bits[a[2]]
        signal=np.unpackbits(words.view(np.uint8),bitorder='little')[:len(f)].astype(bool)&f.eligible.to_numpy()&f.split.eq(stage).to_numpy()
        entries=f.index[signal];entry_indices=pd.Index(t).get_indexer(entries)
        for variant,(score,hours,minimum) in variants.items():
            if stage=='final_confirmation' and variant!=chosen_variant and not variant.startswith('fixed'):continue
            deadline=entries.asi8+int(hours*3600e9) if hours else session_closes[np.minimum(np.searchsorted(session_opens,entries.asi8,side='right')-1+4,len(session_closes)-1)]
            # Keep extended outcomes within their own study period.
            period_end=pd.Timestamp('2026-01-01' if stage=='validation' else '2026-08-20 16:00',tz='America/New_York').value
            deadline=np.minimum(deadline,period_end)
            output=replay(entry_indices,deadline,ns,price,b.close.to_numpy(),b.low.to_numpy(),b.high.to_numpy(),score,int(minimum*60e9),period_end)
            trades=pd.DataFrame(output,columns=['entry_bar','exit_bar','net_dollars','hold_minutes','mae_points','mfe_points','observed_gap_count','running_mark_drawdown','entry_price','exit_price'])
            valid=trades.net_dollars.notna();resolved=trades[valid];loss=resolved.loc[resolved.net_dollars<0,'net_dollars'].sum()
            used=resolved.entry_bar.astype(int).to_numpy();days=t[used].tz_convert('America/New_York').normalize()
            row={'candidate':item['id'],'period':stage,'variant':variant,'trades':len(resolved),'unresolved':int((~valid).sum()),'net_dollars':resolved.net_dollars.sum(),'complete_net_dollars':resolved.net_dollars.sum() if valid.all() else np.nan,'profit_factor':resolved.loc[resolved.net_dollars>0,'net_dollars'].sum()/-loss if loss<0 else np.nan,'win_rate':float(resolved.net_dollars.gt(0).mean()),'expectancy':resolved.net_dollars.mean(),'mean_hold_hours':resolved.hold_minutes.mean()/60,'max_hold_hours':resolved.hold_minutes.max()/60,'five_minute_mark_dd':resolved.running_mark_drawdown.max() if valid.all() else np.nan,'mean_mae_points':resolved.mae_points.mean(),'mean_mfe_points':resolved.mfe_points.mean(),'gap_path_fraction':float(resolved.observed_gap_count.gt(0).mean()),'sessions_with_entry':days.nunique()}
            rows.append(row)
            if variant==chosen_variant or item['id']=='NQ_000926441771':
                trades['entry_time']=[t[int(i)] for i in trades.entry_bar]
                trades['exit_time']=[t[int(i)] if i>=0 else pd.NaT for i in trades.exit_bar]
                trades.to_csv(O/f'{item["id"]}_{stage}_{variant}_trades.csv',index=False)
        if len(rows)%100==0:print('REPLAY',stage,item['id'],len(rows),flush=True)
        pd.DataFrame(rows).to_csv(O/'exit_comparison.csv',index=False)
    print('PERIOD COMPLETE',stage,flush=True)
result=pd.DataFrame(rows)
result=result.merge(baseline[['candidate','period','net_profit','profit_factor','net_win_rate','trades','session_coverage']],on=['candidate','period'],suffixes=('','_original'))
result.to_csv(O/'exit_comparison.csv',index=False)
selected=result[(result.period=='final_confirmation')&(result.variant==chosen_variant)]
summary={'status':'preliminary_observed_quote_pilot_complete','locked100':True,'selected_exit':chosen_variant,'positive_candidates':int(selected.net_dollars.gt(0).sum()),'profit_above_original_candidates':int(selected.net_dollars.gt(selected.net_profit).sum()),'win_rate_above_original_candidates':int(selected.win_rate.gt(selected.net_win_rate).sum()),'median_hold_hours':float(selected.mean_hold_hours.median()),'median_net_dollars':float(selected.net_dollars.median()),'median_profit_factor':float(selected.profit_factor.median()),'unresolved':int(selected.unresolved.sum()),'median_gap_path_fraction':float(selected.gap_path_fraction.median()),'limitations':contract['limits']}
(O/'STATUS.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
