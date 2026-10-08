"""Combined post-training risk, regime/concentration diagnostics and report evidence."""
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from systematic_research.execution_stress import minute_outcomes,schedule_events,marked_drawdowns
from systematic_research.trade_report_metrics import trade_kpis

L=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007');O=L/'FULL_VALIDATION_20261007'/'MINUTE_END_SCENARIO';P=Path('data/processed/nq_entry_discovery_202210_minute_end_scenario')
protocol=json.loads((O/'PROTOCOL.json').read_text());short=json.loads((L/'LOCKED_ENTRIES.json').read_text());verdict=pd.read_csv(O/'candidate_verdicts.csv');survivors=set(verdict[verdict.simulation_survivor].candidate)
m=pd.read_parquet(P/'minutes.parquet');f=pd.read_parquet(P/'opportunities.parquet');bits=np.load(P/'atom_bits.npy',mmap_mode='r');schedule=pd.read_parquet(P/'rth_schedule.parquet')
clock=m.index.asi8;ohlc=m[['open','high','low','close']].to_numpy(float);starts=f.index.asi8;ends=pd.DatetimeIndex(f.planned_exit).asi8;targets=(f.entry_open+f.target_points).to_numpy(float)
post=f.split.isin(['validation','final_confirmation']).to_numpy();sessions=pd.DatetimeIndex(f.loc[post,'anchor'].unique()).sort_values();signals=[]
for item in short:
 a=[q['atom_id'] for q in item['definition']];words=bits[a[0]]&bits[a[1]]&bits[a[2]];signals.append(np.unpackbits(words.view(np.uint8),bitorder='little')[:len(f)].astype(bool))
outcomes={name:minute_outcomes(clock,ohlc,starts,ends,targets,*args) for name,args in protocol['scenarios'].items()}
daily=[]
for _,s in schedule.iterrows():
 path=m[(m.index>=s.open)&(m.index<s.close)];daily.append({'anchor':s.open,'close':path.close.iloc[-1],'range':path.high.max()-path.low.min()})
d=pd.DataFrame(daily).set_index('anchor');d['prior_20session_return']=d.close.shift(1)/d.close.shift(21)-1;d['past_range20']=d.range.shift(1).rolling(20).mean();cut=d.loc[d.index.year<=2024,'past_range20'].median();d['regime']=np.where(d.prior_20session_return>=0,'prior20_up','prior20_down');d['volatility']=np.where(d.past_range20>cut,'past_range_above_IS_median','past_range_below_IS_median')
metrics=[];curves=[];regimes=[];concentration=[]
for k,item in enumerate(short):
 cid=item['id']
 for name,args in protocol['scenarios'].items():
  a=outcomes[name];chosen=schedule_events(signals[k]&post,starts,a);t=f.iloc[chosen][['anchor','target_points','planned_exit']].copy();t['entry_time']=f.index[chosen]+pd.Timedelta(minutes=args[0]);t['entry_price']=a[chosen,4];t['exit_price']=a[chosen,5];t['exit_time']=pd.to_datetime(a[chosen,0].astype(np.int64),utc=True);t['net_dollars']=a[chosen,1];t['status']=np.where(a[chosen,3]==0,'unresolved',np.where(a[chosen,2]==1,'target','timeout'));t['mae_upper_points']=a[chosen,6];t['holding_minutes']=a[chosen,8]
  kp,curve=trade_kpis(t,sessions);dd,lowdd=marked_drawdowns(clock,ohlc,starts,a,chosen,args[0],args[1]);kp.update({'candidate':cid,'period':'post_training_combined','variant':name,'minute_close_dd':dd,'minute_low_dd_bound':lowdd});metrics.append(kp)
  if name in ['base','combined_stress']:
   curve['candidate']=cid;curve['variant']=name;curve['net_profit_cumulative']=curve.balance-100000;curve['date']=curve.index.tz_convert('America/New_York').strftime('%Y-%m-%d');curves.extend(curve.reset_index(drop=True).to_dict('records'));curve.to_csv(O/f'{cid}_{name}_combined_daily.csv',index_label='cash_session_open')
   if cid in survivors:t.to_csv(O/f'{cid}_{name}_combined_trades.csv',index=False)
  if name=='base':
   t['regime']=d.regime.reindex(pd.DatetimeIndex(t.anchor)).to_numpy();t['volatility']=d.volatility.reindex(pd.DatetimeIndex(t.anchor)).to_numpy()
   for field in ['regime','volatility']:
    for label,g in t.groupby(field):
     loss=-g.net_dollars[g.net_dollars<0].sum();regimes.append({'candidate':cid,'cut':field,'regime':label,'trades':len(g),'net_profit':g.net_dollars.sum(),'profit_factor':g.net_dollars[g.net_dollars>0].sum()/loss if loss>0 else np.nan,'net_win_rate':g.net_dollars.gt(0).mean()})
   top=t.net_dollars.nlargest(5).sum();concentration.append({'candidate':cid,'net_profit':t.net_dollars.sum(),'net_after_removing_top5_trades':t.net_dollars.sum()-top,'top5_winners_share_of_gross_wins':top/t.net_dollars[t.net_dollars>0].sum()})
 print('COMBINED',cid,flush=True)
metric=pd.DataFrame(metrics);metric.to_csv(O/'combined_execution_metrics.csv',index=False);pd.DataFrame(regimes).to_csv(O/'regime_results.csv',index=False);pd.DataFrame(concentration).to_csv(O/'profit_concentration.csv',index=False)
pd.DataFrame(curves).to_json(O/'combined_curves.json',orient='records')
allmetrics=pd.concat([pd.read_csv(O/'all_execution_metrics.csv'),metric],ignore_index=True);allmetrics.to_csv(O/'all_periods_execution_metrics.csv',index=False)
base=metric[metric.variant.eq('base')];summary=verdict.merge(base[['candidate','trades','net_profit','profit_factor','net_win_rate','sharpe','minute_close_dd','minute_low_dd_bound','worst_trade','mean_mae_points']],on='candidate').merge(pd.DataFrame(concentration).drop(columns='net_profit'),on='candidate');summary['failure_reasons']=summary.failure_reasons.fillna('Passed declared simulation gates');summary.to_csv(O/'candidate_comparison.csv',index=False)
