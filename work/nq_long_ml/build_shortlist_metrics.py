from pathlib import Path
import json,numpy as np,pandas as pd,quantstats as qs
from systematic_research.fixed_hold import select_nonoverlapping
from systematic_research.trade_report_metrics import trade_kpis

SOURCE=Path('work/nq_long_ml/ten_million_entry_search/final_confirmation')
DATA=Path('data/processed/nq_entry_discovery_202210')
OUT=Path('work/nq_long_ml/shortlist_quant_reports');OUT.mkdir(exist_ok=True)
(OUT/'quantstats').mkdir(exist_ok=True)
shortlist=json.loads((SOURCE/'frozen_shortlist.json').read_text());f=pd.read_parquet(DATA/'opportunities.parquet');bits=np.load(DATA/'atom_bits.npy')
original=pd.read_csv(SOURCE/'results.csv')
rows=[];curves=[];definitions=[]
for x in shortlist:
    cid=x['id'];abc=[q['atom_id'] for q in x['definition']]
    word=bits[abc[0]]&bits[abc[1]]&bits[abc[2]]
    signal=np.unpackbits(word.view(np.uint8),bitorder='little')[:len(f)].astype(bool)
    parts=[]
    for stage in ['in_sample','validation']:
        scope=f[f.split.eq(stage)];signals=signal[f.split.eq(stage).to_numpy()]
        starts=scope.index.asi8
        ends=np.where(scope.exit_available_ns.gt(0),scope.exit_available_ns,pd.DatetimeIndex(scope.planned_exit).asi8).astype(np.int64)
        chosen=select_nonoverlapping(signals,starts,ends)
        t=scope.iloc[chosen].copy()
        t['entry_time']=t.index;t['net_dollars']=t.hypothetical_net_dollars
        t['status']=np.where(t.net_dollars.isna(),'unresolved',np.where(t.mean70_hit.eq(1),'target','timeout'))
        t['exit_time']=pd.to_datetime(ends[chosen],utc=True)
        t['holding_minutes']=(ends[chosen]-starts[chosen])/60e9
        t['gross_points']=(t.net_dollars+25)/20
        t['mae_upper_points']=t.mae_points # Full-window proxy in earlier exports, explicitly labeled.
        t.to_csv(OUT/f'{cid}_{stage}_trades.csv',index=False)
        parts.append(t)
    final=pd.read_csv(SOURCE/f'{cid}_base_trades.csv',parse_dates=['entry_time','exit_time','anchor','planned_exit'])
    parts.append(final)
    combined=pd.concat(parts,ignore_index=True)
    definitions.append({'candidate':cid,'definition':' AND '.join(q['definition'] for q in x['definition']),'families':', '.join(dict.fromkeys(q['family'] for q in x['definition']))})
    for stage,t in [('in_sample',parts[0]),('validation',parts[1]),('final_confirmation',final),('all_periods',combined)]:
        sessions=pd.DatetimeIndex(f.loc[f.split.isin(['in_sample','validation','final_confirmation']) if stage=='all_periods' else f.split.eq(stage),'anchor'].unique()).sort_values()
        t=t.copy()
        t["full_window_mfe_points"]=f.mfe_points.reindex(pd.DatetimeIndex(t.entry_time)).to_numpy(float)
        k,curve=trade_kpis(t,sessions)
        # Verify Sharpe independently against installed QuantStats implementation.
        if np.isfinite(k['sharpe']):assert np.isclose(k['sharpe'],qs.stats.sharpe(curve['return'],periods=252))
        k['candidate']=cid;k['period']=stage;k['variant']='base'
        k['quantstats_daily_profit_factor']=float(qs.stats.profit_factor(curve['return']))
        k['max_minute_drawdown_dollars']=float(original[(original.candidate==cid)&(original.variant=='base')].minute_close_drawdown_dollars.iloc[0]) if stage=='final_confirmation' else None
        k['last_30_minute_entry_fraction']=float((pd.DatetimeIndex(t.entry_time).tz_convert('America/New_York').hour*60+pd.DatetimeIndex(t.entry_time).tz_convert('America/New_York').minute>=930).mean())
        k['close_capped_entry_fraction']=float(((pd.DatetimeIndex(t.planned_exit)-pd.DatetimeIndex(t.entry_time)).total_seconds()<3600).mean())
        rows.append(k)
        curve.index=curve.index.tz_convert('America/New_York').tz_localize(None).normalize()
        curve['candidate']=cid;curve['period']=stage
        curve.reset_index(names='date').to_csv(OUT/f'{cid}_{stage}_daily.csv',index=False)
        if stage in ['in_sample','validation','final_confirmation']:
            curve['net_profit_cumulative']=curve.balance-100000
            curves.extend(curve.reset_index(names='date').assign(date=lambda d:d.date.dt.strftime('%Y-%m-%d')).to_dict('records'))
    print('METRICS',cid,flush=True)
metrics=pd.DataFrame(rows).merge(pd.DataFrame(definitions),on='candidate')
metrics.to_csv(OUT/'all_100_metrics.csv',index=False)
final_metrics=metrics[metrics.period.eq('final_confirmation')].sort_values('sharpe',ascending=False)
final_metrics.to_csv(OUT/'final_period_comparison.csv',index=False)
original.to_csv(OUT/'execution_stress_comparison.csv',index=False)
(OUT/'curves.json').write_text(json.dumps(curves,allow_nan=False))
meta={'reference_capital':100000,'contracts':1,'position_size_compounding':False,'daily_returns':'daily net dollar PNL / prior day account balance; cash sessions without trades included; rf=0,252 sessions/year','trade_profit_factor':'sum positive net trade dollars / abs(sum negative net trade dollars); not daily-return PF','sharpe':'mean daily account return/sample SD * sqrt252; independently matched to QuantStats0.0.81','periods':'IS2022-2024,validation2025,separate2026; embargo sessions excluded; initial history warmup retained as no-trade days in source period calendar','fees':'$25 assumed all-in round trip for base; stress variants separately recorded','drawdown':'main KPI daily account equity; minute-close dollar DD available for 2026 from independently replayed source','earlier_trade_excursions':'earlier exports carry full-hour MAE proxy; final logs contain exit-minute MAE bounds','selection':'same frozen100; no additions or removals','limits':'2026 previously seen; raw signal shortlist selected after10mtrials; correlated variants are not independent strategies; volume-roll provenance unverified; no capital sizing or margin feasibility claim','source':str(SOURCE.resolve()),'data':str(DATA.resolve())}
(OUT/'METHODS.json').write_text(json.dumps(meta,indent=2))
print('TOP\n',final_metrics[['candidate','sharpe','sortino','profit_factor','net_win_rate','target_hit_rate','net_profit','max_daily_drawdown_dollars','trades']].head(8).to_string(index=False))
