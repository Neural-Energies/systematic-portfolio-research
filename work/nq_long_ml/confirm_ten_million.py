"""Freeze supported candidates before replaying the separate 2026 confirmation period."""
from pathlib import Path
import heapq,json,zipfile
import numpy as np
import pandas as pd
from systematic_research.reference_backtest import backtest_reference
from systematic_research.research_partitions import guard_research_sample

R=Path('work/nq_long_ml/ten_million_entry_search');P=Path('data/processed/nq_entry_discovery_202210')
state=json.loads((R/'progress.json').read_text())
if state['definitions_tested']!=10_000_000:raise RuntimeError('Discovery must finish before final confirmation')
guard_research_sample(pd.read_parquet(P/'opportunities.parquet',columns=['split']).index,'ranking')
final=R/'final_confirmation';final.mkdir(exist_ok=True)
heap=[];examined=0
for archive in sorted(R.glob('batch_*.zip')):
    with zipfile.ZipFile(archive) as z:
        for filename in z.namelist():
            if not filename.startswith('passed_validation/'):continue
            x=json.loads(z.read(filename));examined+=1
            a=x['is'];b=x['validation'];i=x['execution_metrics']['is'];v=x['execution_metrics']['validation']
            if a['events']<200 or b['events']<100:continue
            if i['unresolved'] or v['unresolved']:continue
            if i['sessions_with_entry']<30 or v['sessions_with_entry']<20:continue
            if i['observed_net_dollars']<=0 or v['observed_net_dollars']<=0:continue
            pfi=i['gross_positive_net_dollars']/-i['gross_negative_net_dollars'] if i['gross_negative_net_dollars']<0 else 999
            pfv=v['gross_positive_net_dollars']/-v['gross_negative_net_dollars'] if v['gross_negative_net_dollars']<0 else 999
            if min(pfi,pfv)<1.1:continue
            key=(min(a['accuracy'],b['accuracy']),min(pfi,pfv),x['id'])
            if len(heap)<100:heapq.heappush(heap,(key,x))
            elif key>heap[0][0]:heapq.heapreplace(heap,(key,x))
chosen=[x for _,x in sorted(heap,reverse=True)]
# Persist and hash the shortlist BEFORE reading final opportunity labels/returns.
(final/'frozen_shortlist.json').write_text(json.dumps(chosen,indent=2))
f=pd.read_parquet(P/'opportunities.parquet',filters=[('split','==','final_confirmation')])
full_index=pd.read_parquet(P/'opportunities.parquet',columns=['split']).index
m=pd.read_parquet(P/'minutes.parquet');atoms=np.load(P/'atom_bits.npy',mmap_mode='r')
rows=[]
for x in chosen:
    ids=[d['atom_id'] for d in x['definition']]
    words=atoms[ids[0]]&atoms[ids[1]]&atoms[ids[2]]
    signal=pd.Series(np.unpackbits(words.view(np.uint8),bitorder='little')[:len(full_index)].astype(bool),index=full_index).reindex(f.index)
    for variant,cost,delay,penetration in [('base',25.,0,0.),('double_cost',50.,0,0.),('one_minute_latency',25.,1,0.),('one_tick_penetration',25.,0,.25)]:
        trades,marks=backtest_reference(m,f,signal,cost=cost,latency_minutes=delay,penetration=penetration)
        trades.to_csv(final/f'{x["id"]}_{variant}_trades.csv',index=False)
        if trades.empty:
            rows.append({'candidate':x['id'],'variant':variant,'resolved_trades':0});continue
        valid=trades[trades.status.isin(['target','timeout'])]
        if valid.empty:
            rows.append({'candidate':x['id'],'variant':variant,'resolved_trades':0});continue
        losses=valid.net_dollars[valid.net_dollars<0].sum()
        equity=np.r_[0.,marks.equity.to_numpy()] if len(marks) else np.array([0.])
        rows.append({'candidate':x['id'],'variant':variant,'resolved_trades':len(valid),'unknown_trades':int(trades.status.eq('unresolved').sum()),'target_hit_rate':float(valid.status.eq('target').mean()),'net_win_rate':float(valid.net_dollars.gt(0).mean()),'observed_net_dollars':float(valid.net_dollars.sum()),'profit_factor':float(valid.net_dollars[valid.net_dollars>0].sum()/-losses) if losses<0 else None,'minute_close_drawdown_dollars':float(np.max(np.maximum.accumulate(equity)-equity)),'session_coverage':trades[trades.status.ne('already_at_target')].anchor.nunique()/f[f.eligible].anchor.nunique(),'mean_mae_upper_points':float(valid.mae_upper_points.mean())})
    print('CONFIRMED REPLAY',x['id'],flush=True)
result=pd.DataFrame(rows);result.to_csv(final/'results.csv',index=False)
lines=['# Separate-period confirmation replay','',f'Ten million definitions completed; {len(chosen)} candidates frozen from earlier results before this replay. All raw saved candidates remain available; this is a supported, profitable shortlist.','',
'Final period: 2026. These dates have been used in prior research and aggregate inspection, so this is a separate-period replay, not an historically untouched blind holdout. NQ volume-based contract-roll provenance remains unverified. No institutional certification or statistical edge claim follows from hit rate alone.','',
'| Candidate | Execution | Trades | Target hit | Observed net $ | Minute-close DD $ | Session coverage |','|---|---|---:|---:|---:|---:|---:|']
for _,r in result.iterrows():
    if r.get('resolved_trades',0):lines.append(f'| {r.candidate} | {r.variant} | {int(r.resolved_trades)} | {r.target_hit_rate:.1%} | {r.observed_net_dollars:,.0f} | {r.minute_close_drawdown_dollars:,.0f} | {r.session_coverage:.1%} |')
(final/'RESULTS.md').write_text('\n'.join(lines)+'\n')
state['status']='discovery_and_separate_period_replay_complete';state['confirmation_candidates']=len(chosen)
(R/'progress.json').write_text(json.dumps(state,indent=2))
