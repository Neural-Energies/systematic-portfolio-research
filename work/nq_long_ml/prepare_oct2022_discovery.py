from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
import exchange_calendars as xc
from experiment import aggregate
from systematic_research.fixed_hold import fixed_hold_outcomes
from systematic_research.clock_excursion import same_clock_history
from systematic_research.auction_entry import entry_path_diagnostics
from systematic_research.research_partitions import nq_calendar_split

root=Path('data/processed/nq_entry_discovery_202210')
root.mkdir(parents=True,exist_ok=True)
source=Path('../Market Data/NQU6_Export.csv')
raw=pd.read_csv(source,sep='\t')
clock=pd.DatetimeIndex(pd.to_datetime(raw['Date Time'])).tz_localize('America/New_York',ambiguous='raise',nonexistent='raise').tz_convert('UTC')
m=raw.rename(columns={'Open':'open','High':'high','Low':'low','Last':'close','Volume':'volume'})[['open','high','low','close','volume']].set_axis(clock)
if clock.duplicated().any() or not clock.is_monotonic_increasing: raise ValueError('Bad timestamp key')
if not np.isfinite(m.to_numpy()).all(): raise ValueError('Nonfinite source')
if m[['open','high','low','close']].le(0).any().any() or m.volume.lt(0).any(): raise ValueError('Invalid source values')
if (m.high<m[['open','low','close']].max(axis=1)).any() or (m.low>m[['open','high','close']].min(axis=1)).any(): raise ValueError('Invalid OHLC')
m.to_parquet(root/'minutes.parquet')
cal=xc.get_calendar('XNYS',start='2022-10-01',end='2026-09-01')
schedule=cal.schedule.loc['2022-10-12':'2026-08-20']
schedule.to_parquet(root/'rth_schedule.parquet')
entries=[];exits=[];anchors=[]
for _,session in schedule.iterrows():
    grid=pd.date_range(session.open,session.close,freq='5min',inclusive='left')
    entries.extend(grid); exits.extend([min(t+pd.Timedelta(hours=1),session.close) for t in grid]);anchors.extend([session.open]*len(grid))
entries=pd.DatetimeIndex(entries);exits=pd.DatetimeIndex(exits)
f=fixed_hold_outcomes(m,entries,exits)
f['anchor']=anchors
local=entries.tz_convert('America/New_York'); f['slot']=local.hour*60+local.minute
f['previous_hour_stream_open']=f.groupby((f.slot-570)%60).entry_open.shift()
f['opening_up']=f.entry_open.gt(f.previous_hour_stream_open)
f['excursion']=f.mfe_points
sessions=pd.DatetimeIndex(schedule.open)
h=same_clock_history(f,sessions)
h.to_parquet(root/'history_iqr.parquet')
f['target_points']=np.ceil(h.mean70*4)/4
f['percentile70_points']=np.ceil(h.q70*4)/4
f['full_range70_points']=np.nan
fr=f.copy();fr['excursion']=f.mfe_points+f.mae_points
hr=same_clock_history(fr,sessions)
f['full_range70_points']=np.ceil(hr.mean70*4)/4
f['retained_history_samples']=h.retained_samples
f['eligible']=f.opening_up&f.target_points.gt(0)&f.entry_open.notna()
f['split']=nq_calendar_split(f.index)
# Five complete cash sessions after each boundary are embargoed.
embargo=[]
for year in [2025,2026]:
    rows=schedule[schedule.index.year==year].head(5)
    embargo.extend(rows.open.tolist())
f.loc[f.anchor.isin(embargo),'split']='embargo'
for name,column in [('mean70','target_points'),('percentile70','percentile70_points'),('full_range70','full_range70_points')]:
    usable=f.eligible&f[column].gt(0)
    diag=entry_path_diagnostics(m,f.index[usable],pd.DatetimeIndex(f.loc[usable,'planned_exit']),f.loc[usable,column].to_numpy(float))
    f[f'{name}_hit']=np.nan
    f.loc[usable,f'{name}_hit']=diag.reached_up.to_numpy()
    if name=='mean70':
        # Independent minute replay for hypothetical exit endpoint; preserve unknown coverage.
        exit_ns=np.full(len(f),0,dtype=np.int64); pnl=np.full(len(f),np.nan)
        for j in np.flatnonzero(usable.to_numpy()):
            r=f.iloc[j]; path=m.iloc[m.index.searchsorted(f.index[j]):m.index.searchsorted(r.planned_exit)]
            target=r.entry_open+r.target_points
            touched=np.flatnonzero(path.high.to_numpy()>=target)
            end=path.index[touched[0]]+pd.Timedelta(minutes=1) if len(touched) else r.planned_exit
            held=path.loc[path.index<end]
            complete=len(held)==int((end-f.index[j]).total_seconds()/60)
            if not complete: continue
            price=target if len(touched) else held.close.iloc[-1]
            exit_ns[j]=end.value; pnl[j]=(price-r.entry_open)*20-25
        f['exit_available_ns']=exit_ns;f['hypothetical_net_dollars']=pnl
f.to_parquet(root/'opportunities.parquet')
bars=aggregate(m,5); bars.to_parquet(root/'completed_5min.parquet')
sha=hashlib.sha256()
with source.open('rb') as stream:
    for chunk in iter(lambda:stream.read(1048576),b''):sha.update(chunk)
quality={'source':str(source.resolve()),'sha256':sha.hexdigest(),'rows':len(m),'first_utc':str(clock.min()),'last_utc':str(clock.max()),'source_timezone':'America/New_York, human confirmed','calendar':'XNYS cash-session RTH schedule, half-day close respected; excludes cash holidays including Jan 9 2025','price_kind':'export labeled NQU6 across history, contract/volume-roll provenance not yet verified','split':'IS through 2024; validation 2025; final confirmation 2026, five-session boundary embargo','bar_semantics':'assumed start-of-minute export timestamps; verify exporter before live claim','targets':'past 20 cash sessions same clock, opening_up only, minimum five samples, 1.5 IQR past only','counts':f.groupby('split').size().to_dict(),'eligible_counts':f[f.eligible].groupby('split').size().to_dict(),'unresolved_mean70':int(f.loc[f.eligible,'mean70_hit'].isna().sum())}
(root/'import_manifest.json').write_text(json.dumps(quality,indent=2))
print(json.dumps(quality,indent=2),flush=True)
