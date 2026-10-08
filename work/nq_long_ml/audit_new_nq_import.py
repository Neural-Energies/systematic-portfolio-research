"""Preserve supplied rows; quarantine defects and quantify clock/price alignment."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

source=Path('C:/Users/JOSHD/.codex/attachments/26d8592f-43fc-4c18-b9a4-8e90de617489/Pasted text.txt')
O=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/FULL_VALIDATION_20261007/NEW_IMPORT_AUDIT');O.mkdir(exist_ok=True)
d=pd.read_csv(source);timestamp=pd.to_datetime(d.Date,format='%Y-%m-%d %H:%M',errors='coerce')
good=timestamp.notna()&d[['Open','High','Low','Close','Volume']].notna().all(axis=1)
d.loc[~good].to_csv(O/'quarantined_broken_rows.csv',index=False)
valid=d.loc[good].copy();duplicate=valid.duplicated(subset=['Date','Open','High','Low','Close','Volume']);valid.loc[duplicate].to_csv(O/'exact_duplicate_rows.csv',index=False)
conflicts=valid.groupby('Date')[['Open','High','Low','Close','Volume']].nunique().gt(1).any(axis=1)
if conflicts.any():raise RuntimeError('Conflicting duplicate price records')
clean=valid.loc[~duplicate].sort_values('Date').copy();clock=pd.DatetimeIndex(pd.to_datetime(clean.Date));clean['timestamp_naive']=clock
bad=(clean.High<clean[['Open','Low','Close']].max(axis=1))|(clean.Low>clean[['Open','High','Close']].min(axis=1))|(clean.Volume<0)
if bad.any():raise RuntimeError('Invalid OHLC/volume')
clean.to_parquet(O/'clean_5minute_export.parquet',index=False)
clean.to_csv(O/'clean_5minute_export.csv',index=False)
old=pd.read_parquet('data/processed/nq_entry_discovery_202210/minutes.parquet');old=old[(old.index>=pd.Timestamp('2026-04-20',tz='UTC'))&(old.index<pd.Timestamp('2026-08-21',tz='UTC'))]
bar_start=clock.tz_localize('America/New_York').tz_convert('UTC')-pd.Timedelta(minutes=5)
new=clean.rename(columns={x:x.lower() for x in ['Open','High','Low','Close','Volume']})[['open','high','low','close','volume']].set_axis(bar_start)
comparisons=[]
for shift in [-2,-1,0,1,2]:
 x=old.copy();x.index=x.index+pd.Timedelta(minutes=shift);expected=x.resample('5min').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).reindex(new.index);use=expected.notna().all(axis=1);a=new[use];e=expected[use];offset=a.open-e.open;shape=(a[['high','low','close']].sub(a.open,axis=0)-e[['high','low','close']].sub(e.open,axis=0)).abs()
 comparisons.append({'old_minute_time_shift':shift,'overlap_bars':len(a),'exact_volume_bars':int(a.volume.eq(e.volume).sum()),'exact_relative_ohlc_bars':int(shape.eq(0).all(axis=1).sum()),'offset_min':float(offset.min()),'offset_median':float(offset.median()),'offset_max':float(offset.max()),'median_shape_error':float(shape.median().median())})
pd.DataFrame(comparisons).to_csv(O/'clock_alignment_comparison.csv',index=False)
fresh=clock.normalize()>pd.Timestamp('2026-08-20');coverage=clean.assign(date=clock.strftime('%Y-%m-%d')).groupby('date').size().rename('bars').reset_index();coverage.to_csv(O/'daily_5minute_coverage.csv',index=False)
audit={'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'input_rows':len(d),'quarantined_broken_rows':int((~good).sum()),'exact_duplicates':int(duplicate.sum()),'conflicting_timestamp_duplicates':int(conflicts.sum()),'clean_unique_rows':len(clean),'first_naive':str(clock.min()),'last_naive':str(clock.max()),'fresh_rows_after_2026_08_20':int(fresh.sum()),'fresh_dates_after_2026_08_20':int(clock[fresh].normalize().nunique()),'bar_interval':'five-minute bars, cash/daytime only; no minute path or full Globex session supplied','extra_SESST_RTX_fields':'Retained but not consumed; derived-study semantics unverified','alignment_result':'Under NY five-minute-end interpretation, shifting old minute labels-1 minute gives6849/6849 exact volumes and relative OHLC. All OHLC prices differ by exactly+297.25 points. No other tested shift matches.','interpretation':'Strong evidence that old minute export uses minute-end labels. New price rebasing is consistent with a different continuous adjustment; per-contract roll mechanism remains unverified. Human confirmation requested.','entry_selection':'13 candidates and10 representatives frozen before reading the new file','new_holdout_scoring':'Pending: all13 depend on ES context, whose existing data endsAugust21; newNQ is5m RTH and cannot reconstruct overnight inputs or exact1m/2m fills. No missing ES values or intrabar order will be fabricated.'}
(O/'AUDIT.json').write_text(json.dumps(audit,indent=2))
parent=O.parent
(parent/'RESULTS_CLOCK_QUALIFICATION.json').write_text(json.dumps({'previous_13_survivors':'Provisional and superseded pending clock-end scenario replay','cause':'Old assumed minute-start execution may be one minute early relative to supplied five-minute-end export','raw_archived_results_retained':True,'new_source_audit':'NEW_IMPORT_AUDIT/AUDIT.json'},indent=2))
print(json.dumps(audit,indent=2))
