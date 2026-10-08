from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from experiment import aggregate
from rth_named_setups import indicators
from systematic_research.auction_entry import profile_histogram,profile_levels
from systematic_research.research_partitions import guard_research_sample

R=Path('data/processed/nq_entry_discovery_202210')
f=pd.read_parquet(R/'opportunities.parquet');m=pd.read_parquet(R/'minutes.parquet');b=pd.read_parquet(R/'completed_5min.parquet')
c,o,h,l,v=[b[x].astype(float) for x in ['close','open','high','low','volume']]
atr,adx,ema=indicators(b);atr=atr.replace(0,np.nan)
ret=np.log(c/c.shift()); span=(h-l).replace(0,np.nan)
features={}; families={}
def add(name,value,family):
    features[name]=pd.Series(value.to_numpy(),index=pd.DatetimeIndex(b.available_at)).reindex(f.index)
    families[name]=family
for w in [2,3,6,12,24,48,96,144]:
    add(f'momentum_atr_{w}',(c-c.shift(w))/atr,'momentum')
    add(f'ema_distance_{w}',(c-c.ewm(span=w,min_periods=w,adjust=False).mean())/atr,'trend')
    add(f'price_z_{w}',(c-c.rolling(w).mean())/c.rolling(w).std().replace(0,np.nan),'reversion')
    add(f'volume_ratio_{w}',v/v.shift().rolling(w).mean().replace(0,np.nan),'volume')
    add(f'high_breakout_{w}',(c-h.shift().rolling(w).max())/atr,'breakout')
    add(f'low_distance_{w}',(c-l.shift().rolling(w).min())/atr,'support')
    add(f'stochastic_{w}',(c-l.rolling(w).min())/(h.rolling(w).max()-l.rolling(w).min()).replace(0,np.nan),'reversion')
    add(f'efficiency_{w}',(c-c.shift(w)).abs()/c.diff().abs().rolling(w).sum().replace(0,np.nan),'trend')
    add(f'volatility_ratio_{w}',ret.rolling(w).std()/ret.shift().rolling(48).std().replace(0,np.nan),'volatility')
    add(f'volume_signed_{w}',(np.sign(c-o)*v).rolling(w).sum()/v.rolling(w).sum().replace(0,np.nan),'volume')
add('adx14',adx,'trend');add('candle_body_atr',(c-o)/atr,'price_action')
add('candle_close_location',(c-l)/span,'price_action')
add('lower_wick',(pd.concat([c,o],axis=1).min(axis=1)-l)/span,'price_action')
add('upper_wick',(h-pd.concat([c,o],axis=1).max(axis=1))/span,'price_action')
add('bar_range_atr',span/atr,'volatility')
add('return_volatility',ret/ret.shift().rolling(48).std().replace(0,np.nan),'momentum')
for w in [3,6,12,24]:
    weight=np.arange(w)-(w-1)/2
    slope=c.rolling(w).apply(lambda x:float(x@weight/(weight@weight)),raw=True)
    add(f'ols_slope_{w}',slope/atr,'trend')
local=b.index.tz_convert('America/New_York'); slots=local.hour*60+local.minute;days=local.normalize()
rth=(slots>=570)&(slots<960)
rvol=v.where(rth,0)
vwap=(((h+l+c)/3*rvol).groupby(days).cumsum()/rvol.groupby(days).cumsum().replace(0,np.nan))
add('rth_vwap_distance',(c-vwap)/atr,'auction')
add('rth_vwap_reclaim',(c>vwap)&(c.shift()<=vwap.shift()),'auction')
# Profile uses only previous completed scheduled cash session, close allocation approximation.
sched=pd.read_parquet(R/'rth_schedule.parquet'); profiles=[]
for _,s in sched.iterrows():
    start=m.index.searchsorted(s.open);end=m.index.searchsorted(s.close)
    path=m.iloc[start:end]
    if len(path):
        levels=profile_levels(profile_histogram(path,2.,'close'),2.)
    else: levels=(np.nan,np.nan,np.nan)
    profiles.append({'date':s.open.tz_convert('America/New_York').normalize(),'poc':levels[0],'val':levels[1],'vah':levels[2]})
prof=pd.DataFrame(profiles).set_index('date').shift()
print('PROFILE COLUMNS',prof.columns.tolist(),flush=True)
for column in ['poc','val','vah']:
    if column in prof:
        level=pd.Series(days.map(prof[column]),index=b.index)
        add(f'prior_{column}_distance',(c-level)/atr,'auction')
        add(f'prior_{column}_reclaim',(c>level)&(c.shift()<=level),'auction')
for horizon in [15,60]:
    higher=aggregate(m,horizon)
    for w in [3,6,12]:
        value=(higher.close-higher.close.shift(w))/higher.close.pct_change().rolling(24).std().replace(0,np.nan)/higher.close
        known=pd.Series(value.to_numpy(),index=pd.DatetimeIndex(higher.available_at))
        # Last completed higher bar, with explicit freshness limit.
        features[f'completed_{horizon}m_momentum_{w}']=known.reindex(f.index,method='ffill',tolerance=pd.Timedelta(minutes=horizon-1))
        families[f'completed_{horizon}m_momentum_{w}']='mixed_frequency'
# ES context is vendor UTC data; no imputation outside observed coverage.
espaths=[p for folder in ['development_minute_returns','sealed_holdout_minute_returns'] for p in Path('data/processed/databento_research',folder,'symbol=ES').glob('*.parquet')]
if espaths:
    es=pd.concat([pd.read_parquet(p,columns=['timestamp_utc','open','high','low','close','volume']) for p in espaths]).set_index('timestamp_utc').sort_index()
    es=es[~es.index.duplicated()];eb=aggregate(es,5)
    er=pd.Series(np.log(eb.close/eb.open).to_numpy(),index=pd.DatetimeIndex(eb.available_at)).reindex(pd.DatetimeIndex(b.available_at));er.index=b.index
    nqret=np.log(c/o)
    beta=nqret.rolling(128,min_periods=64).cov(er)/er.rolling(128,min_periods=64).var().replace(0,np.nan)
    add('es_adjusted_residual',(nqret-beta*er)/nqret.rolling(48).std().replace(0,np.nan),'intermarket')
    for w in [1,3,6,12]:add(f'es_return_sum_{w}',er.rolling(w).sum(),'intermarket')
train=f.split.eq('in_sample')&f.eligible
guard_research_sample(f.index[train],'threshold')
eligible=f.eligible.to_numpy(bool)
columns=[];registry=[];seen=set()
def atom(name,family,value,definition):
    mask=np.asarray(value,dtype=bool)&eligible
    signature=np.packbits(mask[train.to_numpy()],bitorder='little').tobytes()
    if not mask[train.to_numpy()].any() or signature in seen:return
    seen.add(signature);columns.append(mask)
    registry.append({'atom_id':len(columns)-1,'feature':name,'family':family,'definition':definition})
for name,value in features.items():
    x=value.loc[train].dropna()
    if not len(x):continue
    if set(x.unique()).issubset({False,True,0,1}):
        atom(name,families[name],value.fillna(False),f'{name} is true')
        atom(name,families[name],value.eq(0),f'{name} is false and observed')
        continue
    thresholds=np.unique(x.quantile([.1,.2,.3,.4,.5,.6,.7,.8,.9]).to_numpy())
    for q in thresholds:
        atom(name,families[name],value.gt(q),f'{name} > {float(q):.12g}; threshold fitted on IS only')
        atom(name,families[name],value.lt(q),f'{name} < {float(q):.12g}; threshold fitted on IS only')
for a,z in [(570,630),(630,690),(690,750),(750,810),(810,870),(870,930),(930,960),(570,720),(720,960)]:
    atom(f'time_{a}_{z}','time_of_day',f.slot.ge(a)&f.slot.lt(z),f'NY clock minutes in [{a},{z})')
bools=np.stack(columns)
padding=(-bools.shape[1])%64
packed=np.packbits(np.pad(bools,((0,0),(0,padding))),axis=1,bitorder='little').copy().view('<u8')
np.save(R/'atom_bits.npy',packed)
pd.DataFrame(registry).to_csv(R/'atoms.csv',index=False)
pd.DataFrame(features,index=f.index).to_parquet(R/'features.parquet')
print('ATOMS',len(registry),'FEATURES',len(features),'IS-unique event predicates',flush=True)
(R/'feature_manifest.json').write_text(json.dumps({'atoms':len(registry),'features':len(features),'training_only_thresholds':True,'all_inputs':'completed bars, previous session profiles and fresh completed higher-period bars; current minute open only for eligibility','profile':'2-point bins, minute volume allocated at close, approximate VAP','sources':['https://www.nber.org/papers/w7613','https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf'],'interpretation':'Mechanical quantitative families informed by research; not faithful replication of every cited paper or proof of NQ hourly edge'},indent=2))
