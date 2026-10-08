"""Isolated timestamp-end sensitivity branch; keep all original thresholds frozen."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path('work/nq_long_ml').resolve()))
corrected='data/processed/nq_entry_discovery_202210_minute_end_scenario'
folder=Path(corrected);folder.mkdir(exist_ok=True)
code=Path('work/nq_long_ml/prepare_oct2022_discovery.py').read_text()
code=code.replace("root=Path('data/processed/nq_entry_discovery_202210')",f"root=Path('{corrected}')")
code=code.replace(".tz_convert('UTC')\nm=raw.rename", ".tz_convert('UTC')-pd.Timedelta(minutes=1)\nm=raw.rename")
code=code.replace("assumed start-of-minute export timestamps; verify exporter before live claim","minute-end sensitivity: source timestamps minus1minute supported by6849 exact overlapping5m volumes/OHLC shapes; human confirmation pending")
exec(compile(code,'<prepare-minute-end-scenario>','exec'),{})
code=Path('work/nq_long_ml/build_oct2022_atoms.py').read_text().split("train=f.split.eq('in_sample')")[0]
code=code.replace("R=Path('data/processed/nq_entry_discovery_202210')",f"R=Path('{corrected}')")
namespace={};exec(compile(code,'<clock-end-features>','exec'),namespace)
f=namespace['f'];features=pd.DataFrame(namespace['features'],index=f.index);features.to_parquet(folder/'features.parquet')
locked=pd.read_csv('saved_strategies/NQ_RTH_100_LOCKED_20261007/FULL_VALIDATION_20261007/exact_frozen_atoms.csv')
original=pd.read_csv('data/processed/nq_entry_discovery_202210/atoms.csv');original.to_csv(folder/'atoms.csv',index=False)
words=np.zeros((len(original),int(np.ceil(len(f)/64))),dtype=np.uint64)
for _,atom in locked.iterrows():
 mask=(features[atom.feature]>atom.threshold_full_precision if atom.operator=='>' else features[atom.feature]<atom.threshold_full_precision)&f.eligible
 packed=np.packbits(mask.to_numpy(),bitorder='little');packed=np.pad(packed,(0,int(np.ceil(len(packed)/8))*8-len(packed)))
 words[int(atom.atom_id)]=packed.view(np.uint64)
np.save(folder/'atom_bits.npy',words)
(folder/'FROZEN_THRESHOLD_PROTOCOL.json').write_text(json.dumps({'original_numeric_thresholds_retained':True,'refitted_on_new_clock':False,'source_atoms':'Original FULL_VALIDATION_20261007/exact_frozen_atoms.csv','clock_interpretation':'Sensitivity case supported by new export; awaiting human endpoint confirmation'},indent=2))
print('CLOCK-END DATA AND FEATURES READY',len(f),len(locked),flush=True)

source=Path('work/nq_long_ml/full_locked_validation.py').read_text()
source=source.replace("P=Path('data/processed/nq_entry_discovery_202210')",f"P=Path('{corrected}')")
source=source.replace("O=L/'FULL_VALIDATION_20261007'", "O=L/'FULL_VALIDATION_20261007'/'MINUTE_END_SCENARIO'")
old="values=features.loc[train,field].dropna().replace([np.inf,-np.inf],np.nan).dropna().quantile(np.arange(.1,1,.1)).to_numpy()\n    threshold=values[np.argmin(np.abs(values-printed))]"
new="original_atoms=pd.read_csv(L/'FULL_VALIDATION_20261007'/'exact_frozen_atoms.csv').set_index('atom_id')\n    threshold=float(original_atoms.loc[atom,'threshold_full_precision'])"
assert old in source;source=source.replace(old,new)
source=source.replace("same=canonical.set_index('timestamp_utc')", "same=canonical.assign(timestamp_utc=lambda d:d.timestamp_utc-pd.Timedelta(minutes=1)).set_index('timestamp_utc')")
old="for key in ['trades','net_profit','profit_factor','net_win_rate','target_hit_rate']:\n                    if not np.isclose(kp[key],previous[key],rtol=1e-9,atol=1e-7,equal_nan=True):raise RuntimeError((cid,stage,key,kp[key],previous[key]))\n                checks.append({'candidate':cid,'period':stage,'trade_kpis_reconciled':True})"
new="checks.append({'candidate':cid,'period':stage,'original_net_profit':previous.net_profit,'minute_end_net_profit':kp['net_profit'],'original_pf':previous.profit_factor,'minute_end_pf':kp['profit_factor'],'original_trades':previous.trades,'minute_end_trades':kp['trades']})"
assert old in source;source=source.replace(old,new)
source=source.replace("'all300_base_period_results_reconciled':True", "'all300_base_period_results_reconciled':False,'reason':'New clock-end scenario compared separately with archived original clock results; all original numeric thresholds remain fixed'")
source=source.replace("'bar_timestamp_semantics':'Start-minute assumption; exporter not independently confirmed'", "'bar_timestamp_semantics':'Source minute-end sensitivity: UTC labels minus1minute; new5m end-label interpretation awaiting human confirmation'")
source=source.replace("(L/'CURRENT_REQUIREMENTS.json').write_text", "(O/'CURRENT_REQUIREMENTS.json').write_text")
source=source.replace("'exact_atom_reconstruction':len(needed)", "'exact_atom_reconstruction':len(needed),'original_thresholds_refitted':False")
Path('work/nq_long_ml/full_locked_validation_minute_end.py').write_text(source,encoding='utf-8')
