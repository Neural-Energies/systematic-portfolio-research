"""Compare frozen features with rebuilding from a truncated NQ observation history."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path('work/nq_long_ml').resolve()))
O=Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/FULL_VALIDATION_20261007')
source=Path('work/nq_long_ml/build_oct2022_atoms.py').read_text().split("train=f.split.eq('in_sample')")[0]
short=json.loads(Path('saved_strategies/NQ_RTH_100_LOCKED_20261007/LOCKED_ENTRIES.json').read_text());needed=sorted({q['feature'] for x in short for q in x['definition']})
recorded=pd.read_parquet('data/processed/nq_entry_discovery_202210/features.parquet')
results=[]
for date in ['2025-04-15 15:00:00+00:00','2026-06-15 15:00:00+00:00']:
    cutoff=pd.Timestamp(date)
    insertion="\nf=f.loc[f.index<=cutoff].copy();m=m.loc[m.index<cutoff].copy();b=b.loc[b.available_at<=cutoff].copy()\n"
    code=source.replace("c,o,h,l,v=[b[x].astype(float)",insertion+"c,o,h,l,v=[b[x].astype(float)")
    namespace={'cutoff':cutoff};exec(compile(code,'<frozen-feature-prefix>','exec'),namespace)
    rebuilt=pd.DataFrame(namespace['features'],index=namespace['f'].index)
    latest=rebuilt.index[rebuilt.index<=cutoff][-1];left=recorded.loc[latest,needed].to_numpy(float);right=rebuilt.loc[latest,needed].to_numpy(float)
    mismatch=~np.isclose(left,right,rtol=1e-9,atol=1e-9,equal_nan=True)
    results.append({'cutoff':str(latest),'features_checked':len(needed),'mismatches':int(mismatch.sum()),'columns_mismatched':[needed[i] for i in np.flatnonzero(mismatch)]})
    if mismatch.any():raise RuntimeError(results[-1])
(O/'FEATURE_PREFIX_VERIFICATION.json').write_text(json.dumps({'prefix_checks':results,'NQ_observations_after_cutoff_removed':True,'ES_context':'Aligned completed bars at or before truncated NQ availability clock','scope':'Two boundary spot checks, not an exhaustive formal causality proof'},indent=2))
print(json.dumps(results,indent=2))
