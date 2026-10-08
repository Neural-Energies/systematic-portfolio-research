from pathlib import Path
import json,zipfile
import numpy as np,pandas as pd
from systematic_research.reference_backtest import backtest_reference
P=Path('data/processed/nq_entry_discovery_202210');R=Path('work/nq_long_ml/ten_million_entry_search')
f=pd.read_parquet(P/'opportunities.parquet');m=pd.read_parquet(P/'minutes.parquet');bits=np.load(P/'atom_bits.npy')
checks=[]
for archive in sorted(R.glob('batch_*.zip'))[:3]:
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist()[:1]:
            x=json.loads(z.read(name));abc=[q['atom_id'] for q in x['definition']]
            words=bits[abc[0]]&bits[abc[1]]&bits[abc[2]]
            signal=pd.Series(np.unpackbits(words.view(np.uint8),bitorder='little')[:len(f)].astype(bool),index=f.index)
            for label,split in [('is','in_sample'),('validation','validation')]:
                scope=f[f.split.eq(split)]
                assert len(scope[signal.reindex(scope.index)])==x[label]['events']
                assert int(scope.loc[signal.reindex(scope.index),'mean70_hit'].eq(1).sum())==x[label]['hits']
                trades,_=backtest_reference(m,scope,signal)
                metric=x['execution_metrics'][label]
                assert len(trades)==metric['selected_entries']
                assert abs(trades.net_dollars.sum()-metric['observed_net_dollars'])<1e-8
                cumulative=np.r_[0.,trades.net_dollars.fillna(0).cumsum().to_numpy()]
                dd=float(np.max(np.maximum.accumulate(cumulative)-cumulative))
                assert abs(dd-metric['max_closed_trade_drawdown_dollars'])<1e-8
                checks.append({'candidate':x['id'],'split':split,'counts_pnl_drawdown_match':True})
(R/'independent_execution_audit.json').write_text(json.dumps(checks,indent=2))
print('INDEPENDENT CHECKS',len(checks))
