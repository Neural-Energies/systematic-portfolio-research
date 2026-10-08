from pathlib import Path
import json
import pandas as pd
from experiment import load_minutes
p=Path('work/nq_long_ml/reference_backtests/20261007T003526Z')
m=load_minutes('NQ')
t=pd.read_csv(p/'trades_ES_confirmed_NQ_recovery_E0044_base.csv',parse_dates=['entry_time','exit_time','planned_exit'])
last=None
for _,r in t.iterrows():
    assert last is None or r.entry_time>=last
    window=m.loc[(m.index>=r.entry_time)&(m.index<r.exit_time)]
    assert len(window)==r.holding_minutes
    assert abs(((r.exit_price-r.entry_price)*20-25)-r.net_dollars)<1e-8
    if r.status=='target':
        touches=window[window.high>=r.target_price]
        assert len(touches) and touches.index[0]+pd.Timedelta(minutes=1)==r.exit_time
        assert r.exit_price==r.target_price
    else:
        assert (window.high<r.target_price).all()
        assert r.exit_time==r.planned_exit
        assert r.exit_price==window.close.iloc[-1]
    last=r.exit_time
checks={'independent_candidate_trade_replays':len(t),'nonoverlapping':True,'target_first_touch_and_timeout_price':True,'net_cost_arithmetic':True}
(p/'execution_audit.json').write_text(json.dumps(checks,indent=2))
print(checks)
