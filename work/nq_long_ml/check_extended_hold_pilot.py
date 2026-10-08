"""Independent execution boundary checks without importing the research runner."""
import ast
from pathlib import Path
import numpy as np
from numba import njit

tree=ast.parse(Path('work/nq_long_ml/extended_hold_pilot.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
scope={'np':np,'njit':njit}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'<replay-test>','exec'),scope)
replay=scope['replay']
minute=60_000_000_000
clock=np.arange(30,dtype=np.int64)*5*minute
price=100+np.arange(30,dtype=float)
high=price+.5;low=price-.5
high[0]=1000.;low[0]=-1000. # Before-entry candle must not enter excursion metrics.
out=replay(np.array([0]),np.array([60*minute]),clock,price,price,low,high,np.zeros(30),60*minute,200*minute)
assert out[0,1]==12 and out[0,3]==60 and out[0,2]==215
assert out[0,4]==0 and out[0,5]==12.5
# A model seeing no continuation exits at the first permitted decision.
out=replay(np.array([0]),np.array([120*minute]),clock,price,price,low,high,np.zeros(30),60*minute,200*minute)
assert out[0,1]==12
# Continuation cannot fabricate a quote beyond a period boundary.
out=replay(np.array([0]),np.array([120*minute]),clock,price,price,low,high,np.ones(30),60*minute,100*minute)
assert out[0,1]==-1 and np.isnan(out[0,2])
print('PASS: fixed deadline, pre-entry excursion exclusion, continuation exit and split boundary')
