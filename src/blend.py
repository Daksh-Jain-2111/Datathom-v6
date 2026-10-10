import sys, numpy as np, pandas as pd
from common import score
from model import finalize
tags = sys.argv[1:]
os_ = [pd.read_parquet(f'oof_{t}.parquet').set_index('id').sort_index() for t in tags]
b = os_[0].copy()
for c in ('p10', 'p50', 'p90'):
    b[c] = np.mean([o[c].values for o in os_], axis=0)
P = finalize(b.p10.values, b.p50.values, b.p90.values)
print('blend', tags, round(score(b.brgi.values, *P, b.horizon.values), 4))
for k in ('seen', 'unseen'):
    m = (b.kind == k).values
    print(' ', k, round(score(b.brgi.values[m], P[0][m], P[1][m], P[2][m], b.horizon.values[m]), 4))
