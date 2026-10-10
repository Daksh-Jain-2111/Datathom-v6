"""Final pipeline: train.csv + rebuilt origins 2024Q1-Q3, env columns dropped, blend, widen, write CSV."""
import sys, json
import numpy as np, pandas as pd
from common import load, DATA
from features import build
from augment import build_rows, ENV
from model import feature_sets, fit_predict, finalize

out = sys.argv[1]
cfgs = json.loads(sys.argv[2]) if len(sys.argv) > 2 else [dict(num_leaves=15, r_seen=600, r_un=400),
                                                         dict(num_leaves=15, r_seen=600, r_un=400, anchor='l1', anchor_un='nbr')]
WIDEN = float(sys.argv[3]) if len(sys.argv) > 3 else 1.1
SEEDS = (42, 7, 2026, 11, 99)
tr, te = load()
aug = build_rows(['2024Q1', '2024Q2', '2024Q3'], tr, te)
print('rebuilt rows:', len(aug), aug.origin_quarter.value_counts().to_dict())
trm = pd.concat([tr, aug], ignore_index=True); trm[ENV] = np.nan; te = te.copy(); te[ENV] = np.nan
df = build(trm, te)
df.to_parquet('feat_final.parquet')
full, nobank = feature_sets(df)
full = [c for c in full if c not in ENV]; nobank = [c for c in nobank if c not in ENV]
lab = df[df._is_test == 0]; tst = df[df._is_test == 1]
is_seen = tst.district_id.isin(set(lab.district_id)).values
acc = np.zeros((len(tst), 3))
for cfg in cfgs:
    cfg = dict(cfg); rs, ru = cfg.pop('r_seen'), cfg.pop('r_un')
    an, anu = cfg.pop('anchor', None), cfg.pop('anchor_un', None)
    ps = fit_predict(lab, tst[is_seen], full, rs, params=cfg, seeds=SEEDS, anchor=an)
    pu = fit_predict(lab, tst[~is_seen], nobank, ru, params=cfg, seeds=SEEDS, anchor=anu)
    for j, q in enumerate((0.1, 0.5, 0.9)):
        acc[is_seen, j] += ps[q] / len(cfgs); acc[~is_seen, j] += pu[q] / len(cfgs)
np.save(out.replace('.csv', '_raw.npy'), acc)
p10 = acc[:, 1] - WIDEN * (acc[:, 1] - acc[:, 0]); p90 = acc[:, 1] + WIDEN * (acc[:, 2] - acc[:, 1])
P = np.column_stack(finalize(p10, acc[:, 1], p90))
sub = pd.DataFrame({'id': tst.id.values, 'p10': P[:, 0], 'p50': P[:, 1], 'p90': P[:, 2]})
ss = pd.read_csv(DATA + 'sample_submission.csv'); sub = ss[['id']].merge(sub, on='id', how='left')
assert len(sub) == len(ss) and sub.notna().all().all() and np.isfinite(sub[['p10', 'p50', 'p90']].values).all()
assert (sub.p10 <= sub.p50).all() and (sub.p50 <= sub.p90).all()
sub.to_csv(out, index=False, float_format='%.4f'); print(out, 'written')
