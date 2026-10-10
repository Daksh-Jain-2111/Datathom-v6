"""Fixed-round backtests over several origins; saves OOF predictions for calibration/blending."""
import sys, json
import numpy as np
import pandas as pd
from common import qidx, score, SEED
from model import feature_sets, fit_predict, finalize

cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
tag = cfg.pop('tag', 'base')
r_seen = cfg.pop('r_seen', 400); r_un = cfg.pop('r_un', 250)
folds = cfg.pop('folds', ['2021Q4', '2022Q4', '2023Q4'])
min_origin = cfg.pop('min_origin', None)
drop = set(cfg.pop('drop', []))
use_stack = cfg.pop('stack', False)
gap = cfg.pop('gap', 0)
per_h = cfg.pop('per_h', False)
anchor = cfg.pop('anchor', None)
anchor_un = cfg.pop('anchor_un', None)
df = pd.read_parquet('feat.parquet')
full, nobank = feature_sets(df)
full = [c for c in full if c not in drop]; nobank = [c for c in nobank if c not in drop]
lab = df[df._is_test == 0]
if min_origin: lab_tr = lab[lab.O >= qidx([min_origin])[0]]
else: lab_tr = lab
rng = np.random.default_rng(SEED)
oof = []
for V in folds:
    v = qidx([V])[0]
    dists = lab[lab.O == v].district_id.unique()
    pseudo = set(rng.choice(dists, int(0.15 * len(dists)), replace=False))
    trn = lab_tr[(lab_tr['T'] <= v) & ~lab_tr.district_id.isin(pseudo) & (lab_tr.O <= v - gap)]
    val = lab[lab.O == v]
    vs, vu = val[~val.district_id.isin(pseudo)], val[val.district_id.isin(pseudo)]
    F, N = full, nobank
    if use_stack:
        from stack import add_stack, FULL_EXTRA, NOBANK_EXTRA
        trn, (vs, vu) = add_stack(trn, [vs, vu], full, nobank, max_T=v - 1)
        F, N = full + FULL_EXTRA, nobank + NOBANK_EXTRA
    if per_h:
        ps = {q: np.zeros(len(vs)) for q in (0.1, 0.5, 0.9)}; pu = {q: np.zeros(len(vu)) for q in (0.1, 0.5, 0.9)}
        for hz in (1, 2, 3, 4):
            ms, mu_ = vs.horizon.values == hz, vu.horizon.values == hz
            t = trn[trn.horizon == hz]
            a = fit_predict(t, vs[ms], F, r_seen, params=cfg, anchor=anchor); b = fit_predict(t, vu[mu_], N, r_un, params=cfg, anchor=anchor_un)
            for q in (0.1, 0.5, 0.9): ps[q][ms] = a[q]; pu[q][mu_] = b[q]
    else:
        ps = fit_predict(trn, vs, F, r_seen, params=cfg, anchor=anchor)
        pu = fit_predict(trn, vu, N, r_un, params=cfg, anchor=anchor_un)
    for d, p, kind in ((vs, ps, 'seen'), (vu, pu, 'unseen')):
        o = d[['id', 'district_id', 'horizon', 'brgi', 'brgi_l1']].copy()
        o['fold'] = V; o['kind'] = kind
        o['p10'], o['p50'], o['p90'] = p[0.1], p[0.5], p[0.9]
        oof.append(o)
oof = pd.concat(oof, ignore_index=True)
oof.to_parquet(f'oof_{tag}.parquet')
P = finalize(oof.p10.values, oof.p50.values, oof.p90.values)
for V, g in oof.groupby('fold'):
    m = (oof.fold == V).values
    line = f'{tag} {V}: all={score(oof.brgi.values[m], P[0][m], P[1][m], P[2][m], oof.horizon.values[m]):.4f}'
    for k in ('seen', 'unseen'):
        mk = m & (oof.kind == k).values
        line += f' {k}={score(oof.brgi.values[mk], P[0][mk], P[1][mk], P[2][mk], oof.horizon.values[mk]):.4f}'
    print(line)
print(f'{tag} MEAN all={score(oof.brgi.values, *P, oof.horizon.values):.4f}')
