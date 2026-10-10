"""Interval calibration: scale lower/upper widths around p50 per (kind, horizon),
fitted leave-one-fold-out on OOF predictions to keep the estimate honest."""
import sys
import numpy as np
import pandas as pd
from common import score, pinball
from model import finalize

GRID = np.round(np.arange(0.7, 1.81, 0.05), 2)


def fit_scales(o):
    sc = {}
    for (k, h), g in o.groupby(['kind', 'horizon']):
        y, p10, p50, p90 = g.brgi.values, g.p10.values, g.p50.values, g.p90.values
        a = GRID[np.argmin([pinball(y, p50 - s * (p50 - p10), 0.1).mean() for s in GRID])]
        b = GRID[np.argmin([pinball(y, p50 + s * (p90 - p50), 0.9).mean() for s in GRID])]
        sc[(k, h)] = (a, b)
    return sc


def apply(o, sc):
    o = o.copy()
    for (k, h), (a, b) in sc.items():
        m = (o.kind == k) & (o.horizon == h)
        p50 = o.loc[m, 'p50']
        o.loc[m, 'p10'] = p50 - a * (p50 - o.loc[m, 'p10'])
        o.loc[m, 'p90'] = p50 + b * (o.loc[m, 'p90'] - p50)
    return o


if __name__ == '__main__':
    for tag in sys.argv[1:]:
        o = pd.read_parquet(f'oof_{tag}.parquet')
        cal = []
        for V in o.fold.unique():
            sc = fit_scales(o[o.fold != V])
            cal.append(apply(o[o.fold == V], sc))
        cal = pd.concat(cal)
        P0 = finalize(o.p10.values, o.p50.values, o.p90.values)
        P1 = finalize(cal.p10.values, cal.p50.values, cal.p90.values)
        print(tag, 'raw', round(score(o.brgi.values, *P0, o.horizon.values), 4),
              'calibrated(LOFO)', round(score(cal.brgi.values, *P1, cal.horizon.values), 4))
        print('  scales on all folds:', fit_scales(o))
