"""Shared helpers: data loading, quarter math, metric."""
import numpy as np
import pandas as pd

import os
DATA = os.environ.get('BRGI_DATA', '../data/')  # folder with the competition CSVs
QS = [0.1, 0.5, 0.9]
HW = {1: 1.0, 2: 1.0, 3: 1.25, 4: 1.25}
SEED = 42


def qidx(s):
    """'2024Q4' -> integer quarter index."""
    s = pd.Series(s).astype(str)
    return (s.str[:4].astype(int) * 4 + s.str[-1].astype(int) - 1).values


def qstr(i):
    i = np.asarray(i)
    return np.char.add(np.char.add((i // 4).astype(str), 'Q'), (i % 4 + 1).astype(str))


def pinball(y, p, q):
    d = y - p
    return np.maximum(q * d, (q - 1) * d)


def score(y, P10, P50, P90, h):
    """Horizon-weighted mean pinball, averaged over the 3 quantiles."""
    w = np.vectorize(HW.get)(h)
    out = []
    for q, p in zip(QS, (P10, P50, P90)):
        out.append(np.sum(w * pinball(y, p, q)) / np.sum(w))
    return float(np.mean(out))


def load():
    tr = pd.read_csv(DATA + 'train.csv')
    te = pd.read_csv(DATA + 'test.csv')
    return tr, te
