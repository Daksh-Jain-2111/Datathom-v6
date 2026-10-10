"""LightGBM quantile models: a 'full' model for seen districts and a 'nobank'
model (no own banking/BRGI history) for unseen districts."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from common import QS, SEED

DROP_ALWAYS = {'id', 'district_id', 'state', 'region', 'origin_quarter', 'target_quarter', 'brgi',
               '_is_test', 'O', 'T'}
OWN_BANK_PREFIX = ('brgi_', 'score_', 'cr_', 'dr_', 'cy_', 'dy_', 'c_', 'd_', 'hdfc_', 'nowcast', 'resp_', 'res_part')
OWN_BANK_COLS = {'log_credit', 'log_deposit', 'log_offices', 'cd_ratio', 'credit_per_capita', 'offices_per_100k',
                 'credit_yoy_l1', 'credit_yoy_l2', 'deposit_yoy_l1', 'deposit_yoy_l2', 'offices_yoy_l1',
                 'credit_qoq_l1', 'credit_qoq_std8', 'credit_yoy_mean4', 'bank_gap_in_window', 'rrb_adjusted_l1'}


def feature_sets(df):
    num = [c for c in df.columns if c not in DROP_ALWAYS and df[c].dtype != object and not c.startswith(('s_', 'tgt_'))
           and c != 'nbw_s_resilience_T4']
    full = num
    nobank = [c for c in num if c not in OWN_BANK_COLS and not c.startswith(OWN_BANK_PREFIX)]
    return full, nobank


PARAMS = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=80, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=SEED,
              deterministic=True, force_row_wise=True, num_threads=2)


def anchor_values(d, anchor):
    """Baseline the trees model residuals around (None -> 0)."""
    if anchor is None:
        return np.zeros(len(d))
    if anchor == 'l1':
        return d.brgi_l1.fillna(d.nbw_brgi_l1).fillna(50).values
    if anchor == 'mix':
        return (0.5 * d.brgi_l1.fillna(d.nbw_brgi_l1) + 0.5 * d.nowcast3.fillna(d.nbw_nowcast3)).fillna(50).values
    if anchor == 'nbr':
        return d.nbw_brgi_l1.fillna(50).values
    if anchor == 'nbrmix':
        return (0.5 * d.nbw_brgi_l1.fillna(50) + 25).values
    raise ValueError(anchor)


def fit_predict_cb(trn, tst, feats, rounds, params=None, seeds=(SEED,), anchor=None):
    """CatBoost MultiQuantile: one model gives all three quantiles."""
    from catboost import CatBoostRegressor
    a_tr, a_te = anchor_values(trn, anchor), anchor_values(tst, anchor)
    P = np.zeros((len(tst), 3))
    for s in seeds:
        prm = dict(loss_function='MultiQuantile:alpha=0.1,0.5,0.9', iterations=rounds, learning_rate=0.06,
                   depth=6, l2_leaf_reg=3, random_seed=s, verbose=0, thread_count=2, allow_writing_files=False)
        prm.update(params or {})
        m = CatBoostRegressor(**prm)
        m.fit(trn[feats].values.astype(np.float32), trn.brgi.values - a_tr)
        P += (m.predict(tst[feats].values.astype(np.float32)) + a_te[:, None]) / len(seeds)
    return {0.1: P[:, 0], 0.5: P[:, 1], 0.9: P[:, 2]}


def fit_predict(trn, tst, feats, rounds, params=None, seeds=(SEED,), anchor=None):
    """Return dict q -> predictions on tst."""
    params = dict(params or {})
    if params.pop('algo', 'lgb') == 'cb':
        return fit_predict_cb(trn, tst, feats, rounds, params, seeds, anchor)
    out = {}
    tail = params.pop('tail', {})
    hl = params.pop('w_halflife', None); wq4 = params.pop('w_q4', 1.0); wh = params.pop('w_h', False)
    w = np.ones(len(trn))
    if hl:
        w *= 0.5 ** ((trn.O.max() - trn.O.values) / hl)
    if wq4 != 1.0:
        w *= np.where(trn.O.values % 4 == 3, wq4, 1.0)
    if wh:
        w *= np.where(trn.horizon.values >= 3, 1.25, 1.0)
    a_tr, a_te = anchor_values(trn, anchor), anchor_values(tst, anchor)
    for q in QS:
        p = np.zeros(len(tst))
        for s in seeds:
            prm = dict(PARAMS, **(params or {}), objective='quantile', alpha=q, seed=s)
            if q != 0.5:
                prm.update(tail)
            ds = lgb.Dataset(trn[feats], trn.brgi.values - a_tr, weight=w)
            m = lgb.train(prm, ds, num_boost_round=rounds[q] if isinstance(rounds, dict) else rounds)
            p += (m.predict(tst[feats]) + a_te) / len(seeds)
        out[q] = p
    return out


def fit_es(trn, val, feats, max_rounds=3000, params=None):
    """Early stopping per quantile on val; returns best rounds and val preds."""
    best, preds = {}, {}
    for q in QS:
        prm = dict(PARAMS, **(params or {}), objective='quantile', alpha=q)
        dtr = lgb.Dataset(trn[feats], trn.brgi.values)
        dva = lgb.Dataset(val[feats], val.brgi.values, reference=dtr)
        m = lgb.train(prm, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        best[q] = m.best_iteration
        preds[q] = m.predict(val[feats], num_iteration=m.best_iteration)
    return best, preds


def finalize(p10, p50, p90):
    P = np.sort(np.c_[p10, p50, p90], axis=1)
    P = np.clip(P, 0.0, 100.0)
    return P[:, 0], P[:, 1], P[:, 2]
