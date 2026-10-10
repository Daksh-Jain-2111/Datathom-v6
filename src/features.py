"""Feature engineering for the BRGI datathon.

Every feature for a row (district d, origin O, horizon h, target T=O+h) uses only
information available at origin O, per release_calendar.csv:
  * BRGI labels / component scores: up to quarter O-1 (1-quarter lag)
  * bank_quarterly: up to quarter O-1
  * nightlights_monthly: up to the 2nd month of quarter O (1-month lag)
BRGI = 35*credit_yoy_rank + 20*deposit_yoy_rank + 30*ntl_yoy_rank + 15*resilience_rank,
so many features are cross-sectional ranks / projections of those components.
"""
import numpy as np
import pandas as pd
from common import DATA, qidx

SCORE_COLS = ['score_credit_yoy_l1', 'score_deposit_yoy_l1', 'score_ntl_yoy_l1', 'score_resilience_l1']


def _matrix(df, row, col, val, rows, ncol, col0):
    M = np.full((len(rows), ncol), np.nan)
    ri = pd.Index(rows).get_indexer(df[row])
    ci = df[col].values - col0
    ok = (ri >= 0) & (ci >= 0) & (ci < ncol)
    M[ri[ok], ci[ok]] = df[val].values[ok]
    return M


def _rank_cols(M):
    """Percentile rank of each column (cross-section of districts), NaN-aware."""
    return pd.DataFrame(M).rank(pct=True).values


def build(tr, te, bank=None):
    tr = tr.copy(); te = te.copy()
    tr['_is_test'] = 0; te['_is_test'] = 1
    df = pd.concat([tr, te], ignore_index=True)
    df['O'] = qidx(df.origin_quarter)
    df['T'] = df.O + df.horizon
    dists = sorted(set(df.district_id) | set(pd.read_csv(DATA + 'district_static.csv').district_id))
    dix = pd.Index(dists).get_indexer(df.district_id)
    Q0 = qidx(['2004Q1'])[0]; NQ = qidx(['2025Q4'])[0] - Q0 + 1
    O = df.O.values; T = df['T'].values; h = df.horizon.values

    def gather(M, q, valid):
        q = np.asarray(q); out = np.full(len(df), np.nan)
        c = q - Q0
        ok = valid & (c >= 0) & (c < M.shape[1])
        out[ok] = M[dix[ok], c[ok]]
        return out

    F = {}
    # ---------- BRGI label history (labels only up to O-1) ----------
    lab = tr[['district_id', 'target_quarter', 'brgi']].drop_duplicates(['district_id', 'target_quarter'])
    lab['q'] = qidx(lab.target_quarter)
    B = _matrix(lab, 'district_id', 'q', 'brgi', dists, NQ, Q0)
    for k in range(3, 9):
        F[f'brgi_l{k}'] = gather(B, O - k, np.ones(len(df), bool))
    F['brgi_T4'] = gather(B, T - 4, T - 4 <= O - 1)
    F['brgi_T8'] = gather(B, T - 8, T - 8 <= O - 1)
    # expanding stats up to O-1
    cs = np.nancumsum(np.nan_to_num(B), axis=1); cn = np.cumsum(~np.isnan(B), axis=1)
    cs2 = np.nancumsum(np.nan_to_num(B) ** 2, axis=1)
    s = gather(cs, O - 1, np.ones(len(df), bool)); n = gather(cn.astype(float), O - 1, np.ones(len(df), bool))
    s2 = gather(cs2, O - 1, np.ones(len(df), bool))
    with np.errstate(invalid='ignore', divide='ignore'):
        F['brgi_exp_mean'] = np.where(n > 0, s / n, np.nan)
        F['brgi_exp_std'] = np.where(n > 1, np.sqrt(np.maximum(s2 / n - (s / n) ** 2, 0)), np.nan)
    F['brgi_hist_n'] = np.nan_to_num(n)
    lags = np.column_stack([df.brgi_l1.values, df.brgi_l2.values] + [F[f'brgi_l{k}'] for k in range(3, 9)])
    with np.errstate(all='ignore'):
        import warnings; warnings.simplefilter('ignore')
        F['brgi_mean8'] = np.nanmean(lags, axis=1)
    F['brgi_slope4'] = df.brgi_l1.values - F['brgi_l4']

    # ---------- component-score history (score_*_l1 at origin O == score at O-1) ----------
    for c in SCORE_COLS:
        sc = df[['district_id', 'O', c]].dropna().drop_duplicates(['district_id', 'O'])
        sc['q'] = sc.O - 1
        S = _matrix(sc, 'district_id', 'q', c, dists, NQ, Q0)
        base = c.replace('score_', '').replace('_l1', '')
        F[f's_{base}_T4'] = gather(S, T - 4, T - 4 <= O - 1)
        F[f's_{base}_l4'] = gather(S, O - 4, np.ones(len(df), bool))
        F[f's_{base}_l2'] = gather(S, O - 2, np.ones(len(df), bool))

    # ---------- bank reconstruction (up to O-1) ----------
    bq = pd.read_csv(DATA + 'bank_quarterly.csv') if bank is None else bank.copy()
    bq['q'] = qidx(bq.quarter)
    bq['lc'] = np.log(bq.credit); bq['ld'] = np.log(bq.deposit); bq['lo'] = np.log(bq.offices)
    LC = _matrix(bq, 'district_id', 'q', 'lc', dists, NQ, Q0)
    LD = _matrix(bq, 'district_id', 'q', 'ld', dists, NQ, Q0)
    HD = _matrix(bq, 'district_id', 'q', 'hdfc_merger_window', dists, NQ, Q0)
    for name, L in (('c', LC), ('d', LD)):
        Y = np.full_like(L, np.nan); Y[:, 4:] = L[:, 4:] - L[:, :-4]
        R = _rank_cols(Y)
        for k in (1, 2, 3, 4):
            F[f'{name}r_l{k}'] = gather(R, O - k, np.ones(len(df), bool))
        F[f'{name}r_T4'] = gather(R, T - 4, T - 4 <= O - 1)
        F[f'tgt_{name}r'] = gather(R, T, np.ones(len(df), bool))  # analysis only, never a feature
        F[f'{name}y_T4'] = gather(Y, T - 4, T - 4 <= O - 1)
        last = gather(L, O - 1, np.ones(len(df), bool))
        l5 = gather(L, O - 5, np.ones(len(df), bool))
        l3 = gather(L, O - 3, np.ones(len(df), bool))
        g4 = (last - l5) / 4.0                      # avg qoq over last year
        g2 = (last - l3) / 2.0                      # avg qoq over last 2 quarters
        base = gather(L, T - 4, T - 4 <= O - 1)     # level at T-4 if known
        base = np.where(T - 4 <= O - 1, base, last + (T - 4 - (O - 1)) * g4)
        F[f'{name}_partial'] = np.where(T - 4 <= O - 2, last - base, np.nan)  # realised part of yoy(T)
        F[f'{name}_est_yoy'] = (last + (T - (O - 1)) * g4) - base
        F[f'{name}_est_yoy2'] = (last + (T - (O - 1)) * (0.5 * g4 + 0.5 * g2)) - base
        # seasonal: known part + median of the same-calendar-span growth in past years
        rem = np.column_stack([gather(L, T - 4 * k, T - 4 * k <= O - 1) - gather(L, O - 1 - 4 * k, np.ones(len(df), bool))
                               for k in (1, 2, 3)])
        pastyoy = np.column_stack([gather(L, T - 4 * k, T - 4 * k <= O - 1) - gather(L, T - 4 * k - 4, T - 4 * k <= O - 1)
                                   for k in (2, 3, 4)])
        with np.errstate(all='ignore'):
            import warnings; warnings.simplefilter('ignore')
            R = np.nanmedian(rem, axis=1); PY = np.nanmedian(pastyoy, axis=1)
        known = np.where(T - 4 <= O - 1, last - gather(L, T - 4, T - 4 <= O - 1), np.nan)
        F[f'{name}_seas_est_yoy'] = np.where(T - 4 <= O - 1, known + R, 0.5 * PY + 0.5 * 4 * g4)
    # resilience proxy: rank of (mean - 0.5*sd) of the last 4 credit qoq growths (corr ~0.83 with score)
    QQ = np.full_like(LC, np.nan); QQ[:, 1:] = LC[:, 1:] - LC[:, :-1]
    W = np.stack([np.roll(QQ, k, axis=1) for k in range(4)], axis=2); W[:, :3, :] = np.nan
    with np.errstate(all='ignore'):
        RES = _rank_cols(np.nanmean(W, axis=2) - 0.5 * np.nanstd(W, axis=2, ddof=1))
    for k in (1, 2, 4):
        F[f'resp_l{k}'] = gather(RES, O - k, np.ones(len(df), bool))
    F['resp_T4'] = gather(RES, T - 4, T - 4 <= O - 1)
    F['tgt_res'] = gather(RES, T, np.ones(len(df), bool))  # analysis only
    # known part of the window that defines resilience at T (qoq at T-3..O-1)
    kq = np.column_stack([np.where(T - j <= O - 1, gather(QQ, T - j, T - j <= O - 1), np.nan) for j in range(4)])
    with np.errstate(all='ignore'):
        F['res_part_mean'] = np.nanmean(kq, axis=1)
        F['res_part_min'] = np.nanmin(kq, axis=1)
    F['hdfc_T4'] = gather(HD, T - 4, T - 4 <= O - 1)
    F['hdfc_l1'] = gather(HD, O - 1, np.ones(len(df), bool))

    # ---------- night lights (known to 2nd month of quarter O) ----------
    nl = pd.read_csv(DATA + 'nightlights_monthly.csv')
    nl['m'] = nl.month.str[:4].astype(int) * 12 + nl.month.str[5:7].astype(int) - 1
    M0 = nl.m.min(); NM = (2025 * 12 + 12) - M0
    NLm = _matrix(nl, 'district_id', 'm', 'ntl_sum', dists, NM, M0)
    with np.errstate(divide='ignore', invalid='ignore'):
        LNm = np.log(np.maximum(NLm, 1e-3))
    # quarterly log-mean (complete quarters)
    qm = np.arange(NQ) + Q0  # quarter index -> first month
    NQL = np.full((len(dists), NQ), np.nan)
    for j, q in enumerate(qm):
        fm = (q // 4) * 12 + (q % 4) * 3 - M0
        if fm >= 0 and fm + 3 <= NM:
            blk = NLm[:, fm:fm + 3]
            NQL[:, j] = np.log(np.maximum(blk.mean(axis=1), 1e-3))
    NY = np.full_like(NQL, np.nan); NY[:, 4:] = NQL[:, 4:] - NQL[:, :-4]
    NR = _rank_cols(NY)
    for k in (1, 2, 3, 4):
        F[f'nr_l{k}'] = gather(NR, O - k, np.ones(len(df), bool))
    F['nr_T4'] = gather(NR, T - 4, T - 4 <= O - 1)
    F['tgt_nr'] = gather(NR, T, np.ones(len(df), bool))  # analysis only, never a feature
    F['ny_T4'] = gather(NY, T - 4, T - 4 <= O - 1)
    # partial current quarter: first 2 months of quarter O vs same 2 months a year earlier
    fmO = (O // 4) * 12 + (O % 4) * 3 - M0
    def mgather(M, mi):
        out = np.full(len(df), np.nan); ok = (mi >= 0) & (mi < M.shape[1])
        out[ok] = M[dix[ok], mi[ok]]; return out
    cur2 = np.log(np.maximum((mgather(NLm, fmO) + mgather(NLm, fmO + 1)) / 2, 1e-3))
    prv2 = np.log(np.maximum((mgather(NLm, fmO - 12) + mgather(NLm, fmO - 11)) / 2, 1e-3))
    F['ntl_cur_yoy2m'] = cur2 - prv2
    # level of last 3 known months vs T-4 quarter level (seasonality-naive)
    last3 = np.log(np.maximum((mgather(NLm, fmO - 1) + mgather(NLm, fmO) + mgather(NLm, fmO + 1)) / 3, 1e-3))
    ntlT4 = gather(NQL, T - 4, T - 4 <= O)  # quarter T-4 complete only if <= O-1 (O partially)
    ntlT4 = np.where(T - 4 <= O - 1, ntlT4, np.nan)
    F['ntl_base_gap'] = ntlT4 - gather(NQL, T - 8, T - 8 <= O - 1)  # yoy at T-4 (base effect)
    # seasonal-adjusted projection: yoy at T ~ recent yoy momentum
    F['ntl_est_yoy'] = np.where(np.isnan(F['ntl_cur_yoy2m']), gather(NY, O - 1, np.ones(len(df), bool)), F['ntl_cur_yoy2m'])
    # seasonal-level projection: est log NTL(T) = trailing-4q level + district seasonal factor of T's quarter
    nd = len(dists); Y4 = NQL.reshape(nd, NQ // 4, 4)                       # (district, year, qtr)
    DEV = Y4 - np.nanmean(Y4, axis=2, keepdims=True)                         # seasonal deviation in a year
    yO = (O - Q0) // 4                                                       # index of origin's year
    qT = (T - Q0) % 4
    seas = np.full(len(df), 0.0); cnt = np.zeros(len(df))
    for k in (1, 2, 3):                                                      # last 3 complete years (< origin year)
        yy = yO - k; ok = yy >= 0
        v = np.full(len(df), np.nan); v[ok] = DEV[dix[ok], yy[ok], qT[ok]]
        good = ~np.isnan(v); seas[good] += v[good]; cnt[good] += 1
    seas = np.where(cnt > 0, seas / np.maximum(cnt, 1), np.nan)
    lvl4 = np.nanmean(np.column_stack([gather(NQL, O - k, np.ones(len(df), bool)) for k in (1, 2, 3, 4)]), axis=1)
    est_T = lvl4 + seas
    qT4 = (T - 4 - Q0) % 4
    seas4 = np.full(len(df), 0.0); cnt4 = np.zeros(len(df))
    for k in (1, 2, 3):
        yy = yO - k; ok = yy >= 0
        v = np.full(len(df), np.nan); v[ok] = DEV[dix[ok], yy[ok], qT4[ok]]
        good = ~np.isnan(v); seas4[good] += v[good]; cnt4[good] += 1
    seas4 = np.where(cnt4 > 0, seas4 / np.maximum(cnt4, 1), np.nan)
    base_T4 = np.where(T - 4 <= O - 1, gather(NQL, T - 4, T - 4 <= O - 1),
                       np.where(np.isnan(cur2), lvl4 + seas4, cur2))   # h=4: quarter O from its 2 known months
    F['ntl_seas_est_yoy'] = est_T - base_T4
    F['ntl_base_dev'] = base_T4 - (lvl4 + seas4)          # how abnormal the comparison base is
    F['ntl_lvl_vs_last12'] = last3 - lvl4

    # ---------- cross-sectional ranks within (origin, horizon) ----------
    tmp = pd.DataFrame({'O': O, 'h': h})
    for c in ['c_partial', 'd_partial', 'c_est_yoy', 'd_est_yoy', 'c_est_yoy2', 'd_est_yoy2', 'ntl_cur_yoy2m', 'ntl_est_yoy', 'ntl_base_gap', 'ntl_seas_est_yoy', 'ntl_base_dev', 'c_seas_est_yoy', 'd_seas_est_yoy', 'res_part_mean']:
        tmp[c] = F[c]
        F[c + '_rk'] = tmp.groupby(['O', 'h'])[c].rank(pct=True).values
    res = df.score_resilience_l1.values
    F['nowcast'] = 100 * (0.35 * F['c_est_yoy_rk'] + 0.20 * F['d_est_yoy_rk'] + 0.30 * F['ntl_est_yoy_rk'] + 0.15 * res)
    F['nowcast3'] = 100 * (0.35 * F['c_est_yoy_rk'] + 0.20 * F['d_est_yoy_rk'] + 0.30 * F['ntl_seas_est_yoy_rk'] + 0.15 * res)
    F['nowcast2'] = 100 * (0.35 * F['c_seas_est_yoy_rk'] + 0.20 * F['d_seas_est_yoy_rk'] + 0.30 * F['ntl_seas_est_yoy_rk'] + 0.15 * res)
    F['nowcast_nores'] = 100 * (0.35 * F['c_est_yoy_rk'] + 0.20 * F['d_est_yoy_rk'] + 0.30 * F['ntl_est_yoy_rk']) / 0.85

    feat = pd.DataFrame(F, index=df.index)
    df = pd.concat([df, feat], axis=1)

    # ---------- static / lineage / graph ----------
    st = pd.read_csv(DATA + 'district_static.csv')
    extra = [c for c in st.columns if c not in df.columns]
    df = df.merge(st[['district_id'] + extra], on='district_id', how='left')
    li = pd.read_csv(DATA + 'district_lineage.csv')
    df = df.merge(li, on='district_id', how='left')
    g = pd.read_csv(DATA + 'district_graph.csv')
    deg = g.groupby('district_id').agg(nbr_n=('neighbor_id', 'size'), nbr_dist_mean=('distance_km', 'mean'),
                                       nbr_same_state=('same_state', 'mean')).reset_index()
    df = df.merge(deg, on='district_id', how='left')

    # ---------- neighbour aggregates (inverse-distance weighted, same origin/horizon) ----------
    nb_cols = ['brgi_l1', 'brgi_mean4', 'brgi_exp_mean', 'brgi_T4', 'nowcast', 'nowcast2', 'nowcast3', 'c_est_yoy_rk', 'd_est_yoy_rk', 'ntl_seas_est_yoy_rk', 'c_seas_est_yoy_rk',
               'ntl_est_yoy_rk', 'ntl_cur_yoy2m_rk', 'score_resilience_l1', 'score_credit_yoy_l1', 's_resilience_T4']
    key = df[['district_id', 'O', 'horizon'] + nb_cols].drop_duplicates(['district_id', 'O', 'horizon'])
    gg = g.copy(); gg['w'] = 1.0 / np.maximum(gg.distance_km, 10.0)
    m = gg.merge(key.rename(columns={'district_id': 'neighbor_id'}), on='neighbor_id')
    agg = {}
    for c in nb_cols:
        v = m[c]; ok = v.notna()
        m[c + '_wv'] = np.where(ok, v * m.w, 0.0); m[c + '_ww'] = np.where(ok, m.w, 0.0)
    grp = m.groupby(['district_id', 'O', 'horizon'])
    out = grp[[c + '_wv' for c in nb_cols] + [c + '_ww' for c in nb_cols]].sum()
    nb = pd.DataFrame(index=out.index)
    for c in nb_cols:
        nb['nbw_' + c] = out[c + '_wv'] / out[c + '_ww'].replace(0, np.nan)
    nb = nb.reset_index()
    df = df.merge(nb, on=['district_id', 'O', 'horizon'], how='left')
    # state-level mean of seen districts' BRGI at origin (leave-one-out not needed: own value excluded below)
    sg = df.drop_duplicates(['district_id', 'O']).groupby(['state', 'O']).brgi_l1.agg(['sum', 'count']).reset_index()
    df = df.merge(sg, on=['state', 'O'], how='left')
    own = df.brgi_l1.fillna(0); ownc = df.brgi_l1.notna().astype(int)
    df['state_mean_brgi_l1'] = (df['sum'] - own) / (df['count'] - ownc).replace(0, np.nan)
    df = df.drop(columns=['sum', 'count'])
    df['region_c'] = df.region.astype('category').cat.codes
    df['state_c'] = df.state.astype('category').cat.codes
    return df
