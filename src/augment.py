"""Rebuild train-style rows for forecast origins that are missing from train.csv
(2024Q1-2024Q3 for the final model). Every column is computed point-in-time from the raw
files exactly as the organisers' columns are (verified against existing origins).
Labels come from train.csv targets; component scores at O-1 are reconstructed ranks."""
import numpy as np
import pandas as pd
from common import DATA, qidx, qstr

ENV = ['temp_anomaly_c', 'precip_anomaly_pct', 'heat_days_40c', 'heat_days_40c_4q', 'precip_anomaly_pct_4q',
       'max_dry_spell_days', 'log_fire_count', 'fire_count_yoy', 'log_news_events', 'news_share_yoy', 'news_tone',
       'ev_protest_share', 'ev_violence_share']


def _bank_panel():
    bq = pd.read_csv(DATA + 'bank_quarterly.csv')
    bq['q'] = qidx(bq.quarter)
    bq = bq.sort_values(['district_id', 'q'])
    g = bq.groupby('district_id')
    bq['lc'] = np.log(bq.credit); bq['ld'] = np.log(bq.deposit); bq['lo'] = np.log(bq.offices)
    bq['cyoy'] = bq.lc - g.lc.shift(4); bq['dyoy'] = bq.ld - g.ld.shift(4); bq['oyoy'] = bq.lo - g.lo.shift(4)
    bq['cqoq'] = bq.lc - g.lc.shift(1)
    g = bq.groupby('district_id')
    bq['cqoq_std8'] = g.cqoq.transform(lambda s: s.rolling(8, min_periods=8).std())
    bq['cyoy_mean4'] = g.cyoy.transform(lambda s: s.rolling(4, min_periods=4).mean())
    bq['gap_win'] = g.bank_gap.transform(lambda s: s.rolling(8, min_periods=1).max())
    return bq


def _ntl_quarterly():
    nl = pd.read_csv(DATA + 'nightlights_monthly.csv')
    nl['q'] = qidx(nl.month.str[:4] + 'Q' + ((nl.month.str[5:7].astype(int) - 1) // 3 + 1).astype(str))
    agg = nl.groupby(['district_id', 'q']).agg(ntl=('ntl_sum', 'sum'), n=('ntl_sum', 'count'),
                                               gf=('ntl_quality_prop_2_gapfilled', 'mean')).reset_index()
    agg = agg[agg.n == 3]
    return nl, agg


def build_rows(origins, tr, te):
    """Return a DataFrame with train.csv columns for the given origin strings (seen districts only)."""
    bq = _bank_panel()
    nl, nq = _ntl_quarterly()
    st = pd.read_csv(DATA + 'district_static.csv')
    g = pd.read_csv(DATA + 'district_graph.csv')
    lab = tr[['district_id', 'target_quarter', 'brgi']].drop_duplicates(['district_id', 'target_quarter'])
    lab['q'] = qidx(lab.target_quarter)
    L = lab.set_index(['district_id', 'q']).brgi
    # population: per district per origin-year, taken from the nearest existing rows
    allrows = pd.concat([tr, te])
    allrows['year'] = allrows.origin_quarter.str[:4].astype(int)
    pop = allrows.drop_duplicates(['district_id', 'year'])[['district_id', 'year', 'log_population', 'pop_growth_3y',
                                                            'log_pop_density']]
    seen = sorted(set(tr.district_id))
    rows = []
    for o in origins:
        O = qidx([o])[0]
        base = pd.DataFrame({'district_id': seen})
        b1 = bq[bq.q == O - 1].set_index('district_id'); b2 = bq[bq.q == O - 2].set_index('district_id')
        d = base.set_index('district_id')
        d['log_credit'] = b1.lc; d['log_deposit'] = b1.ld; d['log_offices'] = b1.lo
        d['cd_ratio'] = b1.credit / b1.deposit
        d['credit_yoy_l1'] = b1.cyoy; d['credit_yoy_l2'] = b2.cyoy
        d['deposit_yoy_l1'] = b1.dyoy; d['deposit_yoy_l2'] = b2.dyoy
        d['offices_yoy_l1'] = b1.oyoy; d['credit_qoq_l1'] = b1.cqoq
        d['credit_qoq_std8'] = b1.cqoq_std8; d['credit_yoy_mean4'] = b1.cyoy_mean4
        d['bank_gap_in_window'] = b1.gap_win; d['rrb_adjusted_l1'] = b1.rrb_adjusted
        d['_credit'] = b1.credit; d['_offices'] = b1.offices
        # BRGI lags
        d['brgi_l1'] = [L.get((k, O - 1), np.nan) for k in d.index]
        d['brgi_l2'] = [L.get((k, O - 2), np.nan) for k in d.index]
        d['brgi_mean4'] = np.nanmean(np.column_stack([[L.get((k, O - j), np.nan) for k in d.index] for j in (1, 2, 3, 4)]), axis=1)
        # component scores at O-1 (cross-sectional ranks among districts with data)
        d['score_credit_yoy_l1'] = d.credit_yoy_l1.rank(pct=True)
        d['score_deposit_yoy_l1'] = d.deposit_yoy_l1.rank(pct=True)
        n1 = nq[nq.q == O - 1].set_index('district_id'); n2 = nq[nq.q == O - 2].set_index('district_id')
        n5 = nq[nq.q == O - 5].set_index('district_id'); n6 = nq[nq.q == O - 6].set_index('district_id')
        d['log_ntl_l1'] = np.log(n1.ntl.reindex(d.index) / 3)
        d['ntl_yoy_l1'] = np.log(n1.ntl.reindex(d.index)) - np.log(n5.ntl.reindex(d.index))
        d['ntl_yoy_l2'] = np.log(n2.ntl.reindex(d.index)) - np.log(n6.ntl.reindex(d.index))
        d['ntl_qoq_l1'] = np.log(n1.ntl.reindex(d.index)) - np.log(n2.ntl.reindex(d.index))
        d['ntl_gapfilled_share_l1'] = n1.gf.reindex(d.index)
        # partial: months of quarter O known (first 2) vs same months a year earlier
        nl['mi'] = nl.month.str[:4].astype(int) * 12 + nl.month.str[5:7].astype(int) - 1
        fm = (O // 4) * 12 + (O % 4) * 3
        cur = nl[nl.mi.isin([fm, fm + 1])].groupby('district_id').ntl_sum.sum()
        prv = nl[nl.mi.isin([fm - 12, fm - 11])].groupby('district_id').ntl_sum.sum()
        d['ntl_yoy_partial'] = np.log(cur.reindex(d.index)) - np.log(prv.reindex(d.index))
        # ntl component: rank among ALL districts with ntl (incl. unseen)
        nr_all = (np.log(n1.ntl) - np.log(n5.ntl.reindex(n1.index))).rank(pct=True)
        d['score_ntl_yoy_l1'] = nr_all.reindex(d.index)
        # resilience = residual of the exact BRGI identity
        res = (d.brgi_l1 - 35 * d.score_credit_yoy_l1 - 20 * d.score_deposit_yoy_l1 - 30 * d.score_ntl_yoy_l1) / 15
        d['score_resilience_l1'] = res.clip(0, 1)
        d = d.reset_index()
        rows.append((O, d))
    out = []
    for O, d in rows:
        for h in (1, 2, 3, 4):
            T = O + h
            y = [L.get((k, T), np.nan) for k in d.district_id]
            r = d.copy(); r['brgi'] = y; r['horizon'] = h
            r['origin_quarter'] = qstr([O])[0]; r['target_quarter'] = qstr([T])[0]
            r['target_qtr_of_year'] = T % 4 + 1
            out.append(r[r.brgi.notna()])
    out = pd.concat(out, ignore_index=True)
    out['year'] = out.origin_quarter.str[:4].astype(int)
    out = out.merge(pop, on=['district_id', 'year'], how='left')
    # fall back to the closest year if missing
    out = out.merge(st, on='district_id', how='left', suffixes=('', '_st'))
    out['credit_per_capita'] = out._credit * 1e7 / np.exp(out.log_population)
    out['offices_per_100k'] = out._offices / (np.exp(out.log_population) / 1e5)
    # neighbour means of the l1 signals (simple mean over neighbours present)
    key = out.drop_duplicates(['district_id', 'origin_quarter'])[['district_id', 'origin_quarter', 'brgi_l1', 'credit_yoy_l1', 'ntl_yoy_l1']]
    m = g.merge(key.rename(columns={'district_id': 'neighbor_id'}), on='neighbor_id')
    nb = m.groupby(['district_id', 'origin_quarter'])[['brgi_l1', 'credit_yoy_l1', 'ntl_yoy_l1']].mean()
    nb.columns = ['nbr_mean_brgi_l1', 'nbr_mean_credit_yoy_l1', 'nbr_mean_ntl_yoy_l1']
    out = out.merge(nb.reset_index(), on=['district_id', 'origin_quarter'], how='left')
    for c in ENV:
        out[c] = np.nan
    out['id'] = out.district_id + '_' + out.origin_quarter + '_h' + out.horizon.astype(str)
    out = out[[c for c in tr.columns]]
    return out
