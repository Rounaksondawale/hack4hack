"""mc_* features aimed at merchant_collusion (record: exp/work/feature_log.md).

    python code/merchant_collusion_features.py

Adds the mc_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing mc_* columns are replaced, every other column is left untouched).

merchant_collusion = 2-4 accounts paying one merchant round amounts (Rs 1k/2k/2.5k/5k/10k) again and
again over 1-2 days. Merchants never send money in this data, so the B->A leg is invisible as a
payment. It shows up as an unexplained balance refill: before each fake sale the sender's balance
is topped up to 1.1 x the amount with no recorded inflow.
Every feature looks only at EARLIER rows (strictly before the row's timestamp; rows that share a
timestamp don't see each other):
  - train is computed on train alone,
  - test is computed with train as history (same as cp_/qm_/vb_).
For new unseen data later: add_mc_features(new_df, history=<all earlier rows with the raw columns>).
Merchant-side features are NaN for P2P rows. Labels are never read.
"""
import warnings

import numpy as np
import pandas as pd

from cashback_phishing_features import WORK, SPAN

RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'amount', 'balance_before',
       'ip_risk_score']
ORDER = ['mc_bal_amt_ratio', 'mc_bal_eq_1p1', 'mc_bal_eq_1p1_round', 'mc_unexplained_refill', 'mc_refill_ratio',
         'mc_r_round_24h', 'mc_r_round_nsnd_24h', 'mc_r_round_72h', 'mc_r_round_nsnd_72h',
         'mc_r_round_vol_share_72h', 'mc_r_round_new',
         'mc_pair_round_24h', 'mc_s_round_24h', 'mc_gap_prev_round_h', 'mc_s_n_merchants_round',
         'mc_amt_vs_merchant_ticket',
         'mc_night', 'mc_round_x_night', 'mc_round_x_ip']
MERCHANT_ONLY = ['mc_r_round_24h', 'mc_r_round_nsnd_24h', 'mc_r_round_72h', 'mc_r_round_nsnd_72h',
                 'mc_r_round_vol_share_72h', 'mc_r_round_new', 'mc_pair_round_24h', 'mc_amt_vs_merchant_ticket']
DAY = 86400
EVER = np.int64(1 << 39)  # window longer than any history
TICKET_K = 20             # merchant's typical ticket = median of its last 20 earlier payments


class Past:
    """Eligible rows sorted by (key, time); answers 'earlier eligible rows of the same key in [t - w, t)'."""

    def __init__(self, key, t, elig):
        self.ge, self.te = key[elig], t[elig]
        o = np.lexsort((self.te, self.ge))
        self.ge, self.te, self.idx = self.ge[o], self.te[o], np.flatnonzero(elig)[o]
        self.c = self.ge * SPAN + self.te
        self.q = key * SPAN + t  # query = every row

    def bounds(self, w):
        return np.searchsorted(self.c, self.q - w, 'left'), np.searchsorted(self.c, self.q, 'left')

    def count(self, w):
        lo, hi = self.bounds(w)
        return hi - lo

    def sum(self, v, w):
        cs = np.concatenate([[0.0], np.cumsum(v[self.idx])])
        lo, hi = self.bounds(w)
        return cs[hi] - cs[lo]

    def last(self, v):
        """Value of v at the latest earlier eligible row of the same key (NaN if none)."""
        lo, hi = self.bounds(EVER)
        ok = hi > lo
        return np.where(ok, v[self.idx[np.maximum(hi - 1, 0)]], np.nan)

    def distinct(self, other, w):
        """Distinct values of `other` among earlier eligible rows of the same key in [t - w, t)."""
        pair = pd.factorize(pd.Series(self.ge).astype(str) + '|' + pd.Series(other[self.idx]).astype(str))[0]
        prev_same = pd.Series(np.arange(len(pair))).groupby(pair).shift(1).fillna(-1).to_numpy().astype(np.int64)
        lo, hi = self.bounds(w)
        cnt = np.zeros(len(self.q), dtype=np.int64)
        for k in range(int((hi - lo).max(initial=0))):
            j = hi - 1 - k
            cnt += (j >= lo) & (prev_same[np.maximum(j, 0)] < lo)
        return cnt

    def last_k(self, v, k):
        """Matrix of v at the last k earlier eligible rows of the same key (NaN when missing)."""
        lo, hi = self.bounds(EVER)
        m = np.full((len(self.q), k), np.nan)
        for i in range(k):
            j = hi - 1 - i
            m[:, i] = np.where(j >= lo, v[self.idx[np.maximum(j, 0)]], np.nan)
        return m


def add_mc_features(df, history=None):
    """Return df with mc_* columns added. `history` = earlier rows used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('mc_')])
    d = df[RAW].assign(_target=True)
    if history is not None:
        d = pd.concat([d, history[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and history'
    d = d.sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    t = (d.timestamp.astype('int64') // 10**9).to_numpy()
    amt, bal = d.amount.to_numpy(float), d.balance_before.to_numpy(float)
    mer = (d.receiver_type == 'MERCHANT').to_numpy()
    p2p = ~mer
    rnd = ((d.amount % 500 == 0) & (d.amount >= 1000)).to_numpy()
    rnd_mer = rnd & mer
    acct = pd.factorize(pd.concat([d.sender_id, d.receiver_id]))[0].astype(np.int64)
    snd, rcv = acct[:len(d)], acct[len(d):]
    pair = pd.factorize(d.sender_id + '|' + d.receiver_id)[0].astype(np.int64)
    everyone = np.ones(len(d), bool)
    f = pd.DataFrame(index=d.index)

    # 1. hidden refill (the B->A leg)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(amt > 0, bal / amt, np.nan)
    f['mc_bal_amt_ratio'] = ratio
    f['mc_bal_eq_1p1'] = (np.abs(ratio - 1.1) <= 0.005).astype(int)
    f['mc_bal_eq_1p1_round'] = (f.mc_bal_eq_1p1.to_numpy() & rnd).astype(int)
    sends = Past(snd, t, everyone)
    t_prev, bal_prev, amt_prev = sends.last(t.astype(float)), sends.last(bal), sends.last(amt)
    inflow = Past(rcv, t, p2p)  # money received as a P2P receiver
    cs = np.concatenate([[0.0], np.cumsum(amt[inflow.idx])])
    has_prev = ~np.isnan(t_prev)
    tp = np.where(has_prev, t_prev, 0).astype(np.int64)
    lo = np.searchsorted(inflow.c, snd * SPAN + tp, 'left')
    hi = np.searchsorted(inflow.c, snd * SPAN + t, 'left')
    received = cs[hi] - cs[lo]
    refill = np.where(has_prev, bal - (bal_prev - amt_prev) - received, np.nan)
    f['mc_unexplained_refill'] = refill
    with np.errstate(invalid='ignore', divide='ignore'):
        f['mc_refill_ratio'] = np.where(amt > 0, refill / amt, np.nan)

    # 2. ring burst on the merchant
    r_round = Past(rcv, t, rnd_mer)
    f['mc_r_round_24h'] = r_round.count(DAY)
    f['mc_r_round_nsnd_24h'] = r_round.distinct(snd, DAY)
    f['mc_r_round_72h'] = r_round.count(3 * DAY)
    f['mc_r_round_nsnd_72h'] = r_round.distinct(snd, 3 * DAY)
    vol_all = Past(rcv, t, mer).sum(amt, 3 * DAY)
    with np.errstate(invalid='ignore', divide='ignore'):
        f['mc_r_round_vol_share_72h'] = np.where(vol_all > 0, r_round.sum(amt, 3 * DAY) / vol_all, np.nan)
    before_window = r_round.count(EVER) - f.mc_r_round_72h.to_numpy()
    f['mc_r_round_new'] = ((f.mc_r_round_72h.to_numpy() > 0) & (before_window == 0)).astype(int)

    # 3. sender side
    f['mc_pair_round_24h'] = Past(pair, t, rnd_mer).count(DAY)
    s_round = Past(snd, t, rnd_mer)
    f['mc_s_round_24h'] = s_round.count(DAY)
    f['mc_gap_prev_round_h'] = (t - s_round.last(t.astype(float))) / 3600
    f['mc_s_n_merchants_round'] = s_round.distinct(rcv, EVER)
    with warnings.catch_warnings(), np.errstate(invalid='ignore', divide='ignore'):
        warnings.simplefilter('ignore', RuntimeWarning)
        ticket = np.nanmedian(Past(rcv, t, mer).last_k(amt, TICKET_K), axis=1)
        f['mc_amt_vs_merchant_ticket'] = np.where(ticket > 0, amt / ticket, np.nan)

    # 4. context
    night = (d.timestamp.dt.hour < 6).to_numpy()
    f['mc_night'] = night.astype(int)
    f['mc_round_x_night'] = (rnd & night).astype(int)
    f['mc_round_x_ip'] = np.where(rnd, d.ip_risk_score, 0.0)

    f[MERCHANT_ONLY] = f[MERCHANT_ONLY].astype(float)
    f.loc[~mer, MERCHANT_ONLY] = np.nan
    out = f.loc[d._target.to_numpy(), ORDER].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('mc_')]
    te_cols = [c for c in te.columns if not c.startswith('mc_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr = add_mc_features(tr)
    te = add_mc_features(te, history=tr)

    mc = [c for c in tr.columns if c.startswith('mc_')]
    assert mc == ORDER == [c for c in te.columns if c.startswith('mc_')], 'train/test mc_ columns differ'
    assert (tr.txn_id.to_numpy() == tr_ids).all() and (te.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr.columns if not c.startswith('mc_')] == tr_cols
    assert [c for c in te.columns if not c.startswith('mc_')] == te_cols
    tr.to_parquet(WORK / 'train_joined.parquet', index=False)
    te.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(mc)} mc_ features written | train {tr.shape} | test {te.shape}')
    print(mc)


if __name__ == '__main__':
    main()
