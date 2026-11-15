"""mu_* features aimed at mule_account (record: exp/work/feature_log.md).

    python code/mule_features.py

Adds the mu_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing mu_* columns are replaced, every other column is left untouched).

Mule = money comes in from many people, then goes out in fewer, bigger payments, and almost nothing stays.
Some features look FORWARD in time (the receiver's next payment out, centred +-24h windows). That is fine
because a whole scoring period is given at once, but it means every row needs its neighbours on both sides:
  - train is computed with test as context, test with train as context (they overlap 1-10 Oct),
  - for new unseen data: add_mu_features(new_df, context=<all other known rows>).
Context rows are used only as neighbours; labels are never read.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'exp' / 'work'
HOUR, DAY = 3600, 86400
HORIZON = 7 * DAY  # how far to look for the next / previous payment
SPAN = np.int64(1 << 40)  # larger than any timestamp in seconds
RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'amount', 'balance_before']


def range_sum(ev_acct, ev_t, ev_v, q_acct, q_lo, q_hi):
    """Per query: sum of ev_v over events of the same account with q_lo <= t < q_hi."""
    codes = pd.factorize(np.r_[ev_acct, q_acct])[0].astype(np.int64)
    ec, qc = codes[:len(ev_acct)], codes[len(ev_acct):]
    o = np.lexsort((ev_t, ec))
    key = ec[o] * SPAN + ev_t[o]
    cs = np.r_[0.0, np.cumsum(np.asarray(ev_v, dtype=float)[o])]
    lo = np.searchsorted(key, qc * SPAN + q_lo, 'left')
    hi = np.searchsorted(key, qc * SPAN + q_hi, 'left')
    return cs[hi] - cs[lo]


def nearest(q_acct, q_t, ev_acct, ev_t, ev_cols, direction):
    """Per query: the closest event of the same account strictly before ('backward') or after ('forward'),
    within HORIZON (beyond it = none, so train and test see the same reach despite different data edges)."""
    q = pd.DataFrame({'k': q_acct, 't': q_t, 'i': np.arange(len(q_t))}).sort_values('t', kind='stable')
    e = pd.DataFrame({'k': ev_acct, 't2': ev_t, **ev_cols}).sort_values('t2', kind='stable')
    m = pd.merge_asof(q, e, left_on='t', right_on='t2', by='k', direction=direction,
                      allow_exact_matches=False, tolerance=HORIZON).sort_values('i')
    return m.reset_index(drop=True)


def add_mu_features(df, context=None):
    """Return df with mu_* columns added. `context` = other rows (any time) used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('mu_')])
    d = df[RAW].assign(_target=True)
    if context is not None:
        d = pd.concat([d, context[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and context'
    d = d.sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    t = d.timestamp.astype('int64').to_numpy() // 10**9
    snd, rcv = d.sender_id.to_numpy(), d.receiver_id.to_numpy()
    amt, bal = d.amount.to_numpy(), d.balance_before.to_numpy()
    p2p = (d.receiver_type == 'P2P').to_numpy()
    ones = np.ones(len(t))
    # money INTO an account = P2P rows (account = receiver); money OUT of an account = any row (account = sender)
    in_acct, in_t, in_amt, in_snd = rcv[p2p], t[p2p], amt[p2p], snd[p2p]
    f = pd.DataFrame(index=d.index, dtype=float)

    # ---- A. receiver view (P2P rows only, NaN for merchants): what the receiving account does around this payment
    r, rt = rcv[p2p], t[p2p]
    a = pd.DataFrame(index=np.flatnonzero(p2p))
    a['mu_r_in_n_24h'] = range_sum(in_acct, in_t, ones[p2p], r, rt - DAY, rt + DAY + 1)
    a['mu_r_out_n_24h'] = range_sum(snd, t, ones, r, rt - DAY, rt + DAY + 1)
    # (distinct senders in +-24h was tried: identical to mu_r_in_n_24h, mules rarely get two payments from one sender)
    a['mu_r_out_in_24h'] = a.mu_r_out_n_24h / a.mu_r_in_n_24h
    nxt = nearest(r, rt, snd, t, {'a2': amt, 'b2': bal}, 'forward')
    a['mu_fwd_gap_h'] = ((nxt.t2 - nxt.t) / HOUR).to_numpy()
    a['mu_fwd_ratio'] = (nxt.a2 / amt[p2p]).to_numpy()
    a['mu_fwd_ratio_85_100'] = a.mu_fwd_ratio.between(0.85, 1.0).astype(int)
    a['mu_fwd_left_share'] = ((nxt.b2 - nxt.a2) / nxt.b2.where(nxt.b2 > 0)).to_numpy()
    # pass-through: next payment out / everything received since the previous payment out
    prv = nearest(r, rt, snd, t, {}, 'backward').t2.fillna(pd.Series(rt - HORIZON)).to_numpy().astype(np.int64)
    nxt_t = nxt.t2.fillna(pd.Series(rt)).to_numpy().astype(np.int64)  # only used where a next payment exists
    accum = range_sum(in_acct, in_t, in_amt, r, prv + 1, nxt_t + 1)
    a['mu_fwd_over_accum'] = np.where(nxt.t2.notna(), nxt.a2.to_numpy() / accum, np.nan)
    life_in = pd.Series(in_acct).value_counts()
    life_out = pd.Series(snd).value_counts()
    a['mu_r_life_out_in'] = (pd.Series(r).map(life_out).fillna(0) / pd.Series(r).map(life_in)).to_numpy()
    f = f.join(a)

    # ---- B. sender view (all rows): money that came INTO the sender just before this payment
    f['mu_s_in_n_6h_before'] = range_sum(in_acct, in_t, ones[p2p], snd, t - 6 * HOUR, t)
    in_24h = range_sum(in_acct, in_t, in_amt, snd, t - DAY, t)
    f['mu_s_in_sum_24h_before'] = in_24h
    f['mu_s_amt_over_in_6h'] = amt / pd.Series(range_sum(in_acct, in_t, in_amt, snd, t - 6 * HOUR, t)).replace(0, np.nan).to_numpy()
    f['mu_back_gap_min'] = ((t - nearest(snd, t, in_acct, in_t, {}, 'backward').t2) / 60).to_numpy()
    f['mu_s_life_in'] = pd.Series(snd).map(life_in).fillna(0).to_numpy()

    order = ['mu_r_in_n_24h', 'mu_r_out_n_24h', 'mu_r_out_in_24h', 'mu_fwd_gap_h', 'mu_fwd_ratio',
             'mu_fwd_ratio_85_100', 'mu_fwd_left_share', 'mu_fwd_over_accum', 'mu_r_life_out_in',
             'mu_s_in_n_6h_before', 'mu_s_in_sum_24h_before', 'mu_s_amt_over_in_6h', 'mu_back_gap_min', 'mu_s_life_in']
    out = f.loc[d._target.to_numpy(), order].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('mu_')]
    te_cols = [c for c in te.columns if not c.startswith('mu_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr_new = add_mu_features(tr, context=te)
    te_new = add_mu_features(te, context=tr)

    mu = [c for c in tr_new.columns if c.startswith('mu_')]
    assert mu == [c for c in te_new.columns if c.startswith('mu_')], 'train/test mu_ columns differ'
    assert (tr_new.txn_id.to_numpy() == tr_ids).all() and (te_new.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr_new.columns if not c.startswith('mu_')] == tr_cols
    assert [c for c in te_new.columns if not c.startswith('mu_')] == te_cols
    tr_new.to_parquet(WORK / 'train_joined.parquet', index=False)
    te_new.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(mu)} mu_ features written | train {tr_new.shape} | test {te_new.shape}')
    print(mu)


if __name__ == '__main__':
    main()
