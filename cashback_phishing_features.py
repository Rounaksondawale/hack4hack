"""cp_* features aimed at fake_cashback and phishing_collect (record: exp/work/feature_log.md).

    python code/cashback_phishing_features.py

Adds the cp_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing cp_* columns are replaced, every other column is left untouched).

Every history feature looks only at EARLIER rows (strictly before the row's timestamp), so:
  - train is computed on train alone,
  - test is computed with train as history (test = Oct, train = Jul-Sep), exactly like unseen data.
For new unseen data later: add_cp_features(new_df, history=<all earlier rows with the raw columns>).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'exp' / 'work'
MIN, HOUR, DAY = 60, 3600, 86400
SPAN = np.int64(1 << 40)  # larger than any timestamp in seconds
RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'amount', 'txn_type',
       'new_payee', 'r_kyc_level']
ROUND_LIMITS = np.array([500, 1000, 2000, 5000, 10000, 20000, 25000, 50000, 100000])


def past_sum(group, t, v, w):
    """Per row: sum of v over rows of the same group with t - w <= t_j < t (earlier rows only)."""
    g = pd.factorize(group)[0].astype(np.int64)
    order = np.lexsort((t, g))
    key = g[order] * SPAN + t[order]
    cs = np.r_[0.0, np.cumsum(np.asarray(v, dtype=float)[order])]
    lo = np.searchsorted(key, key - w, 'left')
    hi = np.searchsorted(key, key, 'left')  # excludes the row itself and same-second rows
    out = np.empty(len(t))
    out[order] = cs[hi] - cs[lo]
    return out


def last_before(q_key, q_t, ref_key, ref_t, ref_amt):
    """Per query row: (seconds since, amount of) the latest ref row with the same key strictly before q_t."""
    q = pd.DataFrame({'k': q_key, 't': q_t, 'i': np.arange(len(q_t))}).sort_values('t', kind='stable')
    r = pd.DataFrame({'k': ref_key, 't2': ref_t, 'a2': ref_amt}).sort_values('t2', kind='stable')
    m = pd.merge_asof(q, r, left_on='t', right_on='t2', by='k', direction='backward', allow_exact_matches=False)
    gap, amt = np.full(len(q_t), np.nan), np.full(len(q_t), np.nan)
    gap[m.i] = (m.t - m.t2).to_numpy()
    amt[m.i] = m.a2.to_numpy()
    return gap, amt


def add_cp_features(df, history=None):
    """Return df with cp_* columns added. `history` = earlier rows used only as context."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('cp_')])
    d = df[RAW].assign(_target=True)
    if history is not None:
        d = pd.concat([d, history[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and history'
    d = d.sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    t = d.timestamp.astype('int64').to_numpy() // 10**9
    snd, rcv, amt = d.sender_id.to_numpy(), d.receiver_id.to_numpy(), d.amount.to_numpy()
    p2p = (d.receiver_type == 'P2P').to_numpy()
    collect = (d.txn_type == 'COLLECT').to_numpy()
    new_payee = d.new_payee.to_numpy() == 1
    f = pd.DataFrame(index=d.index)

    # 1. collect requests
    f['cp_is_collect'] = collect.astype(int)  # every COLLECT is P2P in this data, so no collect x p2p flag
    f['cp_collect_new_payee'] = (collect & new_payee).astype(int)

    # 2. amount shape (fake_cashback bait = Rs 1-10; phishing = just under a round limit)
    f['cp_log_amt'] = np.log1p(amt)
    f['cp_whole_rupee'] = (np.round(amt * 100) % 100 == 0).astype(int)
    f['cp_amt_tiny'] = (amt <= 10).astype(int)
    f['cp_amt_large'] = (amt > 5000).astype(int)
    lim = ROUND_LIMITS[np.minimum(np.searchsorted(ROUND_LIMITS, amt, 'right'), len(ROUND_LIMITS) - 1)]
    f['cp_under_round_gap'] = np.where(amt < lim, (lim - amt) / lim, np.nan)  # 0.001 = 0.1% below the limit
    f['cp_just_under_round'] = (f.cp_under_round_gap <= 0.02).astype(int)

    # 3. (no ip_risk_score features: it drifts Sep -> test, see prepare_data.py step 3)

    # 4. sender -> receiver pair history
    pair = pd.Series(snd).str.cat(pd.Series(rcv), sep='>').to_numpy()
    first_pair = ~pd.Series(pair).duplicated().to_numpy()
    f['cp_pair_n_prior'] = pd.Series(pair).groupby(pair).cumcount().to_numpy()
    f['cp_pair_gap_h'] = last_before(pair, t, pair, t, amt)[0] / HOUR
    f['cp_first_pair'] = first_pair.astype(int)
    f['cp_first_pair_not_new_payee'] = (first_pair & ~new_payee).astype(int)

    # 5. reverse flow (fake_cashback): did the receiver pay this sender before? tiny "cashback" first?
    rev = pd.Series(rcv).str.cat(pd.Series(snd), sep='>').to_numpy()  # key of the opposite direction
    gap, ramt = last_before(rev, t, pair, t, amt)
    f['cp_rev_gap_h'] = gap / HOUR
    f['cp_rev_amt'] = ramt
    f['cp_rev_ratio'] = amt / ramt
    tiny = amt <= 10
    f['cp_rev_tiny_gap_h'] = last_before(rev, t, pair[tiny], t[tiny], amt[tiny])[0] / HOUR

    # 6. receiver profile
    f['cp_r_n_prior'] = pd.Series(rcv).groupby(rcv).cumcount().to_numpy()
    first_send = pd.Series(t).groupby(snd).min()
    f['cp_r_has_sent'] = (pd.Series(rcv).map(first_send).to_numpy() < t).astype(int)
    f['cp_r_kyc_min'] = (d.r_kyc_level == 'min').astype(int).to_numpy()

    # 7. sender bursts
    f['cp_snd_newpayee_1h'] = past_sum(snd, t, new_payee, HOUR)

    # 8. receiver bursts of first-time senders, P2P receivers only (0 for merchants)
    for name, w in (('5m', 5 * MIN), ('1h', HOUR), ('1d', DAY)):
        f[f'cp_r_newsnd_{name}'] = past_sum(rcv, t, first_pair, w) * p2p
    week_rate = past_sum(rcv, t, np.ones(len(t)), 7 * DAY) / (7 * 24)
    f['cp_r_burst_ratio_1h'] = past_sum(rcv, t, np.ones(len(t)), HOUR) / (week_rate + 1 / (7 * 24)) * p2p

    out = f[d._target.to_numpy()].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('cp_')]
    te_cols = [c for c in te.columns if not c.startswith('cp_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr = add_cp_features(tr)
    te = add_cp_features(te, history=tr)

    cp = [c for c in tr.columns if c.startswith('cp_')]
    assert cp == [c for c in te.columns if c.startswith('cp_')], 'train/test cp_ columns differ'
    assert (tr.txn_id.to_numpy() == tr_ids).all() and (te.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr.columns if not c.startswith('cp_')] == tr_cols
    assert [c for c in te.columns if not c.startswith('cp_')] == te_cols
    tr.to_parquet(WORK / 'train_joined.parquet', index=False)
    te.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(cp)} cp_ features written | train {tr.shape} | test {te.shape}')
    print(cp)


if __name__ == '__main__':
    main()
