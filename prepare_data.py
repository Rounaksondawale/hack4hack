"""Build model-ready features for the UPI fraud-rings challenge.

    python prepare_data.py             # writes features/train.parquet and features/test.parquet
    python prepare_data.py --check     # prints September-vs-test drift instead of writing files
    python prepare_data.py --keep 0.7  # share of train 'legit' rows to keep (step 6)

Steps (see the data analysis): 1 load, 2 combine train+test, 3 drop drifting columns,
4 single-row features, 5 features linking rows, 6 thin train 'legit' rows to test's density,
8 fold column (`period`), 9 write + verify.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / '01_raw'  # train.csv, test.csv, accounts.csv, merchants.csv, sample_submission.csv
OUT = ROOT / 'data' / '02_processed' / 'features'
MIN, HOUR, DAY = 60, 3600, 86400
SPAN = np.int64(1 << 25)  # larger than the data's time range in seconds (~131 days)
LINK_PREFIXES = ('rcv_', 'dev_', 'snd_', 'pair_', 'fwd_', 'back_', 'chain_', 'collect_')
KEY_COLS = ['sender_id', 'receiver_id', 'device_id']  # for label-history features in train_model.py, not features


def log(msg, t0=time.time()):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


# ---------------------------------------------------------------- steps 1-2, 6
def load(keep, seed=0):
    tr = pd.read_csv(DATA / 'train.csv')
    te = pd.read_csv(DATA / 'test.csv')
    ac = pd.read_csv(DATA / 'accounts.csv')
    mr = pd.read_csv(DATA / 'merchants.csv')
    assert len(tr) == 1_564_060 and len(te) == 260_582
    assert tr.txn_id.is_unique and te.txn_id.is_unique and not set(tr.txn_id) & set(te.txn_id)
    tr['is_test'], te['is_test'] = False, True
    df = pd.concat([tr, te], ignore_index=True)

    # Step 6: test has ~half the traffic per day, fraud bursts unchanged -> drop train 'legit' rows
    # completely (as training rows and as context for the linking features).
    r = np.random.default_rng(seed).random(len(df))
    df = df[~(~df.is_test & (df.label == 'legit') & (r >= keep))].reset_index(drop=True)

    # Fold / period: Jul, Aug, Sep (incl. the October tails of September episodes), test.
    month = df.timestamp.str[:7]
    df['period'] = np.where(df.is_test, 'test', np.where(month == '2026-10', '2026-09', month))
    return df, ac, mr


# ---------------------------------------------------------------- window helper
class Windows:
    """Centred time windows: for each row, the rows of the same group with |t_j - t_i| <= w."""

    def __init__(self, group, t):
        g = pd.factorize(np.asarray(group))[0].astype(np.int64)
        self.order = np.lexsort((t, g))
        self.t = t[self.order]
        self.g = g[self.order]
        self.key = self.g * SPAN + self.t
        self.n = len(t)

    def _bounds(self, w):
        return (np.searchsorted(self.key, self.key - w, 'left'),
                np.searchsorted(self.key, self.key + w, 'right'))

    def _back(self, x):
        out = np.empty(self.n, dtype=x.dtype)
        out[self.order] = x
        return out

    def _segments(self, lo, hi, chunk=100_000):
        """Yield (first row, segment starts, window positions, row of each position) in chunks."""
        for s in range(0, self.n, chunk):
            l, h = lo[s:s + chunk], hi[s:s + chunk]
            k = h - l
            starts = np.r_[0, np.cumsum(k)[:-1]]
            pos = np.arange(k.sum()) - np.repeat(starts, k) + np.repeat(l, k)
            yield s, starts, pos, np.repeat(np.arange(s, s + len(l)), k)

    def count(self, w):
        lo, hi = self._bounds(w)
        return self._back(hi - lo)

    def sum(self, v, w):
        lo, hi = self._bounds(w)
        cs = np.r_[0.0, np.cumsum(np.asarray(v, dtype=float)[self.order])]
        return self._back(cs[hi] - cs[lo])

    def min(self, v, w):
        lo, hi = self._bounds(w)
        vs = np.asarray(v, dtype=float)[self.order]
        out = np.empty(self.n)
        for s, starts, pos, _ in self._segments(lo, hi):
            out[s:s + len(starts)] = np.minimum.reduceat(vs[pos], starts)
        return self._back(out)

    def distinct(self, other, w):
        """Distinct values of `other` in the window = rows whose previous same-pair row is before it."""
        o = pd.factorize(np.asarray(other))[0].astype(np.int64)[self.order]
        idx = np.lexsort((self.t, o, self.g))
        same = np.r_[False, (self.g[idx][1:] == self.g[idx][:-1]) & (o[idx][1:] == o[idx][:-1])]
        prev = np.empty(self.n, dtype=np.int64)
        prev[idx] = np.where(same, np.r_[0, self.t[idx][:-1]], -10**12)
        lo, hi = self._bounds(w)
        out = np.empty(self.n, dtype=np.int64)
        for s, starts, pos, rows in self._segments(lo, hi):
            out[s:s + len(starts)] = np.add.reduceat((prev[pos] < self.t[rows] - w).astype(np.int64), starts)
        return self._back(out)


def cross_count(target_g, target_t, query_g, query_t, before, after):
    """For each query row: target rows of the same group with t in [query_t - before, query_t + after]."""
    keys = np.sort(target_g * SPAN + target_t)
    q = query_g * SPAN + query_t
    return np.searchsorted(keys, q + after, 'right') - np.searchsorted(keys, q - before, 'left')


# ---------------------------------------------------------------- money moving on
def money_links(df, t):
    n = len(df)
    idx = np.arange(n)
    amt = df.amount.to_numpy()
    p2p = (df.receiver_type == 'P2P').to_numpy()
    snd, rcv = df.sender_id.to_numpy(), df.receiver_id.to_numpy()
    res = pd.DataFrame(index=idx, columns=['fwd_mins', 'fwd_ratio', 'back_mins', 'back_ratio'], dtype=float)

    # forward: the receiver's next payment within 30 min after this one
    q = pd.DataFrame({'t': t[p2p], 'acct': rcv[p2p], 'amt': amt[p2p], 'i': idx[p2p]}).sort_values('t', kind='stable')
    out = pd.DataFrame({'t2': t, 'acct': snd, 'amt2': amt, 'j': idx}).sort_values('t2', kind='stable')
    f = pd.merge_asof(q, out, left_on='t', right_on='t2', by='acct', direction='forward',
                      allow_exact_matches=False, tolerance=30 * MIN)
    res.loc[f.i, 'fwd_mins'] = ((f.t2 - f.t) / 60).to_numpy()
    res.loc[f.i, 'fwd_ratio'] = (f.amt2 / f.amt).to_numpy()

    # backward: the sender's last incoming payment within 30 min before this one
    q = pd.DataFrame({'t': t, 'acct': snd, 'amt': amt, 'i': idx}).sort_values('t', kind='stable')
    inc = pd.DataFrame({'t2': t[p2p], 'acct': rcv[p2p], 'amt2': amt[p2p], 'j': idx[p2p]}).sort_values('t2', kind='stable')
    b = pd.merge_asof(q, inc, left_on='t', right_on='t2', by='acct', direction='backward',
                      allow_exact_matches=False, tolerance=30 * MIN)
    res.loc[b.i, 'back_mins'] = ((b.t - b.t2) / 60).to_numpy()
    res.loc[b.i, 'back_ratio'] = (b.amt / b.amt2).to_numpy()

    # chain length: follow links where 85-100% of the amount moves on
    nxt, prv = np.full(n, -1), np.full(n, -1)
    ok = f.amt2.between(0.85 * f.amt, f.amt)
    nxt[f.i[ok]] = f.j[ok].astype(np.int64)
    ok = b.amt.between(0.85 * b.amt2, b.amt2)
    prv[b.i[ok]] = b.j[ok].astype(np.int64)
    down, up = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for _ in range(10):
        down = np.where(nxt >= 0, 1 + down[nxt], 0)
        up = np.where(prv >= 0, 1 + up[prv], 0)
    res['chain_len'] = 1 + up + down
    return res


# ---------------------------------------------------------------- steps 3-5
def build_features(df, ac, mr):
    n = len(df)
    ts = pd.to_datetime(df.timestamp)
    t = (ts - pd.Timestamp('2026-07-01')).dt.total_seconds().to_numpy().astype(np.int64)
    assert t.min() >= 0 and t.max() + 7 * DAY < SPAN
    amount = df.amount.to_numpy()
    acc = ac.set_index('account_id')
    mer = mr.set_index('merchant_id')
    X = pd.DataFrame(index=df.index)

    # Step 3: app_version, ip_risk_score, txn_id and absolute dates are never used as features.
    # Step 4: single-row features.
    X['hour'] = ts.dt.hour
    X['weekday'] = ts.dt.weekday
    X['night'] = (X.hour < 6).astype(np.int8)
    X['evening'] = (X.hour >= 19).astype(np.int8)
    for c in ['amount', 'sim_age_hours', 'device_age_days', 'new_payee', 'pin_retries', 'login_to_txn_sec', 'balance_before']:
        X[c] = df[c]
    X['log_amount'] = np.log1p(amount)
    X['has_paise'] = (np.round(amount * 100) % 100 != 0).astype(np.int8)
    for k in (10, 100, 500):
        X[f'mult{k}'] = (amount % k == 0).astype(np.int8)
    X['just_under_1000'] = ((amount >= 950) & (amount % 1000 >= 950)).astype(np.int8)
    X['amount_to_balance'] = amount / df.balance_before
    X['amount_pct_in_period'] = df.amount.groupby(df.period).rank(pct=True)
    dev_num = df.device_id.str[2:].astype(np.int64)
    own = dev_num == df.sender_id.str[2:].astype(np.int64)
    X['phone_kind'] = np.select([own, dev_num >= 420000], [0, 1], 2).astype(np.int8)  # own / no owner / other account's
    X['sim_fresh'] = (df.sim_age_hours < 48).astype(np.int8)
    X['device_new'] = (df.device_age_days < 2).astype(np.int8)
    X['away_from_home'] = (df.device_city != df.sender_id.map(acc.home_city)).astype(np.int8)
    X['log_login'] = np.log1p(df.login_to_txn_sec)
    X['qr_state'] = np.select([df.qr_vpa_mismatch.isna(), df.qr_vpa_mismatch == 0], [0, 1], 2).astype(np.int8)
    X['sender_min_kyc'] = (df.sender_id.map(acc.kyc_level) == 'min').astype(np.int8)
    X['sender_ios'] = (df.sender_id.map(acc.os) == 'ios').astype(np.int8)
    X['bank_code'] = df.sender_id.map(acc.bank_code)
    X['receiver_min_kyc'] = df.receiver_id.map(acc.kyc_level).map({'min': 1.0, 'full': 0.0})  # NaN for merchants
    X['merchant_age_days'] = (ts.dt.normalize() - pd.to_datetime(df.receiver_id.map(mer.onboarded_on))).dt.days
    for c in ['receiver_type', 'txn_type', 'merchant_category', 'app']:
        X[c] = df[c].astype('category')
    X['bank_code'] = X.bank_code.astype('category')
    log('single-row features done')

    # Step 5: features linking rows (centred windows, train + test together).
    new_payee = df.new_payee.to_numpy()
    rcv = Windows(df.receiver_id, t)
    for name, w in [('1h', HOUR), ('6h', 6 * HOUR), ('24h', DAY), ('72h', 3 * DAY)]:
        X[f'rcv_n_{name}'] = rcv.count(w)
        X[f'rcv_nsend_{name}'] = rcv.distinct(df.sender_id, w)
    X['rcv_newpayee_share_6h'] = rcv.sum(new_payee, 6 * HOUR) / X.rcv_n_6h
    X['rcv_paise_share_6h'] = rcv.sum(X.has_paise, 6 * HOUR) / X.rcv_n_6h
    X['rcv_round_share_24h'] = rcv.sum(X.mult10, DAY) / X.rcv_n_24h
    n_checked = rcv.sum(df.qr_vpa_mismatch.notna(), DAY)
    X['rcv_qr_mismatch_rate_24h'] = np.where(n_checked > 0, rcv.sum(df.qr_vpa_mismatch.fillna(0), DAY) / np.maximum(n_checked, 1), np.nan)
    is_collect = (df.txn_type == 'COLLECT').to_numpy()
    X['rcv_collect_24h'] = rcv.sum(is_collect, DAY)
    X['collect_payers_24h'] = np.nan
    X.loc[is_collect, 'collect_payers_24h'] = Windows(df.receiver_id[is_collect], t[is_collect]).distinct(df.sender_id[is_collect], DAY)
    log('receiver features done')

    X = X.join(money_links(df, t))
    log('money-moving-on features done')

    dev = Windows(df.device_id, t)
    X['dev_n_10m'] = dev.count(10 * MIN)
    X['dev_n_1h'] = dev.count(HOUR)
    X['dev_nsend_all'] = df.groupby('device_id').sender_id.transform('nunique')
    X['dev_nsend_24h'] = dev.distinct(df.sender_id, DAY)
    X['dev_nrcv_24h'] = dev.distinct(df.receiver_id, DAY)
    X['dev_login_1h'] = dev.sum(X.log_login, HOUR) / X.dev_n_1h
    log('phone features done')

    snd = Windows(df.sender_id, t)
    X['snd_n_6h'] = snd.count(6 * HOUR)
    X['snd_n_72h'] = snd.count(3 * DAY)
    X['snd_newpayee_6h'] = snd.sum(new_payee, 6 * HOUR)
    X['snd_out_6h_to_balance'] = snd.sum(amount, 6 * HOUR) / df.balance_before
    X['snd_min_sim_6h'] = snd.min(df.sim_age_hours, 6 * HOUR)
    X['snd_min_devage_6h'] = snd.min(df.device_age_days, 6 * HOUR)
    X['snd_unowned_6h'] = snd.sum(X.phone_kind == 1, 6 * HOUR)
    is_refund = pd.Series((df.txn_type == 'REFUND').to_numpy(), index=df.index)
    X['snd_refund_7d'] = snd.sum(is_refund, 7 * DAY)
    X['snd_refund_period'] = is_refund.groupby([df.sender_id, df.period]).transform('sum')
    merch = df[is_refund].groupby(['sender_id', 'period']).receiver_id.nunique().rename('v').reset_index()
    X['snd_refund_merchants_period'] = df[['sender_id', 'period']].merge(merch, how='left', on=['sender_id', 'period']).v.fillna(0).to_numpy()
    log('sender features done')

    ids = pd.factorize(pd.concat([df.sender_id, df.receiver_id], ignore_index=True))[0].astype(np.int64)
    s, r = ids[:n], ids[n:]
    pair_raw, rev_raw = s * 1_000_000 + r, r * 1_000_000 + s
    X['pair_n_all'] = pd.Series(pair_raw).groupby(pair_raw).transform('size').to_numpy()
    X['pair_reverse_ever'] = np.isin(rev_raw, pair_raw).astype(np.int8)
    codes = pd.factorize(np.r_[pair_raw, rev_raw])[0].astype(np.int64)
    pc, rc = codes[:n], codes[n:]
    small = amount <= 10
    small_keys = np.sort(pc[small] * SPAN + t[small])
    q = rc * SPAN + t
    X['pair_reverse_small_2h'] = np.searchsorted(small_keys, q + 2 * HOUR, 'right') - np.searchsorted(small_keys, q - 2 * HOUR, 'left')
    X['pair_n_24h'] = Windows(pc, t).count(DAY)
    X['pair_share_of_rcv_24h'] = X.pair_n_24h / X.rcv_n_24h
    log('pair features done')

    # Money into this sender (as the receiver of P2P payments) and out of this receiver.
    p2p = (df.receiver_type == 'P2P').to_numpy()
    X['snd_in_24h'] = cross_count(r[p2p], t[p2p], s, t, DAY, DAY)
    X['snd_in_72h_before'] = cross_count(r[p2p], t[p2p], s, t, 3 * DAY, 0)
    X['rcv_out_72h_after'] = np.where(p2p, cross_count(s, t, r, t, 0, 3 * DAY), np.nan)
    log('money in/out features done')
    return X, df[KEY_COLS]


# ---------------------------------------------------------------- checks
def drift_report(X, period):
    """Step 3/6 check: per feature, how well does its value tell September rows from test rows?

    AUC 0.5 = same distribution; the further from 0.5, the more the feature drifts.
    """
    m = period.isin(['2026-09', 'test']).to_numpy()
    y = (period[m] == 'test').to_numpy()
    n_pos, n_neg = float(y.sum()), float((~y).sum())  # float: numpy 1.x on Windows sums bools as int32
    Xm = X[m]
    drift = {}
    for c in Xm:
        v = Xm[c].cat.codes if Xm[c].dtype.name == 'category' else Xm[c].fillna(-1)
        ranks = v.rank().to_numpy()  # Mann-Whitney: AUC from average ranks
        drift[c] = abs((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) - 0.5)
    drift = pd.Series(drift).sort_values(ascending=False)
    link = drift[drift.index.str.startswith(LINK_PREFIXES)]
    print('\n|AUC - 0.5| of September vs test, most drifting first:')
    print(drift.head(15).round(3).to_string())
    print(f'\nMean |AUC - 0.5|: linking features {link.mean():.4f}, single-row features {drift.drop(link.index).mean():.4f}')


def sanity_report(X, df):
    """Each feature should separate the fraud type it was built for (train rows only)."""
    tr = ~df.is_test.to_numpy()
    lab = df.label[tr]
    p2p = (df.receiver_type == 'P2P')[tr]
    chk = pd.DataFrame({
        'phone_no_owner': X.phone_kind[tr] == 1,
        'sim_fresh': X.sim_fresh[tr],
        'p2p_rcv_nsend_6h>=3': (X.rcv_nsend_6h[tr] >= 3).astype(float).where(p2p),
        'fwd_85_100pct': X.fwd_ratio[tr].between(0.85, 1.0),
        'chain_len>=2': X.chain_len[tr] >= 2,
        'dev_nsend_all>=3': X.dev_nsend_all[tr] >= 3,
        'snd_refund_period>=2': X.snd_refund_period[tr] >= 2,
        'merchant_round_share_24h': X.rcv_round_share_24h[tr].where(~p2p),
        'just_under_1000': X.just_under_1000[tr],
        'pair_reverse_small_2h': X.pair_reverse_small_2h[tr] > 0,
    })
    print('\nMean of each check by label (train rows):')
    print(chk.groupby(lab.to_numpy()).mean().round(3).to_string())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', type=float, default=0.5, help="share of train 'legit' rows to keep (default 0.5)")
    ap.add_argument('--check', action='store_true', help='print September-vs-test drift instead of writing files')
    a = ap.parse_args()

    df, ac, mr = load(a.keep)
    log(f'loaded: {len(df):,} rows after keeping {a.keep:.0%} of train legit rows')
    X, keys = build_features(df, ac, mr)
    log(f'{X.shape[1]} features built')

    if a.check:
        drift_report(X, df.period)
        return

    sanity_report(X, df)
    tr, te = ~df.is_test, df.is_test
    print('\nTrain rows by period and label:')
    print(pd.crosstab(df.label[tr], df.period[tr], margins=True).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    train_out = pd.concat([df.loc[tr, ['txn_id', 'period', 'label']], keys[tr], X[tr]], axis=1)
    test_out = pd.concat([df.loc[te, ['txn_id']], keys[te], X[te]], axis=1).reset_index(drop=True)
    ss = pd.read_csv(DATA / 'sample_submission.csv')
    assert len(test_out) == 260_582 and (test_out.txn_id.to_numpy() == ss.id.to_numpy()).all()
    assert not train_out.columns.drop(['label']).str.contains('label').any()
    train_out.to_parquet(OUT / 'train.parquet', index=False)
    test_out.to_parquet(OUT / 'test.parquet', index=False)
    log(f'wrote {OUT / "train.parquet"} {train_out.shape} and {OUT / "test.parquet"} {test_out.shape}')


if __name__ == '__main__':
    main()
