"""rb_* features: one fraud idea each, built to behave the same in October (record: exp/work/feature_log.md).

    python code/robust_features.py

Adds the rb_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing rb_* columns are replaced, every other column is left untouched).

Why these exist: test is the next month (festival season: bigger amounts; a new app version for
everyone; half the daily traffic; a vendor score on a new scale). So every rb_ feature is
  - relative, not absolute: amount vs that day's median or vs the sender's own payments, not rupees;
  - a rank, not a raw score: ip_risk_score as its percentile within the day;
  - a fixed window (24 h / 7 d / 30 d), not "all history": test rows have more history behind them;
  - capped: gaps longer than the window are NaN ("nothing recent"), not huge numbers.
Every history feature looks only at EARLIER rows (strictly before the row's timestamp; rows that share a
timestamp don't see each other). Train is computed on train alone, test with train as history.
Labels are never read. Use together with code/feature_policy.py, which removes the features these replace.
"""
import warnings

import numpy as np
import pandas as pd

from cashback_phishing_features import SPAN, WORK
from merchant_collusion_features import Past

RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'device_id', 'amount',
       'balance_before', 'txn_type', 'merchant_category', 'ip_risk_score', 'device_city', 's_home_city',
       'sim_age_hours', 'device_age_days']
HOUR, DAY = 3600, 86400
WEEK, MONTH = 7 * DAY, 30 * DAY
ROUND_LEVELS = [100, 500, 1000, 5000, 10000]
ORDER = [
    # shared: time of day, day-relative amounts, rank of the vendor score
    'rb_hour', 'rb_night', 'rb_late_evening', 'rb_ip_rank_day', 'rb_amt_rel_day', 'rb_bal_rel_day',
    'rb_amt_vs_s_med30',
    # history over fixed windows (replace all-history counts)
    'rb_s_n_30d', 'rb_s_gap_h', 'rb_r_n_30d', 'rb_pair_n_30d', 'rb_pair_new_30d', 'rb_r_repeat_share_30d',
    'rb_r_round_share_30d', 'rb_dev_nsnd_30d', 'rb_r_sent_30d',
    # draining an account (sim_swap, account_takeover, layering)
    'rb_s_drain_24h', 'rb_s_newpayee_24h', 'rb_sim_fresh_old_dev', 'rb_dev_fresh_far', 'rb_ip_rank_x_fresh_dev',
    # phishing_collect, fake_cashback, qr_swap
    'rb_r_collect_nsnd_7d', 'rb_rev_tiny_7d', 'rb_rev_any_7d', 'rb_r_qr_nsnd_7d',
    # refund_abuse
    'rb_s_refund_7d', 'rb_s_refund_30d', 'rb_s_refund_nmer_30d', 'rb_s_refund_share_30d',
    # loan_app_extortion
    'rb_s_lending_7d', 'rb_pair_n_7d', 'rb_pair_evening_7d',
    # merchant_collusion, velocity_bot
    'rb_round_level', 'rb_round_x_ip_rank', 'rb_burst_ip_rank',
]


def windowed_median(past, v, w, k=20):
    """Median of v over the last k earlier eligible rows of the same key within w (NaN if none)."""
    lo, hi = past.bounds(w)
    m = np.full((len(past.q), k), np.nan)
    for i in range(k):
        j = hi - 1 - i
        m[:, i] = np.where(j >= lo, v[past.idx[np.maximum(j, 0)]], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(m, axis=1)


def add_rb_features(df, history=None):
    """Return df with rb_* columns added. `history` = earlier rows used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('rb_')])
    d = df[RAW].assign(_target=True)
    if history is not None:
        d = pd.concat([d, history[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and history'
    d = d.sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    n = len(d)
    t = (d.timestamp.astype('int64') // 10**9).to_numpy()
    amt, bal = d.amount.to_numpy(float), d.balance_before.to_numpy(float)
    hour = d.timestamp.dt.hour.to_numpy()
    p2p = (d.receiver_type == 'P2P').to_numpy()
    ttype = d.txn_type.to_numpy()
    allrows = np.ones(n, bool)

    acct = pd.factorize(pd.concat([d.sender_id, d.receiver_id]))[0].astype(np.int64)
    snd, rcv = acct[:n], acct[n:]
    pair_codes = pd.factorize(pd.concat([pd.Series(snd * 10**7 + rcv), pd.Series(rcv * 10**7 + snd)]))[0]
    pair, rev_pair = pair_codes[:n].astype(np.int64), pair_codes[n:].astype(np.int64)  # rev = receiver -> sender
    dev = pd.factorize(d.device_id)[0].astype(np.int64)
    f = pd.DataFrame(index=d.index)

    # ---- shared ------------------------------------------------------------------------------
    # day-level normalisation uses only rows of the same file (train ranks within train, test within test)
    day = [d.timestamp.dt.date, d._target]
    f['rb_hour'] = hour
    f['rb_night'] = (hour < 6).astype(int)
    f['rb_late_evening'] = (hour >= 20).astype(int)
    f['rb_ip_rank_day'] = d.groupby(day).ip_risk_score.rank(pct=True).to_numpy()
    f['rb_amt_rel_day'] = amt / d.groupby(day).amount.transform('median').to_numpy()
    f['rb_bal_rel_day'] = bal / d.groupby(day).balance_before.transform('median').to_numpy()
    s_all = Past(snd, t, allrows)
    s_med = windowed_median(s_all, amt, MONTH)
    f['rb_amt_vs_s_med30'] = np.where(s_med > 0, amt / s_med, np.nan)

    # ---- fixed-window history ------------------------------------------------------------------
    f['rb_s_n_30d'] = s_all.count(MONTH)
    gap = (t - s_all.last(t.astype(float))) / HOUR
    f['rb_s_gap_h'] = np.where(gap <= 30 * 24, gap, np.nan)
    r_all = Past(rcv, t, allrows)
    f['rb_r_n_30d'] = r_all.count(MONTH)
    pair_past = Past(pair, t, allrows)
    pair_n30 = pair_past.count(MONTH)
    f['rb_pair_n_30d'] = pair_n30
    f['rb_pair_new_30d'] = (pair_n30 == 0).astype(int)
    first_in_30 = (pair_n30 == 0).astype(float)
    r_n30 = f.rb_r_n_30d.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        f['rb_r_repeat_share_30d'] = np.where(r_n30 > 0, 1 - r_all.sum(first_in_30, MONTH) / r_n30, np.nan)
        rnd500 = ((amt % 500 == 0) & (amt >= 1000)).astype(float)
        f['rb_r_round_share_30d'] = np.where(r_n30 > 0, r_all.sum(rnd500, MONTH) / r_n30, np.nan)
    f['rb_dev_nsnd_30d'] = Past(dev, t, allrows).distinct(snd, MONTH)
    # how often the receiver itself paid others in the last 30 days (sender-keyed rows, queried by receiver)
    lo = np.searchsorted(s_all.c, rcv * SPAN + t - MONTH, 'left')
    hi = np.searchsorted(s_all.c, rcv * SPAN + t, 'left')
    f['rb_r_sent_30d'] = hi - lo

    # ---- draining an account ---------------------------------------------------------------------
    out24 = s_all.sum(amt, DAY)
    with np.errstate(invalid='ignore', divide='ignore'):
        start_bal = bal + out24
        f['rb_s_drain_24h'] = np.where(start_bal > 0, (out24 + amt) / start_bal, np.nan)
    f['rb_s_newpayee_24h'] = s_all.sum(first_in_30, DAY)
    sim_h, dev_d = d.sim_age_hours.to_numpy(float), d.device_age_days.to_numpy(float)
    far = (d.device_city != d.s_home_city).to_numpy()
    f['rb_sim_fresh_old_dev'] = ((sim_h <= 72) & (dev_d >= 30)).astype(int)
    f['rb_dev_fresh_far'] = ((dev_d <= 2) & far).astype(int)
    f['rb_ip_rank_x_fresh_dev'] = np.where(dev_d <= 2, f.rb_ip_rank_day, 0.0)

    # ---- phishing_collect / fake_cashback / qr_swap ---------------------------------------------
    f['rb_r_collect_nsnd_7d'] = Past(rcv, t, ttype == 'COLLECT').distinct(snd, WEEK)
    tiny = amt <= 10
    for name, elig in [('rb_rev_tiny_7d', tiny), ('rb_rev_any_7d', allrows)]:
        p = Past(pair, t, elig)  # rows keyed by their own sender->receiver pair ...
        lo = np.searchsorted(p.c, rev_pair * SPAN + t - WEEK, 'left')  # ... queried reversed
        hi = np.searchsorted(p.c, rev_pair * SPAN + t, 'left')
        f[name] = (hi > lo).astype(int)
    qr_nsnd = Past(rcv, t, (ttype == 'QR') & p2p).distinct(snd, WEEK).astype(float)
    f['rb_r_qr_nsnd_7d'] = np.where(p2p, qr_nsnd, np.nan)

    # ---- refund_abuse ----------------------------------------------------------------------------
    refund = Past(snd, t, ttype == 'REFUND')
    f['rb_s_refund_7d'] = refund.count(WEEK)
    f['rb_s_refund_30d'] = refund.count(MONTH)
    f['rb_s_refund_nmer_30d'] = refund.distinct(rcv, MONTH)
    s_n30 = f.rb_s_n_30d.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        f['rb_s_refund_share_30d'] = np.where(s_n30 > 0, f.rb_s_refund_30d / s_n30, np.nan)

    # ---- loan_app_extortion ----------------------------------------------------------------------
    f['rb_s_lending_7d'] = Past(snd, t, (d.merchant_category == 'lending').to_numpy()).count(WEEK)
    f['rb_pair_n_7d'] = pair_past.count(WEEK)
    f['rb_pair_evening_7d'] = Past(pair, t, hour >= 18).count(WEEK)

    # ---- merchant_collusion / velocity_bot ---------------------------------------------------------
    level = np.zeros(n)
    for r in ROUND_LEVELS:
        level = np.where((amt % r == 0) & (amt >= r), r, level)
    f['rb_round_level'] = level
    f['rb_round_x_ip_rank'] = np.where(level >= 500, f.rb_ip_rank_day, 0.0)
    dev_n5 = Past(dev, t, allrows).count(300) + 1
    f['rb_burst_ip_rank'] = np.where(dev_n5 >= 3, f.rb_ip_rank_day, np.nan)

    out = f.loc[d._target.to_numpy(), ORDER].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('rb_')]
    te_cols = [c for c in te.columns if not c.startswith('rb_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr = add_rb_features(tr)
    te = add_rb_features(te, history=tr)

    rb = [c for c in tr.columns if c.startswith('rb_')]
    assert rb == ORDER == [c for c in te.columns if c.startswith('rb_')], 'train/test rb_ columns differ'
    assert (tr.txn_id.to_numpy() == tr_ids).all() and (te.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr.columns if not c.startswith('rb_')] == tr_cols
    assert [c for c in te.columns if not c.startswith('rb_')] == te_cols
    tr.to_parquet(WORK / 'train_joined.parquet', index=False)
    te.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(rb)} rb_ features written | train {tr.shape} | test {te.shape}')
    print(rb)


if __name__ == '__main__':
    main()
