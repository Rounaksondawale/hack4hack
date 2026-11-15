"""vb_* features aimed at velocity_bot (record: exp/work/feature_log.md).

    python code/velocity_bot_features.py

Adds the vb_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing vb_* columns are replaced, every other column is left untouched).

velocity_bot shows up as bursts on a DEVICE (farm: many senders on one device, one payment every
0-30 s) or on one sender (2-3 merchants in rotation every 2-40 s), not as sender-level counts.
Every feature looks only at EARLIER rows (strictly before the row's timestamp; rows that share a
timestamp don't see each other), EXCEPT vb_dev_gap_next_s, which looks at the device's next row.
  - train is computed on train alone,
  - test is computed with train as history (same as cp_/qm_).
For new unseen data later: add_vb_features(new_df, history=<all other rows with the raw columns>).
Labels are never read.
"""
import warnings

import numpy as np
import pandas as pd

from cashback_phishing_features import WORK, SPAN

RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'device_id', 'amount',
       'login_to_txn_sec', 'device_age_days', 'ip_risk_score']
REG = ['gap_mean_s', 'gap_std_s', 'gap_cv', 'gap_max_s', 'gap_min_s', 'burst_len', 'burst_dur_s', 'burst_rate']
ORDER = (['vb_dev_n_1m', 'vb_dev_n_5m', 'vb_dev_nsnd_5m', 'vb_dev_nsnd_1h', 'vb_dev_gap_prev_s',
          'vb_dev_gap_next_s', 'vb_own_device']
         + [f'vb_dev_{c}' for c in REG] + [f'vb_snd_{c}' for c in REG]
         + ['vb_rcv_n_1m', 'vb_rcv_n_5m', 'vb_rcv_nsnd_5m', 'vb_rcv_gap_prev_s', 'vb_snd_nrcv_10m',
            'vb_snd_rcv_rotation']
         + ['vb_login_fast', 'vb_burst_amt_std', 'vb_burst_amt_max', 'vb_burst_dev_age', 'vb_burst_ip_risk']
         + ['vb_farm_flag', 'vb_bot_score'])
LAST_K = 10      # regularity / amount stats use at most the last 10 earlier rows ...
REG_WIN = 600    # ... within the last 10 minutes
BURST_GAP = 60   # consecutive rows less than 60 s apart form one burst


class View:
    """Rows sorted by (entity, time). For each row: [lo, hi) = strictly earlier rows of the same entity in a window."""

    def __init__(self, key, t):
        self.g = pd.factorize(key)[0].astype(np.int64)
        self.order = np.lexsort((t, self.g))
        self.gs, self.ts = self.g[self.order], t[self.order]
        self.c = self.gs * SPAN + self.ts
        self.hi = np.searchsorted(self.c, self.c, 'left')           # first row with the same time
        self.nx = np.searchsorted(self.c, self.c, 'right')          # first strictly later row
        self.start = np.searchsorted(self.c, self.gs * SPAN, 'left')  # first row of the entity
        self.n = len(t)

    def lo(self, w):
        return np.searchsorted(self.c, self.c - w, 'left')

    def back(self, x):
        """Sorted-order array -> original row order."""
        out = np.empty_like(x)
        out[self.order] = x
        return out

    def count(self, w):
        """Rows of the entity in [t - w, t), plus the row itself."""
        return self.hi - self.lo(w) + 1

    def gap_prev(self):
        p = self.hi - 1
        return np.where(p >= self.start, self.ts - self.ts[np.maximum(p, 0)], np.nan)

    def gap_next(self):
        ok = (self.nx < self.n) & (self.gs[np.minimum(self.nx, self.n - 1)] == self.gs)
        return np.where(ok, self.ts[np.minimum(self.nx, self.n - 1)] - self.ts, np.nan)

    def distinct(self, other, w):
        """Distinct values of `other` among rows of the entity in [t - w, t), plus the row itself."""
        pair = pd.factorize(pd.Series(self.gs).astype(str) + '|' + pd.Series(other[self.order]).astype(str))[0]
        pos = pd.Series(np.arange(self.n))
        prev_same = pos.groupby(pair).shift(1).fillna(-1).to_numpy().astype(np.int64)
        # same-pair row strictly before this row's timestamp = prev of the first row of its tie group
        prev_strict = pd.Series(prev_same).groupby([pair, self.ts]).transform('first').to_numpy()
        lo, hi = self.lo(w), self.hi
        cnt = np.zeros(self.n, dtype=np.int64)
        for k in range(int((hi - lo).max())):
            j = hi - 1 - k
            ok = j >= lo
            cnt += ok & (prev_same[np.maximum(j, 0)] < lo)  # first occurrence of the value inside the window
        return cnt + (prev_strict < lo)

    def last_k(self, values, w=REG_WIN, k=LAST_K):
        """Matrix: column 0 = this row, columns 1..k = the last k earlier rows within w (NaN when missing)."""
        lo = self.lo(w)
        m = np.full((self.n, k + 1), np.nan)
        m[:, 0] = values
        for i in range(k):
            j = self.hi - 1 - i
            ok = j >= lo
            m[:, i + 1] = np.where(ok, values[np.maximum(j, 0)], np.nan)
        return m

    def regularity(self):
        """Gap stats over the last 10 earlier rows within 10 min, and the current burst (gaps < 60 s)."""
        T = self.last_k(self.ts.astype(float))
        gaps = T[:, :-1] - T[:, 1:]
        ng = (~np.isnan(gaps)).sum(1)
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean = np.nanmean(gaps, 1)
            std = np.where(ng >= 2, np.nanstd(gaps, 1), np.nan)
            cv = std / np.maximum(mean, 1)
            mx, mn = np.nanmax(gaps, 1), np.nanmin(gaps, 1)
        # bursts: runs of consecutive rows (in time order) less than BURST_GAP apart
        pos = np.arange(self.n)
        d = np.diff(self.ts, prepend=self.ts[0] - BURST_GAP)
        new_run = (pos == self.start) | (d >= BURST_GAP)
        run_start = np.maximum.accumulate(np.where(new_run, pos, 0))
        p = np.maximum(self.hi - 1, 0)
        ok = (self.hi - 1 >= self.start) & (self.ts - self.ts[p] < BURST_GAP)
        blen = np.where(ok, p - run_start[p] + 2, 1)
        dur = np.where(ok, self.ts - self.ts[run_start[p]], 0)
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = np.where(blen > 1, (blen - 1) / np.maximum(dur, 1) * 60, 0.0)
        return dict(zip(REG, [mean, std, cv, mx, mn, blen, dur, rate]))


def add_vb_features(df, history=None):
    """Return df with vb_* columns added. `history` = other rows used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('vb_')])
    d = df[RAW].assign(_target=True)
    if history is not None:
        d = pd.concat([d, history[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and history'
    t = (d.timestamp.astype('int64') // 10**9).to_numpy()
    snd, rcv = d.sender_id.to_numpy(), d.receiver_id.to_numpy()
    f = pd.DataFrame(index=d.index)

    # 1. device bursts
    dev = View(d.device_id.to_numpy(), t)
    f['vb_dev_n_1m'] = dev.back(dev.count(60))
    f['vb_dev_n_5m'] = dev.back(dev.count(300))
    f['vb_dev_nsnd_5m'] = dev.back(dev.distinct(snd, 300))
    f['vb_dev_nsnd_1h'] = dev.back(dev.distinct(snd, 3600))
    f['vb_dev_gap_prev_s'] = dev.back(dev.gap_prev())
    f['vb_dev_gap_next_s'] = dev.back(dev.gap_next())
    f['vb_own_device'] = (d.device_id.str[2:].astype(int) == d.sender_id.str[2:].astype(int)).astype(int)

    # 2. timing regularity per device and per sender
    for name, view in [('dev', dev), ('snd', View(snd, t))]:
        for k, v in view.regularity().items():
            f[f'vb_{name}_{k}'] = view.back(v)

    # 3. receiver side, and how many receivers one sender rotates through
    rv = View(rcv, t)
    f['vb_rcv_n_1m'] = rv.back(rv.count(60))
    f['vb_rcv_n_5m'] = rv.back(rv.count(300))
    f['vb_rcv_nsnd_5m'] = rv.back(rv.distinct(snd, 300))
    f['vb_rcv_gap_prev_s'] = rv.back(rv.gap_prev())
    sv = View(snd, t)
    f['vb_snd_nrcv_10m'] = sv.back(sv.distinct(rcv, 600))
    f['vb_snd_rcv_rotation'] = f.vb_snd_nrcv_10m / sv.back(sv.count(600))

    # 4. per-transaction bot signs
    burst = f.vb_dev_n_5m.to_numpy() >= 3
    f['vb_login_fast'] = (d.login_to_txn_sec <= 6).astype(int)
    A = dev.last_k(d.amount.to_numpy()[dev.order].astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n_amt = (~np.isnan(A)).sum(1)
        f['vb_burst_amt_std'] = dev.back(np.where(n_amt >= 2, np.nanstd(A, 1), np.nan))
        f['vb_burst_amt_max'] = dev.back(np.nanmax(A, 1))
    f['vb_burst_dev_age'] = np.where(burst, d.device_age_days, np.nan)
    f['vb_burst_ip_risk'] = np.where(burst, d.ip_risk_score, np.nan)

    # 5. combined flags
    f['vb_farm_flag'] = (f.vb_dev_nsnd_5m >= 3).astype(int)
    f['vb_bot_score'] = f.vb_dev_n_5m * f.vb_login_fast * (d.amount % 1 == 0).astype(int)

    out = f.loc[d._target.to_numpy(), ORDER].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('vb_')]
    te_cols = [c for c in te.columns if not c.startswith('vb_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr = add_vb_features(tr)
    te = add_vb_features(te, history=tr)

    vb = [c for c in tr.columns if c.startswith('vb_')]
    assert vb == ORDER == [c for c in te.columns if c.startswith('vb_')], 'train/test vb_ columns differ'
    assert (tr.txn_id.to_numpy() == tr_ids).all() and (te.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr.columns if not c.startswith('vb_')] == tr_cols
    assert [c for c in te.columns if not c.startswith('vb_')] == te_cols
    tr.to_parquet(WORK / 'train_joined.parquet', index=False)
    te.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(vb)} vb_ features written | train {tr.shape} | test {te.shape}')
    print(vb)


if __name__ == '__main__':
    main()
