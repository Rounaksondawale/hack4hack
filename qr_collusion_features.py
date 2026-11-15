"""qm_* features aimed at qr_swap and merchant_collusion (record: exp/work/feature_log.md).

    python code/qr_collusion_features.py

Adds the qm_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing qm_* columns are replaced, every other column is left untouched).

qr_swap = a QR payment that lands in a personal (P2P) account instead of the shop.
merchant_collusion = the same sender paying a merchant round amounts (multiples of 500) again and again.
Every history feature looks only at EARLIER rows (strictly before the row's timestamp), so:
  - train is computed on train alone,
  - test is computed with train as history, exactly like unseen data.
For new unseen data later: add_qm_features(new_df, history=<all earlier rows with the raw columns>).
The collusion features are for merchant rows only (NaN for P2P rows). Labels are never read.
"""
import numpy as np
import pandas as pd

from cashback_phishing_features import WORK

RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'amount', 'txn_type']
ORDER = ['qm_qr_to_p2p', 'qm_round500', 'qm_pair_same_amt_prior', 'qm_pair_round_prior',
         'qm_r_round_share', 'qm_r_repeat_share']


def prior_sum(d, keys, v):
    """Per row: sum of v over rows with the same keys and a strictly earlier timestamp (d sorted by time)."""
    incl = pd.Series(v, index=d.index).groupby([d[k] for k in keys]).cumsum()
    excl = incl - v
    # rows sharing a timestamp must not see each other: take the value at the first row of the tie
    return excl.groupby([d[k] for k in keys] + [d.timestamp]).transform('min').to_numpy()


def add_qm_features(df, history=None):
    """Return df with qm_* columns added. `history` = earlier rows used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('qm_')])
    d = df[RAW].assign(_target=True)
    if history is not None:
        d = pd.concat([d, history[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and history'
    d = d.sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    mer = (d.receiver_type == 'MERCHANT').to_numpy()
    f = pd.DataFrame(index=d.index)

    # qr_swap: QR payment to a personal account
    f['qm_qr_to_p2p'] = ((d.txn_type == 'QR') & (d.receiver_type == 'P2P')).astype(int)

    # merchant_collusion (merchant rows only)
    round500 = ((d.amount % 500 == 0) & (d.amount >= 1000)).to_numpy().astype(float)
    one = np.ones(len(d))
    first_pair = (~d.duplicated(['sender_id', 'receiver_id'])).to_numpy().astype(float)
    r_prior = prior_sum(d, ['receiver_id'], one)
    f['qm_round500'] = round500
    f['qm_pair_same_amt_prior'] = prior_sum(d, ['sender_id', 'receiver_id', 'amount'], one)
    f['qm_pair_round_prior'] = prior_sum(d, ['sender_id', 'receiver_id'], round500)
    with np.errstate(invalid='ignore', divide='ignore'):
        f['qm_r_round_share'] = np.where(r_prior > 0, prior_sum(d, ['receiver_id'], round500) / r_prior, np.nan)
        f['qm_r_repeat_share'] = np.where(r_prior > 0, 1 - prior_sum(d, ['receiver_id'], first_pair) / r_prior, np.nan)
    f.loc[~mer, ORDER[1:]] = np.nan

    out = f.loc[d._target.to_numpy(), ORDER].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('qm_')]
    te_cols = [c for c in te.columns if not c.startswith('qm_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr = add_qm_features(tr)
    te = add_qm_features(te, history=tr)

    qm = [c for c in tr.columns if c.startswith('qm_')]
    assert qm == [c for c in te.columns if c.startswith('qm_')], 'train/test qm_ columns differ'
    assert (tr.txn_id.to_numpy() == tr_ids).all() and (te.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr.columns if not c.startswith('qm_')] == tr_cols
    assert [c for c in te.columns if not c.startswith('qm_')] == te_cols
    tr.to_parquet(WORK / 'train_joined.parquet', index=False)
    te.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(qm)} qm_ features written | train {tr.shape} | test {te.shape}')
    print(qm)


if __name__ == '__main__':
    main()
