"""lc_* features aimed at layering_chain (record: exp/work/feature_log.md).

    python code/layering_features.py

Adds the lc_* columns to exp/work/train_joined.parquet and exp/work/test_joined.parquet in place
(existing lc_* columns are replaced, every other column is left untouched).

A layering chain is a run of P2P hops: A -> B -> C -> ..., each hop forwarding 96.5-99.5% of what just came in,
usually within minutes. All lc_* features are for P2P rows only (NaN for merchant rows), and the hops they look at
are P2P payments only.
Some features look FORWARD (the receiver's next payment out), so context works like mu_*:
train is built with test as context, test with train; new data: add_lc_features(new_df, context=<other rows>).
Labels are never read.
"""
import numpy as np
import pandas as pd

from mule_features import WORK, nearest

HOP_LO, HOP_HI = 0.95, 1.0  # forwarded amount / received amount for a chain hop
HOP_GAP_MIN = 360  # a hop must follow within 6 h
MAX_DEPTH = 50
RAW = ['txn_id', 'timestamp', 'sender_id', 'receiver_id', 'receiver_type', 'amount', 'balance_before']
ORDER = ['lc_in_ratio', 'lc_in_gap_min', 'lc_out_ratio', 'lc_out_gap_min', 'lc_in_hop', 'lc_out_hop',
         'lc_any_hop', 'lc_both_hop', 'lc_depth_up', 'lc_depth_down', 'lc_pre_in_residual']


def chain_depth(hop, link):
    """Per row: number of consecutive hops reached by following `link` while `hop` holds."""
    depth = np.zeros(len(hop), dtype=np.int64)
    safe = np.where(hop, link, 0)
    for _ in range(MAX_DEPTH):
        new = np.where(hop, depth[safe] + 1, 0)
        if (new == depth).all():
            break
        depth = new
    return depth


def add_lc_features(df, context=None):
    """Return df with lc_* columns added. `context` = other rows (any time) used only as neighbours."""
    df = df.drop(columns=[c for c in df.columns if c.startswith('lc_')])
    d = df[RAW].assign(_target=True)
    if context is not None:
        d = pd.concat([d, context[RAW].assign(_target=False)], ignore_index=True)
    assert d.txn_id.is_unique, 'txn_id overlaps between df and context'
    d = d[d.receiver_type == 'P2P'].sort_values(['timestamp', 'txn_id'], kind='mergesort').reset_index(drop=True)
    t = d.timestamp.astype('int64').to_numpy() // 10**9
    snd, rcv, amt = d.sender_id.to_numpy(), d.receiver_id.to_numpy(), d.amount.to_numpy()
    idx = np.arange(len(d), dtype=float)
    f = pd.DataFrame(index=d.index)

    # backward: the single last P2P payment INTO the sender (that payment's receiver == this sender)
    prv = nearest(snd, t, rcv, t, {'a2': amt, 'j': idx}, 'backward')
    f['lc_in_ratio'] = amt / prv.a2.to_numpy()
    f['lc_in_gap_min'] = ((t - prv.t2) / 60).to_numpy()
    # forward: the receiver's next P2P payment OUT
    nxt = nearest(rcv, t, snd, t, {'a2': amt, 'j': idx}, 'forward')
    f['lc_out_ratio'] = nxt.a2.to_numpy() / amt
    f['lc_out_gap_min'] = ((nxt.t2 - t) / 60).to_numpy()

    in_hop = (f.lc_in_ratio.between(HOP_LO, HOP_HI) & (f.lc_in_gap_min <= HOP_GAP_MIN)).to_numpy()
    out_hop = (f.lc_out_ratio.between(HOP_LO, HOP_HI) & (f.lc_out_gap_min <= HOP_GAP_MIN)).to_numpy()
    f['lc_in_hop'], f['lc_out_hop'] = in_hop.astype(int), out_hop.astype(int)
    f['lc_any_hop'] = (in_hop | out_hop).astype(int)
    f['lc_both_hop'] = (in_hop & out_hop).astype(int)
    # how many hops in a row come before / after this one
    f['lc_depth_up'] = chain_depth(in_hop, prv.j.fillna(0).to_numpy().astype(np.int64))
    f['lc_depth_down'] = chain_depth(out_hop, nxt.j.fillna(0).to_numpy().astype(np.int64))
    # ~0 = the sender had almost nothing before the last inflow (pass-through account)
    f['lc_pre_in_residual'] = ((d.balance_before.to_numpy() - prv.a2.to_numpy()) / prv.a2.to_numpy())

    out = f.loc[d._target.to_numpy(), ORDER].set_index(d.loc[d._target, 'txn_id'])
    res = df.join(out, on='txn_id')
    assert len(res) == len(df) and (res.txn_id.to_numpy() == df.txn_id.to_numpy()).all()
    return res


def main():
    tr = pd.read_parquet(WORK / 'train_joined.parquet')
    te = pd.read_parquet(WORK / 'test_joined.parquet')
    tr_cols = [c for c in tr.columns if not c.startswith('lc_')]
    te_cols = [c for c in te.columns if not c.startswith('lc_')]
    tr_ids, te_ids = tr.txn_id.to_numpy().copy(), te.txn_id.to_numpy().copy()

    tr_new = add_lc_features(tr, context=te)
    te_new = add_lc_features(te, context=tr)

    lc = [c for c in tr_new.columns if c.startswith('lc_')]
    assert lc == [c for c in te_new.columns if c.startswith('lc_')], 'train/test lc_ columns differ'
    assert (tr_new.txn_id.to_numpy() == tr_ids).all() and (te_new.txn_id.to_numpy() == te_ids).all()
    assert [c for c in tr_new.columns if not c.startswith('lc_')] == tr_cols
    assert [c for c in te_new.columns if not c.startswith('lc_')] == te_cols
    tr_new.to_parquet(WORK / 'train_joined.parquet', index=False)
    te_new.to_parquet(WORK / 'test_joined.parquet', index=False)
    print(f'{len(lc)} lc_ features written | train {tr_new.shape} | test {te_new.shape}')
    print(lc)


if __name__ == '__main__':
    main()
