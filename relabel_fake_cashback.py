"""Find train rows labelled 'legit' that are really 'fake_cashback' (unreported fraud).

    python relabel_fake_cashback.py     # writes data/02_processed/fc_relabel.csv

A fake_cashback episode: fraudster F sends victim V a bait of Rs 1-10 (P2P PAY),
then V pays F more than Rs 10 within 48 h. An episode is called fake_cashback when at
least one of its rows is labelled fake_cashback and none is labelled phishing_collect
(phishing_collect episodes have the same shape). Its 'legit' rows are relabelled.
Episodes whose rows are all 'legit' are left alone: in train they can't be told apart
from phishing_collect, which is the more common of the two in that zone.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / '01_raw'
OUT = ROOT / 'data' / '02_processed' / 'fc_relabel.csv'
WINDOW_H = 48

df = pd.read_csv(DATA / 'train.csv', parse_dates=['timestamp'])

# 1. link each bait F->V to every later payment V->F (> Rs 10) within the window
bait = df[(df.receiver_type == 'P2P') & (df.txn_type == 'PAY') & df.amount.between(1, 10)]
m = bait.merge(df, left_on=['sender_id', 'receiver_id'], right_on=['receiver_id', 'sender_id'], suffixes=('_b', '_p'))
gap_h = (m.timestamp_p - m.timestamp_b).dt.total_seconds() / 3600
m = m[(gap_h > 0) & (gap_h <= WINDOW_H) & (m.amount_p > 10)]
m['episode'] = m.sender_id_b + '->' + m.receiver_id_b

# 2. long format: one row per (episode, txn) with its role
rows = pd.concat([
    m[['episode', 'txn_id_b', 'label_b']].set_axis(['episode', 'txn_id', 'label'], axis=1).assign(role='bait'),
    m[['episode', 'txn_id_p', 'label_p']].set_axis(['episode', 'txn_id', 'label'], axis=1).assign(role='payback'),
]).drop_duplicates(['episode', 'txn_id'])

# 3. episode type from its labels
labels = rows.groupby('episode').label.agg(set)
fc_eps = labels[labels.map(lambda s: 'fake_cashback' in s and 'phishing_collect' not in s)].index

out = rows[rows.episode.isin(fc_eps) & (rows.label == 'legit')].drop_duplicates('txn_id').assign(label_new='fake_cashback')
out = out[['txn_id', 'label', 'label_new', 'role', 'episode']].sort_values(['episode', 'role'])

# 4. verify
assert out.txn_id.is_unique
assert set(out.txn_id) <= set(df.loc[df.label == 'legit', 'txn_id'])
print(f'episodes: {labels.size} total, {len(fc_eps)} fake_cashback')
print(f'legit -> fake_cashback: {len(out)} rows', out.role.value_counts().to_dict())

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)
print('wrote', OUT)
