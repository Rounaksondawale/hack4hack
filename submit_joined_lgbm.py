"""Final lgbm_big on exp/work/train_joined.parquet -> outputs/submissions/submission_v7.csv.

    python code/submit_joined_lgbm.py

1. Validate (fit Jul-Aug, score Sep) to get the number of rounds and September probabilities.
2. Tune per-class probability multipliers on all of September (tune_scales, test-like legit weight).
3. Refit on every train row for 1.1x the best iteration, predict test_joined.parquet, apply the multipliers.
Features are the same as code/train_joined_lgbm.py (every column except ids, timestamps and dates).
"""
import numpy as np
import pandas as pd

from train_model import LABELS, MODELS, ROOT, OUTPUTS, crossfit_scaled, f1_table, log, macro_f1, tune_scales
from write_submission import write_submission

WORK = ROOT / 'exp' / 'work'
OUT = OUTPUTS / 'submissions' / 'submission_v7.csv'
DROP = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
        's_opened_date', 'r_opened_date', 'm_onboard_date']

df = pd.read_parquet(WORK / 'train_joined.parquet')
te = pd.read_parquet(WORK / 'test_joined.parquet')
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09')
y = df.label.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y).any(), 'unknown label in train_joined.parquet'
y = y.astype(int)

X = df.drop(columns=DROP)
Xt = te.drop(columns=[c for c in DROP if c in te.columns])[X.columns]  # same columns, same order
for c in X.select_dtypes('object').columns:
    X[c] = X[c].astype('category')
    Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)  # same category codes as train
fit_rows, val_rows = month.isin(['2026-07', '2026-08']).to_numpy(), (month == '2026-09').to_numpy()
log(f'fit {fit_rows.sum():,} rows (Jul-Aug), validate {val_rows.sum():,} rows (Sep), {X.shape[1]} features')

# 1. validate
model = MODELS['lgbm_big']()
p = model.validate(X[fit_rows], y[fit_rows], X[val_rows], y[val_rows])
yv = y[val_rows]
log(f'best iteration {model.rounds}')

# 2. per-class multipliers
raw = p.argmax(1)
honest = crossfit_scaled(yv, p, pd.factorize(df.sender_id[val_rows])[0])
w = tune_scales(yv, p)
print(f'\nSep macro-F1  raw {macro_f1(yv, raw):.4f}  (test-like {macro_f1(yv, raw, 2):.4f})')
print(f'Sep macro-F1  per-class scaled, cross-fit {macro_f1(yv, honest):.4f}  (test-like {macro_f1(yv, honest, 2):.4f})')
print('\nPer-class multipliers:', dict(zip(LABELS, w.round(3))))
rep = pd.DataFrame({'f1_raw': f1_table(yv, raw).f1, 'f1_scaled': f1_table(yv, honest).f1,
                    'support': f1_table(yv, raw).support})
print('\nPer-label F1 on September:')
print(rep.round(3).to_string())

# 3. refit on all train rows, predict test
log(f'final fit on {len(X):,} rows for {int(model.rounds * 1.1)} rounds')
pt = model.final(X, y, Xt)
pred = (pt * w).argmax(1)
labels = np.array(LABELS)[pred]
print('\nPredicted test label shares (%):')
print((pd.Series(labels).value_counts(normalize=True) * 100).round(3).to_string())
write_submission(te.txn_id, labels, OUT)
