"""Train lgbm_big (from train_model.py) on exp/work/train_joined.parquet: fit Jul-Aug, score macro-F1 on Sep.

    python code/train_joined_lgbm.py

Uses every column of train_joined.parquet (raw + joined + f_* features) except ids, timestamps and dates.
October rows count as September (same as prepare_data.py). Validation only: no submission is written.
"""
import numpy as np
import pandas as pd

from train_model import LABELS, MODELS, ROOT, OUTPUTS, crossfit_scaled, f1_table, log, macro_f1, report_confusion

DATA = ROOT / 'exp' / 'work' / 'train_joined.parquet'
DROP = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
        's_opened_date', 'r_opened_date', 'm_onboard_date']

df = pd.read_parquet(DATA)
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09')
y = df.label.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y).any(), 'unknown label in train_joined.parquet'
y = y.astype(int)

X = df.drop(columns=DROP)
for c in X.select_dtypes('object').columns:
    X[c] = X[c].astype('category')
fit_rows, val_rows = month.isin(['2026-07', '2026-08']).to_numpy(), (month == '2026-09').to_numpy()
log(f'fit {fit_rows.sum():,} rows (Jul-Aug), validate {val_rows.sum():,} rows (Sep), {X.shape[1]} features')

model = MODELS['lgbm_big']()
p = model.validate(X[fit_rows], y[fit_rows], X[val_rows], y[val_rows])
yv = y[val_rows]
log(f'best iteration {model.rounds}')

raw = p.argmax(1)
scaled = crossfit_scaled(yv, p, pd.factorize(df.sender_id[val_rows])[0])
print(f'\nSep macro-F1  raw {macro_f1(yv, raw):.4f}  (test-like {macro_f1(yv, raw, 2):.4f})')
print(f'Sep macro-F1  per-class scaled {macro_f1(yv, scaled):.4f}  (test-like {macro_f1(yv, scaled, 2):.4f})')

rep = pd.DataFrame({'f1_raw': f1_table(yv, raw).f1, 'f1_scaled': f1_table(yv, scaled).f1,
                    'support': f1_table(yv, raw).support})
print('\nPer-label F1 on September:')
print(rep.round(3).to_string())
print('\nTop 25 features by gain:')
print(model.top_gain(X.columns, 25).to_string())

OUTPUTS.mkdir(exist_ok=True)
report_confusion(yv, raw, 'lgbm_big [train_joined, none] on September', OUTPUTS / 'confusion_lgbm_big_joined_none.csv')
