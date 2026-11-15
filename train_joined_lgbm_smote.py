"""lgbm_big on exp/work/train_joined.parquet with resampled training rows: fit Jul-Aug, score macro-F1 on Sep.

    python code/train_joined_lgbm_smote.py

Training rows (Jul-Aug) only: keep 20% of legit (drop 80%), then SMOTENC each fraud class up to 20,000 rows.
September is NOT resampled, so the score is on the real class mix. Class weights are off for training
(the data is already rebalanced); the September early-stopping set keeps them, as in train_model.py.
SMOTE cannot take NaN: numeric NaN -> -999, categorical NaN -> 'NA' (same fill on September).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from train_model import (LABELS, MODELS, ROOT, OUTPUTS, class_weights, crossfit_scaled, f1_table, log, macro_f1,
                         report_confusion)

DATA = ROOT / 'exp' / 'work' / 'train_joined.parquet'
DROP = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
        's_opened_date', 'r_opened_date', 'm_onboard_date']
LEGIT_KEEP = 0.2       # drop 80% of legit
FRAUD_TARGET = 20_000  # SMOTE each fraud class up to this many rows

df = pd.read_parquet(DATA)
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09')
y = df.label.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y).any(), 'unknown label in train_joined.parquet'
y = y.astype(int)

X = df.drop(columns=DROP)
cat_cols = list(X.select_dtypes('object').columns)
num_cols = [c for c in X.columns if c not in cat_cols]
X[cat_cols] = X[cat_cols].fillna('NA')
X[num_cols] = X[num_cols].fillna(-999).astype(np.float64)
fit_rows, val_rows = month.isin(['2026-07', '2026-08']).to_numpy(), (month == '2026-09').to_numpy()

# --- resample the training rows only
Xf, yf = X[fit_rows].reset_index(drop=True), y[fit_rows]
rng = np.random.default_rng(0)
keep = (yf != 0) | (rng.random(len(yf)) < LEGIT_KEEP)
Xf, yf = Xf[keep].reset_index(drop=True), yf[keep]
counts = np.bincount(yf, minlength=len(LABELS))
target = {c: max(int(counts[c]), FRAUD_TARGET) for c in range(1, len(LABELS))}
log(f'after legit downsample: {len(yf):,} rows; SMOTENC fraud classes to {FRAUD_TARGET:,} ...')
Xf, yf = SMOTENC(categorical_features=cat_cols, sampling_strategy=target, random_state=0).fit_resample(Xf, yf)
yf = np.asarray(yf)
print(pd.DataFrame({'before': counts, 'after': np.bincount(yf, minlength=len(LABELS))}, index=LABELS).to_string())

for c in cat_cols:  # same categories on both sides for LightGBM
    cats = pd.Index(X[c].unique())
    X[c] = pd.Categorical(X[c], categories=cats)
    Xf[c] = pd.Categorical(Xf[c], categories=cats)
Xv, yv = X[val_rows], y[val_rows]
log(f'fit {len(yf):,} rows (resampled Jul-Aug), validate {len(yv):,} rows (Sep), {X.shape[1]} features')

# --- lgbm_big params, no class weights on the (balanced) training rows
params = {**MODELS['lgbm_big']().params, 'num_threads': 8}  # 8 physical cores: ~19% faster, same predictions
dtrain = lgb.Dataset(Xf, yf)
dvalid = lgb.Dataset(Xv, yv, weight=class_weights(yv), reference=dtrain)
m = lgb.train(params, dtrain, 3000, valid_sets=[dvalid],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
log(f'best iteration {m.best_iteration}')
p = m.predict(Xv, num_iteration=m.best_iteration)

raw = p.argmax(1)
scaled = crossfit_scaled(yv, p, pd.factorize(df.sender_id[val_rows])[0])
print(f'\nSep macro-F1  raw {macro_f1(yv, raw):.4f}  (test-like {macro_f1(yv, raw, 2):.4f})')
print(f'Sep macro-F1  per-class scaled {macro_f1(yv, scaled):.4f}  (test-like {macro_f1(yv, scaled, 2):.4f})')

rep = pd.DataFrame({'f1_raw': f1_table(yv, raw).f1, 'f1_scaled': f1_table(yv, scaled).f1,
                    'support': f1_table(yv, raw).support})
print('\nPer-label F1 on September:')
print(rep.round(3).to_string())
gain = pd.Series(m.feature_importance('gain'), index=X.columns).sort_values(ascending=False)
print('\nTop 25 features by gain:')
print((gain / gain.sum()).head(25).round(3).to_string())

OUTPUTS.mkdir(exist_ok=True)
report_confusion(yv, raw, 'lgbm_big [train_joined, SMOTE, none] on September',
                 OUTPUTS / 'confusion_lgbm_big_joined_smote_none.csv')
