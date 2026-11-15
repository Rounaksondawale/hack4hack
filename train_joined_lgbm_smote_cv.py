"""Month-rotation version of train_joined_lgbm_smote.py: every month is scored once by a model trained on the other two.

    python code/train_joined_lgbm_smote_cv.py

Folds: Jul+Aug -> Sep, Jul+Sep -> Aug, Aug+Sep -> Jul (October rows count as September).
Per fold, training months only: keep 20% of legit, SMOTENC each fraud class up to 20,000 rows, lgbm_big without
class weights. The held-out month is NOT resampled (real class mix) and is used for early stopping, as before.
SMOTE cannot take NaN: numeric NaN -> -999, categorical NaN -> 'NA'.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC

from train_model import LABELS, MODELS, ROOT, OUTPUTS, class_weights, crossfit_scaled, f1_table, log, macro_f1

DATA = ROOT / 'exp' / 'work' / 'train_joined.parquet'
DROP = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
        's_opened_date', 'r_opened_date', 'm_onboard_date']
LEGIT_KEEP = 0.2       # drop 80% of legit
FRAUD_TARGET = 20_000  # SMOTE each fraud class up to this many rows
MONTHS = ['2026-07', '2026-08', '2026-09']

df = pd.read_parquet(DATA)
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09').to_numpy()
y = df.label.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y).any(), 'unknown label in train_joined.parquet'
y = y.astype(int)

X = df.drop(columns=DROP)
cat_cols = list(X.select_dtypes('object').columns)
num_cols = [c for c in X.columns if c not in cat_cols]
X[cat_cols] = X[cat_cols].fillna('NA')
X[num_cols] = X[num_cols].fillna(-999).astype(np.float64)
cats = {c: pd.Index(X[c].unique()) for c in cat_cols}
params = {**MODELS['lgbm_big']().params, 'num_threads': 8}  # 8 physical cores: ~19% faster, same predictions


def as_category(A):
    for c in cat_cols:  # same categories on every side for LightGBM
        A[c] = pd.Categorical(A[c], categories=cats[c])
    return A


results, per_label = [], {}
for held in MONTHS:
    train_months = [m for m in MONTHS if m != held]
    fit_rows, val_rows = np.isin(month, train_months), month == held
    log(f'===== fold: train {"+".join(train_months)} -> score {held}')

    Xf, yf = X[fit_rows].reset_index(drop=True), y[fit_rows]
    keep = (yf != 0) | (np.random.default_rng(0).random(len(yf)) < LEGIT_KEEP)
    Xf, yf = Xf[keep].reset_index(drop=True), yf[keep]
    counts = np.bincount(yf, minlength=len(LABELS))
    target = {c: max(int(counts[c]), FRAUD_TARGET) for c in range(1, len(LABELS))}
    Xf, yf = SMOTENC(categorical_features=cat_cols, sampling_strategy=target, random_state=0).fit_resample(Xf, yf)
    yf = np.asarray(yf)
    Xf, Xv, yv = as_category(Xf), as_category(X[val_rows].copy()), y[val_rows]
    log(f'fit {len(yf):,} rows (resampled), score {len(yv):,} rows ({held})')

    dtrain = lgb.Dataset(Xf, yf)
    dvalid = lgb.Dataset(Xv, yv, weight=class_weights(yv), reference=dtrain)
    m = lgb.train(params, dtrain, 3000, valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    p = m.predict(Xv, num_iteration=m.best_iteration)

    raw = p.argmax(1)
    scaled = crossfit_scaled(yv, p, pd.factorize(df.sender_id[val_rows])[0])
    results.append(dict(scored=held, trained_on='+'.join(train_months), best_iter=m.best_iteration,
                        f1_raw=macro_f1(yv, raw), f1_scaled=macro_f1(yv, scaled),
                        f1_raw_testlike=macro_f1(yv, raw, 2), f1_scaled_testlike=macro_f1(yv, scaled, 2)))
    per_label[held] = f1_table(yv, scaled).f1
    log(f'{held}: macro-F1 raw {results[-1]["f1_raw"]:.4f}, scaled {results[-1]["f1_scaled"]:.4f} '
        f'(best iteration {m.best_iteration})')

table = pd.DataFrame(results)
table.loc['MEAN'] = ['mean', '', table.best_iter.mean(), *table[['f1_raw', 'f1_scaled', 'f1_raw_testlike',
                                                                  'f1_scaled_testlike']].mean()]
print('\nMacro-F1 per held-out month (scaled = per-class multipliers, cross-fitted over that month\'s senders):')
print(table.round(4).to_string())

labels = pd.DataFrame(per_label)
labels['mean'] = labels.mean(axis=1)
print('\nPer-label F1 (per-class scaled) per held-out month:')
print(labels.round(3).to_string())

OUTPUTS.mkdir(exist_ok=True)
table.to_csv(OUTPUTS / 'cv_lgbm_big_joined_smote.csv')
log(f'wrote {OUTPUTS / "cv_lgbm_big_joined_smote.csv"}')
