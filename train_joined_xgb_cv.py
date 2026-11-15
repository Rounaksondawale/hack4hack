"""XGBoost version of train_joined_lgbm_smote_cv.py, with 26 redundant columns dropped and plain random resampling.

    python code/train_joined_xgb_cv.py

Folds: Jul+Aug -> Sep, Jul+Sep -> Aug, Aug+Sep -> Jul (October rows count as September).
Per fold, training months only: keep 20% of legit, randomly repeat each fraud class up to 20,000 rows.
The held-out month is NOT resampled (real class mix) and is used for early stopping.
NaN stays NaN (XGBoost handles it), so no -999 filling.
"""
import gc

import numpy as np
import pandas as pd
import xgboost as xgb

from train_model import LABELS, ROOT, OUTPUTS, class_weights, crossfit_scaled, f1_table, log, macro_f1

DATA = ROOT / 'exp' / 'work' / 'train_joined.parquet'
DROP = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
        's_opened_date', 'r_opened_date', 'm_onboard_date']
# Redundant for trees (checked on the full train file): exact duplicates, 0/1 cut-offs of a kept numeric column,
# and monotonic transforms (log / inverse) of a kept column.
REDUNDANT = [
    'vb_own_device', 'mc_night',                                          # duplicates of f_own_device, rb_night
    'f_sim_lt_24h', 'f_sim_lt_48h', 'f_sim_lt_168h',                      # sim_age_hours
    'f_dev_lt_1d', 'f_dev_lt_2d', 'f_dev_lt_7d',                          # device_age_days
    'f_amt_bal_gt50', 'f_amt_bal_gt80',                                   # f_amt_bal_ratio
    'cp_amt_tiny', 'cp_amt_large',                                        # amount
    'cp_just_under_round',                                                # cp_under_round_gap
    'cp_first_pair',                                                      # cp_pair_n_prior
    'rb_pair_new_30d',                                                    # rb_pair_n_30d
    'vb_login_fast',                                                      # login_to_txn_sec
    'vb_farm_flag',                                                       # vb_dev_nsnd_5m
    'rb_night', 'rb_late_evening',                                        # rb_hour
    'f_pin_retry_any',                                                    # pin_retries
    'lc_in_hop', 'lc_out_hop',                                            # lc_depth_up, lc_depth_down
    'f_log_sim_age', 'f_log_login', 'cp_log_amt', 'mc_bal_amt_ratio',     # log / inverse of kept columns
]
LEGIT_KEEP = 0.2       # drop 80% of legit
FRAUD_TARGET = 20_000  # repeat each fraud class up to this many rows
MONTHS = ['2026-07', '2026-08', '2026-09']
PARAMS = dict(objective='multi:softprob', num_class=len(LABELS), eval_metric='mlogloss', device='cuda',
              tree_method='hist', learning_rate=0.05, max_depth=8, min_child_weight=5, subsample=0.8,
              colsample_bytree=0.5, max_cat_to_onehot=1, max_bin=128, seed=0)  # 128 bins: fits the 4 GB GPU

df = pd.read_parquet(DATA)
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09').to_numpy()
y = df.label.map({name: i for i, name in enumerate(LABELS)}).to_numpy()
assert not np.isnan(y).any(), 'unknown label in train_joined.parquet'
y = y.astype(int)

sender = df.sender_id.to_numpy()
X = df.drop(columns=DROP + REDUNDANT)
del df  # free memory: the full frame is not needed after this
for c in X.columns:
    if X[c].dtype == object:
        X[c] = X[c].astype('category')  # same categories on every row subset
    else:
        X[c] = X[c].astype(np.float32)  # XGBoost bins in float32 anyway; halves memory
log(f'{X.shape[1]} features')


def resample(yf, rng):
    """Row indices: 20% of legit, each fraud class repeated (with replacement) up to FRAUD_TARGET."""
    parts = [np.flatnonzero((yf == 0) & (rng.random(len(yf)) < LEGIT_KEEP))]
    for c in range(1, len(LABELS)):
        idx = np.flatnonzero(yf == c)
        extra = rng.choice(idx, max(FRAUD_TARGET - len(idx), 0), replace=True)
        parts += [idx, extra]
    return np.concatenate(parts)


results, per_label = [], {}
for held in MONTHS:
    train_months = [m for m in MONTHS if m != held]
    fit_rows, val_rows = np.isin(month, train_months), month == held
    log(f'===== fold: train {"+".join(train_months)} -> score {held}')

    pick = np.flatnonzero(fit_rows)[resample(y[fit_rows], np.random.default_rng(0))]
    Xf, yf = X.iloc[pick], y[pick]  # one copy of the resampled rows only
    Xv, yv = X[val_rows], y[val_rows]
    log(f'fit {len(yf):,} rows (resampled, per class {np.bincount(yf).tolist()}), score {len(yv):,} rows ({held})')

    dtrain = xgb.QuantileDMatrix(Xf, yf, enable_categorical=True, max_bin=PARAMS['max_bin'])  # compressed bins
    dvalid = xgb.QuantileDMatrix(Xv, yv, weight=class_weights(yv), enable_categorical=True, ref=dtrain,
                                max_bin=PARAMS['max_bin'])
    del Xf, Xv
    m = xgb.train(PARAMS, dtrain, 3000, evals=[(dvalid, 'valid')], early_stopping_rounds=100, verbose_eval=100)
    p = m.predict(dvalid, iteration_range=(0, m.best_iteration + 1))

    raw = p.argmax(1)
    scaled = crossfit_scaled(yv, p, pd.factorize(sender[val_rows])[0])
    results.append(dict(scored=held, trained_on='+'.join(train_months), best_iter=m.best_iteration,
                        f1_raw=macro_f1(yv, raw), f1_scaled=macro_f1(yv, scaled),
                        f1_raw_testlike=macro_f1(yv, raw, 2), f1_scaled_testlike=macro_f1(yv, scaled, 2)))
    per_label[held] = f1_table(yv, scaled).f1
    log(f'{held}: macro-F1 raw {results[-1]["f1_raw"]:.4f}, scaled {results[-1]["f1_scaled"]:.4f} '
        f'(best iteration {m.best_iteration})')
    del m, dtrain, dvalid, p
    gc.collect()  # free this fold's model and data before the next fold

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
table.to_csv(OUTPUTS / 'cv_xgb_joined_resampled.csv')
log(f'wrote {OUTPUTS / "cv_xgb_joined_resampled.csv"}')
