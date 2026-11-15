"""Label-noise flags on the robust feature set, and does down-weighting noisy rows help?

1. Out-of-fold (OOF) probabilities, 5 folds grouped by calendar day, LightGBM on model_matrix features
   (same recipe as exp/clean_labels.ipynb: all fraud + 15% legit per fold, legit odds corrected afterwards).
   - run A: on Jul-Aug rows only  -> weights for the Jul-Aug -> Sep experiment (no September label leaks in)
   - run B: on all train rows      -> exp/noise_flags_v2.csv (adds p_given) for the final refit
2. lgbm_big fit Jul-Aug, scored on Sep, three ways:
   base     class weights only (current robust model)
   gce      class weight x p_given^0.7, renormalised per class (gradient of generalized cross-entropy q=0.7,
            i.e. between cross-entropy and the noise-robust MAE)
   drop     flagged rows (model disagrees with >= 90% confidence) removed
Scores: Sep macro-F1, test-like (legit x2), and on the Sep rows where the OOF model agrees with the label.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from feature_policy import model_matrix
from train_model import LABELS, MODELS, ROOT, PARAMS, class_weights, f1_table, log, macro_f1

LEGIT_KEEP, TH, Q = 0.15, 0.90, 0.7
K = len(LABELS)
OOF_PARAMS = dict(objective='multiclass', num_class=K, learning_rate=0.1, num_leaves=63, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1)


def oof_proba(X, y, groups):
    rng = np.random.default_rng(42)
    oof = np.zeros((len(y), K))
    for f, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        tr = tr[(y[tr] != 0) | (rng.random(len(tr)) < LEGIT_KEEP)]
        cnt = np.bincount(y[tr], minlength=K)
        w = 1 / np.sqrt(cnt[y[tr]])
        m = lgb.train(OOF_PARAMS, lgb.Dataset(X.iloc[tr], y[tr], weight=w / w.mean()), 300)
        oof[va] = m.predict(X.iloc[va])
        log(f'  fold {f}: fit {len(tr)} rows, scored {len(va)}')
    oof[:, 0] /= LEGIT_KEEP
    return oof / oof.sum(1, keepdims=True)


def flags(oof, y):
    pred = oof.argmax(1)
    p_given = oof[np.arange(len(y)), y]
    return pred, p_given, (pred != y) & (oof.max(1) >= TH)


df = pd.read_parquet(ROOT / 'exp' / 'work' / 'train_joined.parquet')
X = model_matrix(df)
y = df.label.map({n: i for i, n in enumerate(LABELS)}).to_numpy().astype(int)
day = df.timestamp.dt.date.astype(str).to_numpy()
month = df.timestamp.dt.strftime('%Y-%m').replace('2026-10', '2026-09')
fit, val = month.isin(['2026-07', '2026-08']).to_numpy(), (month == '2026-09').to_numpy()
log(f'{X.shape[1]} features, {len(y)} rows')

log('OOF run A: Jul-Aug only')
oofA = oof_proba(X[fit], y[fit], day[fit])
predA, pgA, flagA = flags(oofA, y[fit])

log('OOF run B: all train rows')
oofB = oof_proba(X, y, day)
predB, pgB, flagB = flags(oofB, y)
out = pd.DataFrame({'txn_id': df.txn_id, 'label': df.label, 'suggested': np.array(LABELS)[predB],
                    'p_suggested': oofB.max(1), 'p_given': pgB, 'flag': flagB, 'keep_for_training': ~flagB})
out.to_csv(ROOT / 'exp' / 'noise_flags_v2.csv', index=False)
log('wrote exp/noise_flags_v2.csv')

old = pd.read_csv(ROOT / 'exp' / 'noise_flags.csv').set_index('txn_id').reindex(df.txn_id)
g =pd.DataFrame({'label': df.label.to_numpy(), 'agree_old': old.suggested.to_numpy() == df.label.to_numpy(),
                  'agree_new': predB == y, 'flag_old': old.flag.to_numpy().astype(bool), 'flag_new': flagB,
                  'p_given': pgB}).groupby('label')
stats = pd.DataFrame({'rows': g.size(), 'agree_old_%': g.agree_old.mean() * 100, 'agree_new_%': g.agree_new.mean() * 100,
                      'flagged_old_%': g.flag_old.mean() * 100, 'flagged_new_%': g.flag_new.mean() * 100,
                      'median_p_given': g.p_given.median()}).reindex(LABELS)
print('\n=== NOISE FLAGS: old (notebook) vs new (robust features) ===')
print(stats.round(2).to_string(), flush=True)
print(f'OOF macro-F1 on given labels: {macro_f1(y, predB):.4f}', flush=True)

# ---- training variants
Xf, yf, Xv, yv = X[fit], y[fit], X[val], y[val]
agree_val = (predB[val] == yv)  # Sep rows whose label the OOF model believes
cw = class_weights(yf)
g = pgA ** Q
norm = np.bincount(yf, weights=g, minlength=K) / np.bincount(yf, minlength=K)
w_gce = cw * g / norm[yf]
variants = {'base': (np.ones(len(yf), bool), cw),
            'gce': (np.ones(len(yf), bool), w_gce),
            'drop': (~flagA, None)}


def train(keep, w):
    Xk, yk = Xf[keep], yf[keep]
    w = class_weights(yk) if w is None else w[keep]
    params = {**PARAMS, **MODELS['lgbm_big']().params}
    m = lgb.train(params, lgb.Dataset(Xk, yk, weight=w), 3000,
                  valid_sets=[lgb.Dataset(Xv, yv, weight=class_weights(yv))],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
    return m, m.predict(Xv, num_iteration=m.best_iteration)


rows, per = {}, {}
for name, (keep, w) in variants.items():
    log(f'training {name}: {keep.sum()} rows')
    m, p = train(keep, w)
    a = p.argmax(1)
    rows[name] = {'rows': int(keep.sum()), 'best_iter': m.best_iteration,
                  'Sep macro-F1': macro_f1(yv, a), 'Sep test-like': macro_f1(yv, a, 2),
                  'Sep agreed-rows macro-F1': macro_f1(yv[agree_val], a[agree_val])}
    per[name] = f1_table(yv, a).f1
    print(pd.Series(rows[name]).round(4).to_string(), flush=True)
    np.save(ROOT / 'outputs' / f'noise_val_proba_{name}.npy', p)

print('\n=== SUMMARY ===')
print(pd.DataFrame(rows).T.round(4).to_string())
print(pd.DataFrame(per).round(3).to_string())
