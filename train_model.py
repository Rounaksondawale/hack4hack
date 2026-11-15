"""Compare several models on the features from prepare_data.py (macro-F1), then write a submission.

    python code/prepare_data.py                   # once: writes data/02_processed/features/{train,test}.parquet
    python code/train_model.py                    # compare all models: fit Jul-Aug, score Sep; refit best on Jul-Sep
    python code/train_model.py --models lgbm,hgb  # only these models
    python code/train_model.py --features history,neighbors   # compare feature sets (prepared | history | neighbors)
    python code/train_model.py --final none       # comparison only, write no submission
    python code/train_model.py --scale off        # skip per-class probability scaling

Every model is scored twice on September: raw argmax, and with per-class probability multipliers tuned for macro-F1
(cross-fitted over September's senders, so the scaled score is not tuned on the rows it is scored on).

Models: lgbm (LightGBM), lgbm_big (bigger/slower LightGBM), hgb (sklearn HistGradientBoosting),
et (ExtraTrees), rf (RandomForest); 'blend' = mean of all chosen models' probabilities.
Results table -> outputs/model_comparison.csv, submission -> outputs/submissions/submission_<model>.csv.
"""
import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from prepare_data import DATA as RAW, DAY, HOUR, KEY_COLS, Windows
from write_submission import write_submission

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / 'data' / '02_processed' / 'features'  # written by prepare_data.py
OUTPUTS = ROOT / 'outputs'
LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover',
          'phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
MONTHS = ['2026-07', '2026-08', '2026-09']
PARAMS = dict(objective='multiclass', num_class=len(LABELS), learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, seed=0, verbose=-1,
              deterministic=True, force_row_wise=True)  # same predictions on every run

# Label history: the months whose labels each period's rows may use (never their own month).
VAL_SOURCES = {'2026-07': ['2026-08'], '2026-08': ['2026-07'], '2026-09': ['2026-07', '2026-08']}
FINAL_SOURCES = {'2026-07': ['2026-08', '2026-09'], '2026-08': ['2026-07', '2026-09'],
                 '2026-09': ['2026-07', '2026-08'], 'test': MONTHS}
FEATURE_SETS = {'prepared': 'prepared', 'history': '+history', 'neighbors': '+history+neighbors'}  # name -> table tag
HIST_KEYS = {'snd': 'sender_id', 'rcv': 'receiver_id', 'dev': 'device_id'}


def log(msg, t0=time.time()):
    print(f'[{time.time() - t0:6.0f}s] {msg}', flush=True)


def class_weights(y):
    """Inverse square root of class size, legit = 1 (every class counts equally in macro-F1)."""
    n = np.bincount(y, minlength=len(LABELS))
    return np.sqrt(n[0] / np.maximum(n, 1))[y]


def full_proba(model, p):
    """sklearn drops classes absent from the training rows; put the columns back into LABELS order."""
    out = np.zeros((len(p), len(LABELS)))
    out[:, model.classes_] = p
    return out


def to_numeric(X):
    """Float32 matrix for the forests: categories -> codes (-1 = missing), +-inf -> NaN."""
    X = X.copy()
    for c in X.select_dtypes('category').columns:
        X[c] = X[c].cat.codes
    a = X.to_numpy(np.float32)
    a[np.isinf(a)] = np.nan
    return a


# ---------------------------------------------------------------- models
# Each model: validate(X, y, Xv, yv) -> probabilities for Xv (remembers the size it needs for the refit),
#             final(X, y, Xt) -> probabilities for Xt after refitting on all training rows.
class LGBM:
    numeric = False

    def __init__(self, **params):
        self.params = {**PARAMS, **params}
        self.m = None

    def validate(self, X, y, Xv, yv):
        dtrain = lgb.Dataset(X, y, weight=class_weights(y))
        dvalid = lgb.Dataset(Xv, yv, weight=class_weights(yv), reference=dtrain)
        self.m = lgb.train(self.params, dtrain, 3000, valid_sets=[dvalid],
                           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)])
        self.rounds = self.m.best_iteration
        return self.m.predict(Xv, num_iteration=self.rounds)

    def final(self, X, y, Xt):
        # ~10% more rounds than the best iteration, as there is 50% more data.
        m = lgb.train(self.params, lgb.Dataset(X, y, weight=class_weights(y)), int(self.rounds * 1.1))
        return m.predict(Xt)

    def top_gain(self, columns, k=20):
        gain = pd.Series(self.m.feature_importance('gain'), index=columns).sort_values(ascending=False)
        return (gain / gain.sum()).head(k).round(3)


class HGB:
    numeric = False  # takes the DataFrame: native NaN and categorical columns

    def _make(self, **kw):
        return HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=200,
                                              l2_regularization=5.0, categorical_features='from_dtype',
                                              random_state=0, **kw)

    def validate(self, X, y, Xv, yv):
        m = self._make(max_iter=500, early_stopping=True, n_iter_no_change=20, validation_fraction=0.1)
        m.fit(X, y, sample_weight=class_weights(y))
        self.rounds = m.n_iter_
        return full_proba(m, m.predict_proba(Xv))

    def final(self, X, y, Xt):
        m = self._make(max_iter=int(self.rounds * 1.1), early_stopping=False)
        m.fit(X, y, sample_weight=class_weights(y))
        return full_proba(m, m.predict_proba(Xt))


class Forest:
    numeric = True

    def __init__(self, cls, **params):
        self.make = lambda: cls(n_estimators=150, min_samples_leaf=20, bootstrap=True, max_samples=0.2,
                                n_jobs=-1, random_state=0, **params)

    def validate(self, X, y, Xv, yv):
        m = self.make().fit(X, y, sample_weight=class_weights(y))
        return full_proba(m, m.predict_proba(Xv))

    def final(self, X, y, Xt):
        m = self.make().fit(X, y, sample_weight=class_weights(y))
        return full_proba(m, m.predict_proba(Xt))


MODELS = {
    # 'lgbm': lambda: LGBM(),
    'lgbm_big': lambda: LGBM(num_leaves=255, min_data_in_leaf=100, feature_fraction=0.5, learning_rate=0.03),
    # 'hgb': lambda: HGB(),
    # 'et': lambda: Forest(ExtraTreesClassifier, max_features=0.5),
    # 'rf': lambda: Forest(RandomForestClassifier, max_features=0.3),
}


# ---------------------------------------------------------------- metrics and history features
def f1_table(y, p):
    rows = {}
    for c, name in enumerate(LABELS):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        rows[name] = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                          f1=2 * tp / max(2 * tp + fp + fn, 1), support=tp + fn)
    return pd.DataFrame(rows).T


def macro_f1(y, p, legit_weight=1.0):
    """Macro-F1; legit_weight=2 mimics test, which has about half September's fraud share."""
    k = len(LABELS)
    w = np.where(y == 0, legit_weight, 1.0)
    cm = np.bincount(y * k + p, weights=w, minlength=k * k).reshape(k, k)  # rows = true, columns = predicted
    return np.mean(2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1))


# ---------------------------------------------------------------- per-class probability scaling
TUNE_LEGIT_WEIGHT = 2.0  # tune for the test-like mix (test has about half September's fraud share); 1.0 = plain Sep mix
SCALE_GRID = np.exp(np.linspace(-2, 2, 15))  # multipliers 0.14 ... 7.4, 1.0 included


def tune_scales(y, P, passes=2):
    """Per-class multipliers for the probabilities that maximize macro-F1 (coordinate ascent, legit fixed at 1)."""
    w = np.ones(len(LABELS))
    best = macro_f1(y, (P * w).argmax(1), TUNE_LEGIT_WEIGHT)
    for _ in range(passes):
        improved = False
        for c in range(1, len(LABELS)):
            for g in SCALE_GRID:
                trial = w.copy()
                trial[c] = g
                s = macro_f1(y, (P * trial).argmax(1), TUNE_LEGIT_WEIGHT)
                if s > best + 1e-6:
                    best, w, improved = s, trial, True
        if not improved:
            break
    return w


def crossfit_scaled(y, P, groups):
    """Honest scaled predictions: tune on half of September's senders, predict the other half, and swap."""
    fold = groups % 2
    pred = np.empty(len(y), dtype=int)
    for f in (0, 1):
        w = tune_scales(y[fold != f], P[fold != f])
        pred[fold == f] = (P[fold == f] * w).argmax(1)
    return pred


def report_confusion(y, p, title, save_to=None, top=12):
    """Print where each true class goes: row-normalized confusion matrix (% of the true class) and the biggest mistakes."""
    k = len(LABELS)
    cm = np.bincount(y * k + p, minlength=k * k).reshape(k, k)
    pct = pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100, index=[f'{i} {n}' for i, n in enumerate(LABELS)],
                       columns=[str(i) for i in range(k)])
    print(f'\nConfusion matrix, {title} (rows = true class, columns = predicted class number, % of the true class):')
    print(pct.round(1).to_string())
    errs = pd.DataFrame([dict(true=LABELS[i], predicted=LABELS[j], rows=cm[i, j], pct_of_true=100 * cm[i, j] / max(cm[i].sum(), 1),
                              pct_of_predicted=100 * cm[i, j] / max(cm[:, j].sum(), 1))
                         for i in range(k) for j in range(k) if i != j])
    print(f'\nBiggest mistakes, {title} (pct_of_true = how much of the true class goes there; '
          f'pct_of_predicted = how much of that prediction is this wrong class):')
    print(errs.sort_values('rows', ascending=False).head(top).round(1).to_string(index=False))
    if save_to is not None:
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(save_to)
        log(f'wrote {save_to}')


def label_history(C, sources):
    """Rows of each fraud type that a row's sender, P2P receiver and phone have in other months (per month).

    Fraud collection accounts and repeat victims come back across months. Merchants get NaN: busy
    merchants collect every fraud type, so their history carries no signal.
    """
    fraud = C[C.label.notna() & (C.label != 'legit')]
    cols = {k: [f'hist_{k}_{L}' for L in LABELS[1:]] for k in HIST_KEYS}
    out = pd.DataFrame(np.nan, index=C.index, columns=sum(cols.values(), []))
    for period, srcs in sources.items():
        rows = (C.period == period).to_numpy()
        src = fraud[fraud.period.isin(srcs)]
        for k, col in HIST_KEYS.items():
            counts = src.groupby([col, 'label']).size().unstack(fill_value=0).reindex(columns=LABELS[1:], fill_value=0)
            out.loc[rows, cols[k]] = counts.reindex(C.loc[rows, col]).fillna(0).to_numpy() / len(srcs)
    out.loc[(C.receiver_type != 'P2P').to_numpy(), cols['rcv']] = np.nan
    return out


# ---------------------------------------------------------------- neighbour features from a first-pass model
STAGE1 = {**PARAMS, 'learning_rate': 0.1, 'min_data_in_leaf': 100}
STAGE1_ROUNDS = 100  # fixed (no early stopping on the month being predicted)


def load_times(C):
    """Seconds since 2026-07-01 of every row; the feature files have no absolute time, the raw csv files do."""
    raw = pd.concat([pd.read_csv(RAW / f, usecols=['txn_id', 'timestamp']) for f in ('train.csv', 'test.csv')])
    ts = pd.to_datetime(C.txn_id.map(raw.set_index('txn_id').timestamp))
    assert ts.notna().all(), 'some txn_id has no timestamp in the raw csv files'
    return (ts - pd.Timestamp('2026-07-01')).dt.total_seconds().to_numpy().astype(np.int64)


def stage1_probs(C, feats, y, sources):
    """Class probabilities of every row from a first model that never saw that row's own month.

    sources: period -> months whose labels the model may train on (same maps as the label history).
    Uses the prepared features only (no label history), so no label of a period leaks into its own prediction.
    """
    per = C.period.to_numpy()
    P = np.zeros((len(C), len(LABELS)))
    for period, srcs in sources.items():
        tr, te = np.isin(per, srcs), per == period
        m = lgb.train(STAGE1, lgb.Dataset(C.loc[tr, feats], y[tr], weight=class_weights(y[tr])), STAGE1_ROUNDS)
        P[te] = m.predict(C.loc[te, feats])
        log(f'  first-pass model: train on {srcs}, predict {period}')
    return P


def neighbor_features(C, t, P):
    """What the first-pass model thinks of the OTHER rows around this one (same sender / receiver / phone).

    Fraud comes in rings and bursts, so a row whose neighbours look fraudulent is more likely fraud itself,
    even when the row looks normal alone. Centred time windows, the row itself excluded.
    """
    out = pd.DataFrame(index=C.index)
    for j, name in enumerate(LABELS):
        out[f'stage1_{name}'] = P[:, j]
    p_fraud = 1 - P[:, 0]
    for k, col in HIST_KEYS.items():
        win = Windows(C[col], t)

        def others_mean(v, w):
            n = win.count(w) - 1
            return np.where(n > 0, (win.sum(v, w) - v) / np.maximum(n, 1), np.nan)

        out[f'nbr_{k}_pfraud_1h'] = others_mean(p_fraud, HOUR)
        out[f'nbr_{k}_pfraud_24h'] = others_mean(p_fraud, DAY)
        for j in range(1, len(LABELS)):
            out[f'nbr_{k}_{LABELS[j]}_6h'] = others_mean(P[:, j], 6 * HOUR)
    return out


def load_features():
    tr = pd.read_parquet(FEATURES / 'train.parquet')
    te = pd.read_parquet(FEATURES / 'test.parquet').assign(period='test')
    C = pd.concat([tr, te], ignore_index=True)
    feats = [c for c in te.columns if c not in ['txn_id', 'period'] + KEY_COLS]
    return C, feats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', default=','.join(MODELS), help=f'comma-separated, from: {", ".join(MODELS)}')
    ap.add_argument('--features', default='history',
                    help='comma-separated feature sets to compare: prepared, history (+ label history, default), '
                         'neighbors (history + first-pass neighbour features)')
    ap.add_argument('--final', default='best',
                    help="what to refit on Jul-Sep and write as submission: best (default), none, 'blend' or a model name")
    ap.add_argument('--scale', choices=['on', 'off'], default='on',
                    help='also score every model with per-class probability scaling (default on)')
    a = ap.parse_args()
    names = a.models.split(',')
    assert all(n in MODELS for n in names), f'unknown model in --models; choose from {list(MODELS)}'
    assert a.final in ['best', 'none', 'blend'] + names, '--final must be best, none, blend or one of --models'
    hist_modes = a.features.split(',')  # feature-set names; the variable `hist` below holds one of them
    assert all(h in FEATURE_SETS for h in hist_modes), f'--features must be from {list(FEATURE_SETS)}'

    C, feats = load_features()
    y = C.label.map({name: i for i, name in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    per = C.period.to_numpy()
    fit_rows, val_rows, train_rows, test_rows = np.isin(per, MONTHS[:2]), per == MONTHS[2], per != 'test', per == 'test'
    log(f'train {fit_rows.sum():,} rows (Jul-Aug), validate {val_rows.sum():,} rows (Sep), {len(feats)} features')

    times, nbr_cache = {}, {}

    def feature_matrix(hist, sources):
        parts = [C[feats]]
        if hist in ('history', 'neighbors'):
            parts.append(label_history(C, sources))
        if hist == 'neighbors':
            scheme = 'final' if sources is FINAL_SOURCES else 'val'
            if scheme not in nbr_cache:
                log(f'first-pass models for the neighbour features ({scheme}) ...')
                if 't' not in times:
                    times['t'] = load_times(C)
                s1_sources = sources if scheme == 'final' else {**sources, 'test': MONTHS[:2]}  # val: test rows from Jul-Aug
                nbr_cache[scheme] = neighbor_features(C, times['t'], stage1_probs(C, feats, y, s1_sources))
            parts.append(nbr_cache[scheme])
        return pd.concat(parts, axis=1) if len(parts) > 1 else parts[0]

    def tag(hist):
        return FEATURE_SETS[hist]

    yv = y[val_rows]
    groups = pd.factorize(C.sender_id[val_rows])[0]  # scaling is tuned/scored on disjoint sets of senders

    # --- compare: fit Jul-Aug, score Sep
    results, val_proba, fitted, val_pred = [], {}, {}, {}

    def score(name, hist, p, seconds):
        """Add a raw row and (with --scale on) a per-class-scaled row; scaled scores are cross-fitted, so honest."""
        for scaling in (['none', 'per-class'] if a.scale == 'on' else ['none']):
            pred = p.argmax(1) if scaling == 'none' else crossfit_scaled(yv, p, groups)
            val_pred[name, hist, scaling] = pred
            results.append(dict(model=name, features=tag(hist), scaling=scaling, sep_f1=macro_f1(yv, pred),
                                sep_f1_testlike=macro_f1(yv, pred, 2), seconds=seconds))
            log(f'{name} [{tag(hist)}, {scaling}]: Sep macro-F1 {results[-1]["sep_f1"]:.4f}, test-like {results[-1]["sep_f1_testlike"]:.4f}')

    for hist in hist_modes:
        X = feature_matrix(hist, VAL_SOURCES)
        Xn = to_numeric(X) if any(MODELS[n]().numeric for n in names) else None
        for name in names:
            model, t0 = MODELS[name](), time.time()
            A = Xn if model.numeric else X
            log(f'{name} [{tag(hist)}] ...')
            p = model.validate(A[fit_rows], y[fit_rows], A[val_rows], y[val_rows])
            val_proba[name, hist] = p
            if name == 'lgbm':
                print(f'\nTop 20 features by gain ({tag(hist)}):\n{model.top_gain(X.columns).to_string()}\n')
            if hasattr(model, 'm'):
                model.m = None  # free the booster; only the round count is needed for the refit
            fitted[name, hist] = model
            score(name, hist, p, round(time.time() - t0))
        if len(names) > 1:
            val_proba['blend', hist] = np.mean([val_proba[n, hist] for n in names], axis=0)
            score('blend', hist, val_proba['blend', hist], 0)

    table = pd.DataFrame(results).sort_values('sep_f1', ascending=False).reset_index(drop=True)
    print('\nModel comparison on September (fit on Jul-Aug), best first.')
    print('Scaled rows: multipliers tuned on half of the senders, scored on the other half (and swapped).')
    print(table.round(4).to_string())
    if a.scale == 'on' and 'lgbm_big' in names:
        for hist in hist_modes:
            raw, sc = val_pred['lgbm_big', hist, 'none'], val_pred['lgbm_big', hist, 'per-class']
            rep = pd.DataFrame({'f1_raw': f1_table(yv, raw).f1, 'f1_scaled': f1_table(yv, sc).f1,
                                'scale': tune_scales(yv, val_proba['lgbm_big', hist])})  # scale: tuned on all of Sep
            rep.loc['MACRO'] = [macro_f1(yv, raw), macro_f1(yv, sc), np.nan]
            print(f'\nlgbm_big [{tag(hist)}]: per-label F1 on September, raw vs per-class scaled:')
            print(rep.round(3).to_string())
    OUTPUTS.mkdir(exist_ok=True)
    table.to_csv(OUTPUTS / 'model_comparison.csv', index=False)
    log(f'wrote {OUTPUTS / "model_comparison.csv"}')

    # Where the mistakes are: the best row of the table, plus raw lgbm_big if it is not that row.
    tag_to_set = {v: k for k, v in FEATURE_SETS.items()}
    targets = [(table.iloc[0].model, tag_to_set[table.iloc[0].features], table.iloc[0].scaling)]
    if 'lgbm_big' in names:
        targets += [('lgbm_big', hist, 'none') for hist in hist_modes if ('lgbm_big', hist, 'none') not in targets]
    for name, hist, scaling in targets:
        title = f'{name} [{tag(hist)}, {scaling}] on September'
        report_confusion(yv, val_pred[name, hist, scaling], title,
                         OUTPUTS / f'confusion_{name}_{tag(hist).strip("+")}_{scaling}.csv')

    if a.final == 'none':
        return

    # --- final: refit the chosen model(s) on all of Jul-Sep, predict test
    pool = table if a.final == 'best' else table[table.model == a.final]
    best = pool.iloc[0]
    hist, parts = tag_to_set[best.features], (names if best.model == 'blend' else [best.model])
    scaled = best.scaling == 'per-class'
    log(f'final: {best.model} [{best.features}, {best.scaling}] (Sep macro-F1 {best.sep_f1:.4f})')

    # Validation predictions of the winner (cross-fitted if scaled), for the reports below.
    pred_val = val_pred[best.model, hist, best.scaling]
    # Final multipliers: tuned on all of September's probabilities from the Jul-Aug model.
    scales = tune_scales(yv, val_proba[best.model, hist]) if scaled else np.ones(len(LABELS))
    if scaled:
        print('\nPer-class multipliers applied to the final probabilities:')
        print(pd.Series(scales, index=LABELS).round(2).to_string())
    print('\nPer-label results on September:')
    print(f1_table(yv, pred_val).round(3).to_string())

    X = feature_matrix(hist, FINAL_SOURCES)
    Xn = to_numeric(X) if any(MODELS[n]().numeric for n in parts) else None
    probas = []
    for name in parts:
        model = fitted[name, hist]  # keeps the round count found on September, not the fitted model
        A = Xn if model.numeric else X
        probas.append(model.final(A[train_rows], y[train_rows], A[test_rows]))
        log(f'refit {name} on Jul-Sep')
    pred = (np.mean(probas, axis=0) * scales).argmax(1)

    key = f'{best.model}{"" if hist == "prepared" else "_" + hist}{"_scaled" if scaled else ""}'
    write_submission(C.txn_id[test_rows].to_numpy(), np.array(LABELS)[pred], OUTPUTS / 'submissions' / f'submission_{key}.csv')
    log(f'wrote {OUTPUTS / "submissions" / f"submission_{key}.csv"}')

    print('\nPredicted label share per 10k rows: Sep validation vs test')
    shares = pd.concat([pd.Series(np.array(LABELS)[pred_val]).value_counts(normalize=True).rename('sep_pred'),
                        pd.Series(np.array(LABELS)[y[val_rows]]).value_counts(normalize=True).rename('sep_true'),
                        pd.Series(np.array(LABELS)[pred]).value_counts(normalize=True).rename('test_pred')], axis=1)
    print((shares * 1e4).round(1).reindex(LABELS).to_string())


if __name__ == '__main__':
    main()
