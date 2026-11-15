"""Train on train.parquet, predict test.parquet, write outputs/submissions/submission_v9.csv (format: id,label).

    python code/make_submission_v9.py

Same recipe as the best row of train_model.py (lgbm_big + label history + per-class probability scaling):
  1. fit on Jul-Aug, early-stop on Sep -> best round count and Sep probabilities
  2. tune per-class multipliers on those Sep probabilities (macro-F1, test-like class mix)
  3. refit on Jul-Sep with that round count, predict test, apply the multipliers, write the submission
Needs data/02_processed/features/{train,test}.parquet (python code/prepare_data.py).
"""
import numpy as np
import pandas as pd

from train_model import (FINAL_SOURCES, LABELS, MODELS, MONTHS, OUTPUTS, VAL_SOURCES, label_history, load_features,
                         log, macro_f1, tune_scales)
from write_submission import write_submission

OUT = OUTPUTS / 'submissions' / 'submission_v9.csv'


def main():
    C, feats = load_features()
    for c in feats:  # the parquet stores these as strings; LightGBM needs category (one set of categories for train + test)
        if C[c].dtype == object:
            C[c] = C[c].astype('category')
    y = C.label.map({name: i for i, name in enumerate(LABELS)}).fillna(-1).astype(int).to_numpy()
    per = C.period.to_numpy()
    fit_rows, val_rows, train_rows, test_rows = np.isin(per, MONTHS[:2]), per == MONTHS[2], per != 'test', per == 'test'
    log(f'fit {fit_rows.sum():,} (Jul-Aug), validate {val_rows.sum():,} (Sep), test {test_rows.sum():,}, {len(feats)} features')

    # 1. validation fit: round count + Sep probabilities
    X = pd.concat([C[feats], label_history(C, VAL_SOURCES)], axis=1)
    model = MODELS['lgbm_big']()
    p_val = model.validate(X[fit_rows], y[fit_rows], X[val_rows], y[val_rows])
    log(f'best round {model.rounds}; Sep macro-F1 raw {macro_f1(y[val_rows], p_val.argmax(1)):.4f}')

    # 2. per-class multipliers
    scales = tune_scales(y[val_rows], p_val)
    log(f'Sep macro-F1 scaled (in-sample) {macro_f1(y[val_rows], (p_val * scales).argmax(1)):.4f}')
    print(pd.Series(scales, index=LABELS).round(2).to_string())
    model.m = None  # free the booster

    # 3. refit on Jul-Sep, predict test
    X = pd.concat([C[feats], label_history(C, FINAL_SOURCES)], axis=1)
    p_test = model.final(X[train_rows], y[train_rows], X[test_rows])
    pred = (p_test * scales).argmax(1)

    write_submission(C.txn_id[test_rows].to_numpy(), np.array(LABELS)[pred], OUT)
    print('\nPredicted label share of test:')
    print(pd.Series(np.array(LABELS)[pred]).value_counts(normalize=True).reindex(LABELS).round(4).to_string())


if __name__ == '__main__':
    main()
