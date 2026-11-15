"""Add the 40 f_* features to exp/work/test_joined.parquet (train already has them).

    python code/add_f_features_test.py

Runs add_features() from exp/add_ato_simswap_features.ipynb (cell 2, the single source of that code)
on test, with train as sender history, as the notebook intends: add_features(test, history=train).
Every other column of test is left untouched.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'exp' / 'work'
NB = ROOT / 'exp' / 'add_ato_simswap_features.ipynb'

cell = json.loads(NB.read_text(encoding='utf-8'))['cells'][2]
src = ''.join(cell['source'])
assert 'def add_features(df, history=None)' in src, 'notebook cell 2 no longer holds add_features'
ns = {'np': np, 'pd': pd}
exec(src, ns)
add_features = ns['add_features']

tr = pd.read_parquet(WORK / 'train_joined.parquet')
te = pd.read_parquet(WORK / 'test_joined.parquet')
te_cols = [c for c in te.columns if not c.startswith('f_')]
ids = te.txn_id.to_numpy().copy()

te = add_features(te, history=tr)
f_tr = [c for c in tr.columns if c.startswith('f_')]
f_te = [c for c in te.columns if c.startswith('f_')]
assert sorted(f_te) == sorted(f_tr), 'test f_ columns differ from train'
assert (te.txn_id.to_numpy() == ids).all()
assert [c for c in te.columns if not c.startswith('f_')] == te_cols
te.to_parquet(WORK / 'test_joined.parquet', index=False)
print(f'{len(f_te)} f_ features written | test {te.shape}')
