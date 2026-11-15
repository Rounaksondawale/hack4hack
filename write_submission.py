"""Write and validate a submission CSV in the format from PROBLEM_STATEMENT.md.

Format: header `id,label`, exactly one row per txn_id in test.csv, labels spelled exactly as the 12 below, file <= 15 MB.

As a library:  write_submission(test.txn_id, predicted_labels, 'outputs/submissions/x.csv')
As a script:   python code/write_submission.py <predictions.csv> <out.csv>   (predictions.csv has columns id,label)
"""
import os
import sys
import pandas as pd

LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover',
          'phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
TEST_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', '01_raw', 'test.csv')
MAX_BYTES = 15 * 1024 * 1024


def write_submission(ids, labels, path, test_csv=TEST_CSV):
    sub = pd.DataFrame({'id': list(ids), 'label': list(labels)})
    expected = pd.read_csv(test_csv, usecols=['txn_id']).txn_id

    bad = sorted(set(sub.label) - set(LABELS))
    assert not bad, f'labels not in the allowed 12: {bad}'
    assert sub.id.is_unique, 'duplicate ids'
    assert len(sub) == len(expected) and set(sub.id) == set(expected), \
        f'ids must match test.csv exactly: {len(sub)} rows vs {len(expected)} expected'

    sub = sub.set_index('id').loc[expected].reset_index()      # same order as test.csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sub.to_csv(path, index=False)
    size = os.path.getsize(path)
    assert size <= MAX_BYTES, f'{size / 1e6:.1f} MB is over the 15 MB upload limit'
    print(f'wrote {path}: {len(sub)} rows, {size / 1e6:.1f} MB')
    return sub


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    p = pd.read_csv(src)
    write_submission(p['id'], p['label'], dst)
