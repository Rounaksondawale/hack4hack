"""Which columns the model may use, so it learns fraud logic instead of training-period quirks.

    from feature_policy import model_matrix
    X = model_matrix(df)          # same call for train, validation and test

Nothing is deleted from the parquet files; the policy is applied when building the model matrix.
Evidence for every entry: Sep vs October-test drift check (PSI + adversarial validation, 2026-11-14)
and the PROBLEM_STATEMENT notes (festival season, new app version for everyone, fraud rings adapt).
"""
import numpy as np
import pandas as pd

IDS = ['txn_id', 'timestamp', 'label', 'sender_id', 'receiver_id', 'device_id',
       's_opened_date', 'r_opened_date', 'm_onboard_date']

DROP = {
    # October pushed a new app version to everyone: test is 5.0-5.2, train 3.0-4.8 (PSI 6.9, alone
    # separates Sep from test perfectly). Fraud's "older version" signal cannot exist in test.
    'app_version': 'new app version for everyone in October',
    # vendor score changed scale (Sep p99 0.58 vs test 0.47) -> replaced by rb_ip_rank_day and products of it
    'ip_risk_score': 'vendor scale changed; use rb_ip_rank_day',
    'f_ip_x_fresh_dev': 'raw ip score; use rb_ip_rank_x_fresh_dev',
    'vb_burst_ip_risk': 'raw ip score; use rb_burst_ip_rank',
    'mc_round_x_ip': 'raw ip score; use rb_round_x_ip_rank',
    # festival season: amounts about 17% higher in October -> rupee levels shift; use relative amounts
    'amount': 'festival inflation; use rb_amt_rel_day / ratios',
    'balance_before': 'rupee level; use rb_bal_rel_day / ratios',
    'cp_log_amt': 'same as amount',
    'f_bal_after': 'rupee level',
    'f_amt_sum_1h': 'rupee level',
    'f_amt_sum_24h': 'rupee level',
    'mu_s_in_sum_24h_before': 'rupee level',
    'mc_unexplained_refill': 'rupee level; mc_refill_ratio keeps the idea',
    'vb_burst_amt_max': 'rupee level',
    'vb_burst_amt_std': 'rupee level',
    'f_amt_z': 'all-history z-score drifts (Sep 239 vs test 53); use rb_amt_vs_s_med30',
    # identities, not behaviour: a ring living in one city or bank in training says nothing about October
    'device_city': 'identity (hundreds of values)',
    's_home_city': 'identity', 'r_home_city': 'identity', 'm_city': 'identity',
    's_bank': 'identity', 'r_bank': 'identity',
    'm_category': 'duplicate of merchant_category',
    # ID-encoding artifacts of the synthetic data, not something a payments company would know
    'f_own_device': 'device-id digits = account digits (ID artifact); use rb_dev_nsnd_30d',
    'vb_own_device': 'ID artifact',
    'mc_bal_eq_1p1': 'exact 1.1x balance is a generator quirk; mc_bal_amt_ratio keeps the idea',
    'mc_bal_eq_1p1_round': 'generator quirk',
    # "all history" counts grow with how much history sits behind a row (test has more) -> fixed windows
    'f_n_prior_txn': 'all-history; use rb_s_n_30d',
    'cp_r_n_prior': 'all-history; use rb_r_n_30d',
    'cp_pair_n_prior': 'all-history; use rb_pair_n_30d',
    'cp_first_pair': 'all-history; use rb_pair_new_30d',
    'cp_first_pair_not_new_payee': 'all-history',
    'cp_r_has_sent': 'all-history; use rb_r_sent_30d',
    'qm_r_repeat_share': 'all-history; use rb_r_repeat_share_30d',
    'qm_r_round_share': 'all-history; use rb_r_round_share_30d',
    'mu_r_life_out_in': 'all-history',
    'mu_s_life_in': 'all-history',
    'mc_s_n_merchants_round': 'all-history',
    'f_device_n_senders': 'all-history; use rb_dev_nsnd_30d',
    'f_first_sender_device': 'all-history',
    'f_first_sender_city': 'all-history',
    'f_gap_h': 'uncapped gap (PSI 0.14); use rb_s_gap_h',
    'f_prev_gap_h': 'uncapped gap',
    # rb_ ideas that the data doesn't show (fire on <= 0.1% of their target class): pure noise for the model
    'rb_r_collect_nsnd_7d': 'phishers send one request per victim account here',
    'rb_r_qr_nsnd_7d': 'each fake QR receives only one payment here',
    'rb_pair_evening_7d': 'loan-app victims pay different lending merchants (rb_s_lending_7d covers it)',
}

# gaps: anything longer than the cap means "nothing recent" -> NaN, so longer test history can't shift them
CAP = {
    'cp_pair_gap_h': 168, 'cp_rev_gap_h': 168, 'cp_rev_tiny_gap_h': 168, 'mc_gap_prev_round_h': 168,
    'vb_dev_gap_prev_s': 7 * 86400, 'vb_dev_gap_next_s': 7 * 86400, 'vb_rcv_gap_prev_s': 7 * 86400,
}


# Test has about half the legit traffic per day while fraud bursts are unchanged, so gaps between ordinary
# payments stretch in October. Their percentile within the calendar day (computed inside each file) is
# unchanged by thinning (rb_s_gap_h PSI 0.10 -> 0.03). Only continuous gaps are ranked: for counts most rows
# tie on one value, the tied group's percentile moves with its share, and ranking made drift far worse
# (PSI up to 7), so counts stay raw (their drift is 0.02-0.05).
RANK_DAY = ['rb_s_gap_h', 'vb_dev_gap_prev_s', 'vb_dev_gap_next_s', 'vb_rcv_gap_prev_s', 'mu_back_gap_min']


def model_matrix(df):
    """Feature matrix for LightGBM: policy drops, gap caps, day ranks, categories, fixed column order."""
    X = df.drop(columns=[c for c in IDS + list(DROP) if c in df.columns])
    for c, cap in CAP.items():
        if c in X.columns:
            X[c] = X[c].where(X[c] <= cap)
    day = df.timestamp.dt.date
    for c in RANK_DAY:
        if c in X.columns:
            X[c] = X[c].groupby(day).rank(pct=True)
    for c in X.select_dtypes('object').columns:
        X[c] = X[c].astype('category')
    return X[sorted(X.columns)]  # train and test files store columns in different orders
