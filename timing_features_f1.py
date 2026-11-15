"""Compare macro-F1 of timing-feature sets (last gap, windows, last-3 stats) on a time-based split.

Train: Jul-Aug 2026.  Validation: Sep 2026 (mimics "predict the next month").
Same LightGBM settings for every feature set, so only the features differ.
Timing features look backwards only (per sender), so computing them on all of train leaks no labels.

Outputs (./timing_features_results):
  f1_matrix.csv         rows = feature sets, columns = macro F1 + per-class F1 + delta vs baseline
  f1_comparison.png     macro-F1 bars + per-class F1 change vs baseline
"""
import json
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import f1_score

DATA = r"C:\Hackathon\hack4hack\data\01_raw\train.csv"
OUT = Path(__file__).parent / "timing_features_results"
SPLIT = pd.Timestamp("2026-09-01")
CLASSES = ["legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
           "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion", "merchant_collusion"]
WINDOWS = {"10min": 600, "1hour": 3600, "1day": 86400}

NUMERIC = ["amount", "sim_age_hours", "device_age_days", "sender_account_age_days", "new_payee", "pin_retries",
           "login_to_txn_sec", "balance_before", "app_version", "ip_risk_score", "qr_vpa_mismatch"]
CATEGORICAL = ["receiver_type", "txn_type", "merchant_category", "device_city", "app"]
BASE = NUMERIC + CATEGORICAL + ["hour", "dow"]
GROUPS = {
    "last_gap": ["last_gap"],
    "windows": [f"cnt_{w}" for w in WINDOWS],
    "last3_stats": ["mean_gap_3", "min_gap_3"],
}
FEATURE_SETS = {
    "A baseline": BASE,
    "B + last gap": BASE + GROUPS["last_gap"],
    "C + windows (10m/1h/1d)": BASE + GROUPS["windows"],
    "D + last-3 stats": BASE + GROUPS["last3_stats"],
    "E + all timing": BASE + sum(GROUPS.values(), []),
}
PARAMS = dict(objective="multiclass", num_class=len(CLASSES), learning_rate=0.1, num_leaves=63,
              min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              max_bin=127, num_threads=16, seed=42, verbose=-1)
ROUNDS = 200


def build():
    df = pd.read_csv(DATA)
    df["ts"] = pd.to_datetime(df["timestamp"])
    df["sec"] = (df["ts"] - df["ts"].min()).dt.total_seconds()
    df = df.sort_values(["sender_id", "sec"], kind="stable").reset_index(drop=True)
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")

    by = df.groupby("sender_id")
    df["last_gap"] = by["sec"].diff()                       # NaN for a sender's first txn
    g = df.groupby("sender_id")["last_gap"]
    gaps = pd.concat([df["last_gap"], g.shift(1), g.shift(2)], axis=1)  # gaps of the last 3 txns
    df["mean_gap_3"] = gaps.mean(axis=1)
    df["min_gap_3"] = gaps.min(axis=1)

    code = df["sender_id"].astype("category").cat.codes.to_numpy().astype("int64")
    big = int(df["sec"].max()) + max(WINDOWS.values()) + 1
    key = code * big + df["sec"].to_numpy().astype("int64")
    for name, sec in WINDOWS.items():
        start = np.searchsorted(key, key - sec, side="left")
        df[f"cnt_{name}"] = np.arange(len(df)) - start      # earlier txns by this sender in the window
    df["y"] = df["label"].map({c: i for i, c in enumerate(CLASSES)})
    return df


def run(df):
    tr, va = df[df["ts"] < SPLIT], df[df["ts"] >= SPLIT]
    print(f"train {len(tr):,}  valid {len(va):,}", flush=True)
    rows = []
    for name, feats in FEATURE_SETS.items():
        t0 = time.time()
        model = lgb.train(PARAMS, lgb.Dataset(tr[feats], tr["y"]), num_boost_round=ROUNDS)
        pred = model.predict(va[feats]).argmax(axis=1)
        per = f1_score(va["y"], pred, labels=range(len(CLASSES)), average=None, zero_division=0)
        rows.append({"feature_set": name, "n_features": len(feats), "macro_f1": per.mean(),
                     **dict(zip(CLASSES, per))})
        print(f"{name:28s} macro-F1 {per.mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)
    res = pd.DataFrame(rows).set_index("feature_set")
    res.insert(1, "delta_vs_baseline", res["macro_f1"] - res.loc["A baseline", "macro_f1"])
    return res


def plot(res):
    ink, muted, blue = "#222222", "#6b6b6b", "#2a6fdb"
    delta = res[CLASSES].sub(res.loc["A baseline", CLASSES], axis=1).drop(index="A baseline")
    cmap = LinearSegmentedColormap.from_list("div", ["#d95f02", "#f2f2f2", blue])
    lim = max(0.02, float(np.abs(delta.to_numpy()).max()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), gridspec_kw={"height_ratios": [1, 1.25]})
    sets = res.index.tolist()
    bars = ax1.barh(sets, res["macro_f1"], color=[muted] + [blue] * (len(sets) - 1), height=0.6)
    ax1.invert_yaxis()
    ax1.set_xlim(res["macro_f1"].min() * 0.9, res["macro_f1"].max() * 1.06)
    for b, f1, d in zip(bars, res["macro_f1"], res["delta_vs_baseline"]):
        txt = f"{f1:.4f}" + ("" if d == 0 else f"  ({d:+.4f})")
        ax1.annotate(txt, (f1, b.get_y() + b.get_height() / 2), xytext=(5, 0), textcoords="offset points",
                     va="center", fontsize=10, color=ink)
    ax1.set_title("Macro-F1 on September (trained on Jul-Aug), 12 classes", loc="left", color=ink)
    ax1.set_xlabel("Macro-F1 (axis starts above zero)", color=ink)
    ax1.spines[["top", "right"]].set_visible(False)

    im = ax2.imshow(delta.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax2.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right")
    ax2.set_yticks(range(len(delta)), delta.index)
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            v = delta.iat[i, j]
            ax2.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8, color=ink)
    ax2.set_title("Per-class F1 change vs baseline (blue = better, orange = worse)", loc="left", color=ink)
    ax2.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax2, label="F1 change", fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(OUT / "f1_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    result = run(build())
    result.round(4).to_csv(OUT / "f1_matrix.csv")
    plot(result)
    print("\n", result.round(3).to_string())
    print("\nSaved to", OUT)
