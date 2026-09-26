"""
Second half of the full symmetric grid (5-fold x 5-repeat = 25 folds, same
methodology/splits as verify_full_grid.py and the paper's Table 10):
  - the 5 non-CatBoost models under "balanced" (re-derive per-fold arrays so
    every config in the final comparison has matched, paired fold data --
    the existing Table 10 only has summary mean/std for these, not per-fold
    arrays, and a valid paired corrected t-test needs per-fold pairing)
  - CatBoost under "none" and "balanced" (same reason; also the slow part)

Because RepeatedStratifiedKFold(n_splits, n_repeats, random_state) applied to
the same (X_arr, y_0) ordering is fully deterministic, fold i here is
guaranteed identical to fold i in verify_full_grid.py -- safe to combine the
two per-fold CSVs afterwards for a fully paired 12-configuration comparison.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

from config.config import DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE
from src.preprocessing import prepare
from src.models import build_pipeline

OUT = Path(__file__).parent / "verify_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "verify_full_grid2_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


N_SPLITS, N_REPEATS = 5, 5

df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1
X_arr = X_all.reset_index(drop=True)

rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
splits = list(rcv.split(X_arr, y_0))
log(f"Total folds: {len(splits)} ({N_SPLITS}-fold x {N_REPEATS}-repeat)")

jobs = [
    ("logistic", "class_weight", "logistic_balanced"),
    ("decision_tree", "class_weight", "decision_tree_balanced"),
    ("random_forest", "class_weight", "random_forest_balanced"),
    ("xgboost", "class_weight", "xgboost_balanced"),
    ("lightgbm", "class_weight", "lightgbm_balanced"),
    ("catboost", "none", "catboost_none"),
    ("catboost", "class_weight", "catboost_balanced"),
]

per_fold = {}
summary = []
for key, strat, name in jobs:
    t0 = time.time()
    scores = []
    for tr_idx, te_idx in splits:
        X_tr, X_te = X_arr.iloc[tr_idx], X_arr.iloc[te_idx]
        y_tr, y_te = y_0[tr_idx], y_0[te_idx]
        pipe = build_pipeline(key, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy=strat)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        scores.append(f1_score(y_te, y_pred, average="macro"))
    dt = time.time() - t0
    arr = np.array(scores)
    per_fold[name] = arr
    summary.append({"config": name, "mean": arr.mean(), "std": arr.std(), "seconds": dt})
    log(f"{name:26s} macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}   ({dt:.1f}s)")
    pd.DataFrame(per_fold).to_csv(OUT / "verify_full_grid2_perfold.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "verify_full_grid2_summary.csv", index=False)

log("\nDone.")
LOG.close()
