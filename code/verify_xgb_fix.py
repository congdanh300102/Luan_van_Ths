"""
Re-run xgboost_balanced only, at the same 25-fold (5x5) splits as every other
script in this batch, after fixing the class_weight no-op bug in
src/models.py (XGBClassifier now threads a computed sample_weight through
fit() when imbalance_strategy="class_weight"). Replaces the buggy
xgboost_balanced column in verify_full_grid2_perfold.csv.
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
LOG = open(OUT / "verify_xgb_fix_log.txt", "w", encoding="utf-8")


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

t0 = time.time()
scores = []
for tr_idx, te_idx in splits:
    X_tr, X_te = X_arr.iloc[tr_idx], X_arr.iloc[te_idx]
    y_tr, y_te = y_0[tr_idx], y_0[te_idx]
    pipe = build_pipeline("xgboost", CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy="class_weight")
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    scores.append(f1_score(y_te, y_pred, average="macro"))
dt = time.time() - t0
arr = np.array(scores)
log(f"xgboost_balanced (fixed)  macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}   ({dt:.1f}s)")

pd.Series(arr, name="xgboost_balanced").to_csv(OUT / "verify_xgb_fix_perfold.csv", index=False)
log("\nDone.")
LOG.close()
