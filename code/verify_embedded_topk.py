"""
Re-verify the embedded (in-fold) feature-selection claim in Section 4.2:
"LightGBM ranking was recomputed inside every training fold with k=9 fixed
... The embedded result is 0.5663 +/- 0.0092 versus 0.5691 +/- 0.0099 for the
fixed 11-feature configuration." Same methodology as
conference_paper/rerun_analysis2/rerun_stageG.py's embedded-selection block,
rerun fresh in the current (bug-fixed) environment for independent check.
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
from src.feature_selection import importance_from_pipeline

OUT = Path(__file__).parent / "verify_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "verify_embedded_topk_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


N_SPLITS, N_REPEATS = 5, 5
K_FIXED = 9

df_raw = pd.read_excel(DATA_RAW)
full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_full, y_full = prepare(df_raw, full_drop_cols, TARGET_COL)
y0_full = y_full - 1
cat_full = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_full.columns]
num_full = [c for c in NUMERICAL_COLS if c in X_full.columns]
X_full_arr = X_full.reset_index(drop=True)

rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
splits = list(rcv.split(X_full_arr, y0_full))
log(f"Total folds: {len(splits)} ({N_SPLITS}-fold x {N_REPEATS}-repeat)")

t0 = time.time()
scores = []
for fi, (tr_idx, te_idx) in enumerate(splits):
    X_tr, X_te = X_full_arr.iloc[tr_idx], X_full_arr.iloc[te_idx]
    y_tr, y_te = y0_full[tr_idx], y0_full[te_idx]

    rank_pipe = build_pipeline("lightgbm", cat_full, num_full, RANDOM_STATE, imbalance_strategy="class_weight")
    rank_pipe.fit(X_tr, y_tr)
    imp = importance_from_pipeline(rank_pipe)
    top_k_features = imp.index.tolist()[:K_FIXED]

    cs = [c for c in cat_full if c in top_k_features]
    ns = [c for c in num_full if c in top_k_features]
    p = build_pipeline("lightgbm", cs, ns, RANDOM_STATE, imbalance_strategy="class_weight")
    p.fit(X_tr[top_k_features], y_tr)
    y_pred = p.predict(X_te[top_k_features])
    f1m = f1_score(y_te, y_pred, average="macro")
    scores.append(f1m)
    if (fi + 1) % 5 == 0:
        log(f"  fold {fi+1}/{len(splits)} done ({time.time()-t0:.0f}s elapsed)")

dt = time.time() - t0
arr = np.array(scores)
log(f"\nEmbedded top-{K_FIXED} (in-fold selection): macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}  ({dt:.1f}s)")
pd.DataFrame({"fold": range(len(scores)), "macro_f1": scores}).to_csv(
    OUT / "verify_embedded_topk_perfold.csv", index=False)
log("\nDone.")
LOG.close()
