"""
Stage G -- repeated stratified CV (5-fold x 5-repeat = 25 partitions) under
class_weight methodology, for all 6 models. Saves PER-FOLD Macro-F1 (not just
mean/std) so a proper paired statistical test (corrected resampled t-test,
Nadeau & Bengio 2003) can be run in Stage H.

Also runs an embedded (in-fold) feature-selection variant: for each of the 25
folds, rank the 13-variable pool using ONLY that fold's training data
(LightGBM importance), keep a pre-specified k=9, refit and evaluate on that
fold's test partition -- this avoids the test-set-informed feature selection
present in a single train/test top-k curve.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import time
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score, classification_report

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE,
)
from src.preprocessing import prepare
from src.models import build_pipeline, compute_sample_weights
from src.feature_selection import importance_from_pipeline

OUT = Path(__file__).parent / "conference_paper_out2"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageG_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


N_SPLITS, N_REPEATS = 5, 5

df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1
cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]

X_arr = X_all.reset_index(drop=True)
classes = sorted(np.unique(y_0))

model_specs = [
    ("logistic", "logistic", "class_weight"),
    ("decision_tree", "decision_tree", "class_weight"),
    ("random_forest", "random_forest", "class_weight"),
    ("xgboost", "xgboost", "class_weight"),
    ("lightgbm", "lightgbm", "class_weight"),
    ("catboost", "catboost", "class_weight"),
    ("catboost_none", "catboost", "none"),
]
rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
splits = list(rcv.split(X_arr, y_0))
log(f"Total folds: {len(splits)} ({N_SPLITS}-fold x {N_REPEATS}-repeat)")

per_fold_scores = {label: [] for label, _, _ in model_specs}
summary_rows = []

for label, key, strategy in model_specs:
    t0 = time.time()
    for fi, (tr_idx, te_idx) in enumerate(splits):
        X_tr, X_te = X_arr.iloc[tr_idx], X_arr.iloc[te_idx]
        y_tr, y_te = y_0[tr_idx], y_0[te_idx]
        pipe = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy=strategy)
        if key == "xgboost" and strategy == "class_weight":
            pipe.fit(X_tr, y_tr, classifier__sample_weight=compute_sample_weights(y_tr))
        else:
            pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        f1m = f1_score(y_te, y_pred, average="macro")
        per_fold_scores[label].append(f1m)
    dt = time.time() - t0
    arr = np.array(per_fold_scores[label])
    summary_rows.append({"model": label, "macro_f1_mean": arr.mean(), "macro_f1_std": arr.std(),
                          "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seconds": dt})
    log(f"{label:16s} macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}   ({dt:.1f}s)")

pd.DataFrame(summary_rows).to_csv(OUT / "repeatedcv_summary_cw.csv", index=False)
pd.DataFrame(per_fold_scores).to_csv(OUT / "repeatedcv_perfold_cw.csv", index=False)
log("\nSaved repeatedcv_summary_cw.csv and repeatedcv_perfold_cw.csv")

# ---------------------------------------------------------------------------
# Embedded (in-fold) feature selection: LightGBM, 13-variable pool, k=9
# fixed a priori, ranking computed on TRAIN fold only.
# ---------------------------------------------------------------------------
log("\n=== Embedded in-fold feature selection (LightGBM, k=9, 13-variable pool) ===")
from src.preprocessing import DATASET1_FEATURE_GROUPS  # noqa
full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_full, y_full = prepare(df_raw, full_drop_cols, TARGET_COL)
y0_full = y_full - 1
cat_full = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_full.columns]
num_full = [c for c in NUMERICAL_COLS if c in X_full.columns]
X_full_arr = X_full.reset_index(drop=True)

K_FIXED = 9
embedded_scores = []
t0 = time.time()
rcv2 = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
for fi, (tr_idx, te_idx) in enumerate(rcv2.split(X_full_arr, y0_full)):
    X_tr, X_te = X_full_arr.iloc[tr_idx], X_full_arr.iloc[te_idx]
    y_tr, y_te = y0_full[tr_idx], y0_full[te_idx]

    # Rank using TRAIN fold only
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
    embedded_scores.append(f1m)

dt = time.time() - t0
arr = np.array(embedded_scores)
log(f"Embedded top-{K_FIXED} (in-fold selection): macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}  ({dt:.1f}s)")
pd.DataFrame({"fold": range(len(embedded_scores)), "macro_f1": embedded_scores}).to_csv(
    OUT / "embedded_topk_lightgbm_cw.csv", index=False)

log("\nStage G complete.")
LOG.close()
