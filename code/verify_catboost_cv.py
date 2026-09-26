"""
Repeated-CV check of the single-split finding: does LightGBM with NO
imbalance handling ("none") actually beat both CatBoost--none and every
balanced/weighted configuration, contradicting the paper's claim that
CatBoost--none is the best tested configuration overall?

5-fold x 3-repeat = 15 folds (reduced repeats to keep runtime reasonable
given CatBoost's cost; still enough to check whether the single-split
ordering is a fluke).
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
from scipy import stats

from config.config import DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE
from src.preprocessing import prepare
from src.models import build_pipeline

OUT = Path(__file__).parent / "verify_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "verify_cv_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


N_SPLITS, N_REPEATS = 5, 3

df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1
X_arr = X_all.reset_index(drop=True)

configs = [
    ("lightgbm", "none", "LightGBM + none"),
    ("lightgbm", "class_weight", "LightGBM + balanced"),
    ("catboost", "none", "CatBoost + none"),
    ("catboost", "class_weight", "CatBoost + balanced"),
]

rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
splits = list(rcv.split(X_arr, y_0))
log(f"Total folds: {len(splits)} ({N_SPLITS}-fold x {N_REPEATS}-repeat)")

per_fold = {}
summary = []
for model_key, strategy, name in configs:
    t0 = time.time()
    scores = []
    for tr_idx, te_idx in splits:
        X_tr, X_te = X_arr.iloc[tr_idx], X_arr.iloc[te_idx]
        y_tr, y_te = y_0[tr_idx], y_0[te_idx]
        pipe = build_pipeline(model_key, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy=strategy)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        scores.append(f1_score(y_te, y_pred, average="macro"))
    dt = time.time() - t0
    arr = np.array(scores)
    per_fold[name] = arr
    summary.append({"config": name, "mean": arr.mean(), "std": arr.std(), "seconds": dt})
    log(f"{name:24s} macroF1 = {arr.mean():.4f} +/- {arr.std():.4f}   ({dt:.1f}s)")

pd.DataFrame(summary).to_csv(OUT / "verify_cv_summary.csv", index=False)
pd.DataFrame(per_fold).to_csv(OUT / "verify_cv_perfold.csv", index=False)

# Corrected paired t-test: LightGBM+none vs each other config
log("\n=== Corrected resampled paired t-test (baseline = LightGBM + none) ===")
ratio = (1 / N_SPLITS) / (1 - 1 / N_SPLITS)
n = len(splits)
baseline = per_fold["LightGBM + none"]
for name, arr in per_fold.items():
    if name == "LightGBM + none":
        continue
    d = baseline - arr
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    corrected_var = var_d * (1.0 / n + ratio)
    t_stat = mean_d / np.sqrt(corrected_var) if corrected_var > 0 else np.nan
    df_ = n - 1
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_)) if not np.isnan(t_stat) else np.nan
    log(f"LightGBM+none vs {name:24s} mean_diff={mean_d:+.4f}  t={t_stat:.3f}  p={p_val:.4f}")

log("\nDone.")
LOG.close()
