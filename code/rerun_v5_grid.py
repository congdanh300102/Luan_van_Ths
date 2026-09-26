"""
Re-run for SEBL2026 revision round 3 (paper version 5).

Reviewer requests that make this re-run necessary:
  * "Report hold-out results for all six models under BOTH treatments"
    -> the paper only ever reported class-weighted hold-out numbers for the six
       models (Table 3) plus a single unweighted CatBoost row (Table 5).
  * "correct both statements or add the missing comparisons" (Table 10)
    -> the published Table 10 was assembled from several separate runs made
       BEFORE the xgboost class_weight fix and the lightgbm importance_type fix
       in src/models.py, so its rows are not mutually comparable. This script
       produces every hold-out and every repeated-CV number from ONE pass with
       the current code, on the identical 25 folds, so the whole family of
       comparisons is paired.

Outputs (code/rerun_v5_out/):
  holdout_grid.csv           6 models x {class_weight, none}, 4 metrics
  confusion_<cfg>.csv        confusion matrix, best config + lightgbm_balanced
  perclass_<cfg>.csv         per-class precision/recall/F1
  repeatedcv_perfold.csv     25 x 12 Macro-F1 matrix (identical folds)
  repeatedcv_summary.csv     mean +/- sd per configuration
  paired_tests.csv           Nadeau-Bengio corrected t-test + Holm vs. best
  extra_contrasts.csv        specific pairwise contrasts the reviewer asked for
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare
from src.models import build_pipeline
from src.evaluation import compute_metrics

OUT = Path(__file__).parent / "rerun_v5_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "rerun_v5_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


MODEL_KEYS = ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
STRATEGIES = ["class_weight", "none"]
PRETTY = {
    "logistic": "Logistic Regression", "decision_tree": "Decision Tree",
    "random_forest": "Random Forest", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "catboost": "CatBoost",
}

df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y0 = y_all - 1
cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]
log(f"Config: {len(cat11)} categorical + {len(num11)} numeric = {len(cat11)+len(num11)} features")
log(f"n = {len(X_all)}   class counts = {np.bincount(y0).tolist()}")

# ---------------------------------------------------------------------------
# 1) Hold-out grid: 6 models x 2 imbalance treatments
# ---------------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X_all, y0, test_size=TEST_SIZE, stratify=y0, random_state=RANDOM_STATE
)
log(f"\n=== 1) Hold-out grid (train {len(X_tr)} / test {len(X_te)}) ===")

holdout_rows = []
fitted = {}
for key in MODEL_KEYS:
    for strat in STRATEGIES:
        t0 = time.time()
        pipe = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te) + 1
        y_proba = pipe.predict_proba(X_te)
        y_true = y_te + 1
        m = compute_metrics(y_true, y_pred, y_proba)
        acc = float((y_pred == y_true).mean())
        cfg = f"{key}_{'balanced' if strat == 'class_weight' else 'none'}"
        fitted[cfg] = (pipe, y_true, y_pred)
        holdout_rows.append({
            "config": cfg, "model": PRETTY[key],
            "treatment": "class weighting" if strat == "class_weight" else "no weighting",
            "accuracy": acc, "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"],
            "seconds": time.time() - t0,
        })
        log(f"  {cfg:26s} acc={acc:.4f}  macroF1={m['f1_macro']:.4f}  "
            f"wF1={m['f1_weighted']:.4f}  auc={m['roc_auc']:.4f}  ({time.time()-t0:.0f}s)")

df_hold = pd.DataFrame(holdout_rows)
df_hold.to_csv(OUT / "holdout_grid.csv", index=False)
best_holdout = df_hold.loc[df_hold["f1_macro"].idxmax(), "config"]
log(f"\nBest hold-out Macro-F1 config: {best_holdout}")

# ---------------------------------------------------------------------------
# 2) Confusion matrix + per-class metrics for the configs the paper discusses
# ---------------------------------------------------------------------------
log("\n=== 2) Confusion matrices / per-class metrics ===")
for cfg in sorted({best_holdout, "lightgbm_balanced", "lightgbm_none", "xgboost_balanced"}):
    pipe, y_true, y_pred = fitted[cfg]
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                 columns=[f"pred_{i}" for i in range(1, 6)]).to_csv(OUT / f"confusion_{cfg}.csv")
    rep = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5],
                                output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUT / f"perclass_{cfg}.csv")
    log(f"\n{cfg} confusion matrix (rows=true 1..5, cols=pred 1..5):")
    log(str(cm))
    log(pd.DataFrame(rep).T.round(4).to_string())

# ---------------------------------------------------------------------------
# 3) Repeated stratified CV, 5-fold x 5-repeat, identical folds for all configs
# ---------------------------------------------------------------------------
N_SPLITS, N_REPEATS = 5, 5
rcv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
X_arr = X_all.reset_index(drop=True)
splits = list(rcv.split(X_arr, y0))
log(f"\n=== 3) Repeated CV: {len(splits)} folds ({N_SPLITS}-fold x {N_REPEATS}-repeat) ===")

per_fold, cv_rows = {}, []
for key in MODEL_KEYS:
    for strat in STRATEGIES:
        cfg = f"{key}_{'balanced' if strat == 'class_weight' else 'none'}"
        t0 = time.time()
        scores = []
        for tr_idx, te_idx in splits:
            pipe = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
            pipe.fit(X_arr.iloc[tr_idx], y0[tr_idx])
            scores.append(f1_score(y0[te_idx], pipe.predict(X_arr.iloc[te_idx]), average="macro"))
        arr = np.array(scores)
        per_fold[cfg] = arr
        cv_rows.append({"config": cfg, "model": PRETTY[key],
                        "treatment": "class weighting" if strat == "class_weight" else "no weighting",
                        "macro_f1_mean": arr.mean(), "macro_f1_std": arr.std(ddof=1),
                        "seconds": time.time() - t0})
        log(f"  {cfg:26s} macroF1 = {arr.mean():.4f} +/- {arr.std(ddof=1):.4f}   ({time.time()-t0:.0f}s)")
        pd.DataFrame(per_fold).to_csv(OUT / "repeatedcv_perfold.csv", index=False)
        pd.DataFrame(cv_rows).to_csv(OUT / "repeatedcv_summary.csv", index=False)

combined = pd.DataFrame(per_fold)

# ---------------------------------------------------------------------------
# 4) Nadeau-Bengio corrected paired t-test, best vs. every other, Holm-adjusted
# ---------------------------------------------------------------------------
N = N_SPLITS * N_REPEATS
RATIO = (1 / N_SPLITS) / (1 - 1 / N_SPLITS)   # n_test / n_train = 0.25 for 5-fold
DF = N - 1


def corrected_t(a, b):
    """Nadeau & Bengio (2003) corrected resampled paired t-test, a minus b."""
    d = np.asarray(a) - np.asarray(b)
    mean_d = d.mean()
    var_c = max(d.var(ddof=1) * (1 / N + RATIO), 1e-15)
    t = mean_d / np.sqrt(var_c)
    p = 2 * stats.t.sf(abs(t), DF)
    return mean_d, t, p


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * pvals[idx])
        adj[idx] = min(run, 1.0)
    return adj


means = combined.mean().sort_values(ascending=False)
best = means.index[0]
log(f"\n=== 4) Corrected paired t-test vs. best CV config: {best} ({means.iloc[0]:.4f}) ===")

others = [c for c in means.index if c != best]
rows = [dict(zip(("comparator", "mean_diff", "t", "p_raw"), (c,) + corrected_t(combined[best], combined[c])))
        for c in others]
res = pd.DataFrame(rows)
res["p_holm"] = holm(res["p_raw"].values)
res["mean_comparator"] = [combined[c].mean() for c in res["comparator"]]
res["std_comparator"] = [combined[c].std(ddof=1) for c in res["comparator"]]
res = res.sort_values("mean_comparator", ascending=False)
res.to_csv(OUT / "paired_tests.csv", index=False)
for _, r in res.iterrows():
    log(f"  {best} - {r['comparator']:26s} diff={r['mean_diff']:+.4f}  t={r['t']:+6.2f}  "
        f"p_raw={r['p_raw']:.6f}  p_holm={r['p_holm']:.6f}")

# ---------------------------------------------------------------------------
# 5) Specific contrasts the reviewer asked for
# ---------------------------------------------------------------------------
log("\n=== 5) Reviewer-requested contrasts (uncorrected for multiplicity) ===")
contrasts = [
    ("lightgbm_none", "xgboost_none"),
    ("catboost_none", "lightgbm_balanced"),
    ("lightgbm_none", "lightgbm_balanced"),
    ("xgboost_none", "xgboost_balanced"),
    ("catboost_none", "catboost_balanced"),
    ("random_forest_none", "random_forest_balanced"),
    ("decision_tree_none", "decision_tree_balanced"),
    ("logistic_none", "logistic_balanced"),
]
crows = []
for a, b in contrasts:
    if a not in combined or b not in combined:
        continue
    md, t, p = corrected_t(combined[a], combined[b])
    crows.append({"config_a": a, "config_b": b, "mean_a": combined[a].mean(),
                  "mean_b": combined[b].mean(), "mean_diff": md, "t": t, "p_raw": p})
    log(f"  {a:24s} - {b:24s} diff={md:+.4f}  t={t:+6.2f}  p={p:.6f}")
cres = pd.DataFrame(crows)
cres["p_holm_within_family"] = holm(cres["p_raw"].values)
cres.to_csv(OUT / "extra_contrasts.csv", index=False)

log("\nDone.")
LOG.close()
