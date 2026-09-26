"""
Stage A — re-run for reviewer feedback on the SEBL2026 paper.

1. Main 6-model comparison on the 11-feature configuration (adds Logistic
   Regression, testing whether the previously-reported sklearn API
   incompatibility still occurs in the current environment).
2. CatBoost ablation: auto_class_weights=None (imbalance_strategy="none")
   vs. the Table-3 setting (smote_moderate -> auto_class_weights="Balanced").
3. Top-k feature curve re-run with LightGBM (instead of XGBoost) on the
   13-variable pool, plus Gain-based group importance for LightGBM.
4. SHAP re-run with LightGBM on the 11-feature main configuration (the same
   model already reported as the LightGBM row of Table 3), plus SHAP-based
   group importance.

All splits/pipelines replicate exactly what pages/2_Huan_Luyen.py and
pages/7_Phan_Tich_Dac_Trung.py do for Dataset A (bo A).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare, DATASET1_FEATURE_GROUPS, dataset1_feature_to_group
from src.models import build_pipeline, available_models
from src.evaluation import compute_metrics
from src.feature_selection import (
    importance_from_pipeline, group_importance_table, evaluate_performance_vs_k,
)

OUT = Path(__file__).parent / "conference_paper_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageA_log.txt", "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


# ── Load data (Dataset A / bo A) ─────────────────────────────────────────────
df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)  # y_all is 1-based
y_0 = y_all - 1  # 0-based, matches pages/2_Huan_Luyen.py

cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]
log(f"11-feature main config: {len(cat11)} categorical + {len(num11)} numerical = {len(cat11)+len(num11)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
)
log(f"Train: {len(X_train)}  Test: {len(X_test)}")

# ══════════════════════════════════════════════════════════════════════════
# 1) Main 6-model comparison (11-feature config, smote_moderate) — incl. LR
# ══════════════════════════════════════════════════════════════════════════
log("\n=== STAGE A.1: Main model comparison (11 features, smote_moderate) ===")
model_keys = ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
rows = []
for key in model_keys:
    try:
        pipe = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy="smote_moderate")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test) + 1
        y_proba = pipe.predict_proba(X_test)
        y_true = y_test + 1
        acc = (y_pred == y_true).mean()
        m = compute_metrics(y_true, y_pred, y_proba)
        rows.append({"model": key, "accuracy": acc, "f1_macro": m["f1_macro"],
                     "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"], "status": "OK"})
        log(f"  {key:16s} OK   acc={acc:.4f}  macroF1={m['f1_macro']:.4f}  "
            f"weightedF1={m['f1_weighted']:.4f}  rocauc={m['roc_auc']:.4f}")
    except Exception as e:
        rows.append({"model": key, "accuracy": np.nan, "f1_macro": np.nan,
                     "f1_weighted": np.nan, "roc_auc": np.nan, "status": f"ERROR: {e}"})
        log(f"  {key:16s} ERROR: {e}")
        log(traceback.format_exc())

df_main = pd.DataFrame(rows)
df_main.to_csv(OUT / "table3_with_logreg.csv", index=False)
log("\nSaved table3_with_logreg.csv")

# ══════════════════════════════════════════════════════════════════════════
# 2) CatBoost ablation: auto_class_weights None vs Balanced
# ══════════════════════════════════════════════════════════════════════════
log("\n=== STAGE A.2: CatBoost ablation (imbalance_strategy) ===")
cb_rows = []
for strat in ["smote_moderate", "none"]:
    pipe = build_pipeline("catboost", cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test) + 1
    y_proba = pipe.predict_proba(X_test)
    y_true = y_test + 1
    acc = (y_pred == y_true).mean()
    m = compute_metrics(y_true, y_pred, y_proba)
    auto_cw = "None" if strat == "none" else "Balanced"
    cb_rows.append({"imbalance_strategy": strat, "auto_class_weights": auto_cw,
                     "accuracy": acc, "f1_macro": m["f1_macro"],
                     "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]})
    log(f"  strategy={strat:16s} auto_class_weights={auto_cw:9s} "
        f"acc={acc:.4f}  macroF1={m['f1_macro']:.4f}  weightedF1={m['f1_weighted']:.4f}  rocauc={m['roc_auc']:.4f}")

pd.DataFrame(cb_rows).to_csv(OUT / "catboost_ablation.csv", index=False)
log("Saved catboost_ablation.csv")

# ══════════════════════════════════════════════════════════════════════════
# 3) Top-k curve with LightGBM on 13-variable pool (matches page 7 recipe)
# ══════════════════════════════════════════════════════════════════════════
log("\n=== STAGE A.3: Top-k feature curve with LightGBM (13-variable pool) ===")
# BUGFIX: DROP_COLS already removes SEX/LOAIKH, so building the "13-variable
# pool" from X_all (prepared with DROP_COLS) silently collapses to the same
# 11 features as the main config. pages/7_Phan_Tich_Dac_Trung.py avoids this
# by re-preparing the raw data with full_drop_cols (DROP_COLS minus SEX/LOAIKH)
# — replicate that exactly here.
full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_all_full, y_all_full = prepare(df_raw, full_drop_cols, TARGET_COL)
y_0_full = y_all_full - 1

cat_full = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_all_full.columns]
num_full = [c for c in NUMERICAL_COLS if c in X_all_full.columns]
log(f"13-variable pool: {len(cat_full)} categorical + {len(num_full)} numerical = {len(cat_full)+len(num_full)}")

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_all_full, y_0_full, test_size=TEST_SIZE, stratify=y_0_full, random_state=RANDOM_STATE
)

pipe_lgb_full = build_pipeline("lightgbm", cat_full, num_full, RANDOM_STATE, imbalance_strategy="smote_moderate")
pipe_lgb_full.fit(X_train1, y_train1)
importance_lgb = importance_from_pipeline(pipe_lgb_full)
ordered_lgb = importance_lgb.index.tolist()
log("LightGBM feature importance ranking (13-variable pool):")
for i, (feat, imp) in enumerate(importance_lgb.items(), 1):
    log(f"  {i:2d}. {feat:20s} {imp:.2f}")

importance_lgb.rename("importance").reset_index().rename(columns={"index": "feature"}).to_csv(
    OUT / "lightgbm_importance_13.csv", index=False)


def _build_and_eval_lgb(subset):
    cat_sub = [c for c in cat_full if c in subset]
    num_sub = [c for c in num_full if c in subset]
    p = build_pipeline("lightgbm", cat_sub, num_sub, random_state=RANDOM_STATE,
                        imbalance_strategy="smote_moderate")
    p.fit(X_train1[subset], y_train1)
    y_pred = p.predict(X_test1[subset]) + 1
    y_proba = p.predict_proba(X_test1[subset])
    y_true = y_test1 + 1
    acc = (y_pred == y_true).mean()
    m = compute_metrics(y_true, y_pred, y_proba)
    return {"accuracy": acc, "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}


ks1 = [3, 5, 7, 9, 11, 13]
df_k_lgb = evaluate_performance_vs_k(_build_and_eval_lgb, ordered_lgb, ks1)
df_k_lgb.to_csv(OUT / "lightgbm_topk_curve.csv", index=False)
log("\nLightGBM top-k curve:")
log(df_k_lgb.to_string(index=False))

# Group importance for LightGBM (13-variable pool)
grp_lgb = group_importance_table(importance_lgb, dataset1_feature_to_group)
grp_lgb.to_csv(OUT / "lightgbm_group_importance.csv", index=False)
log("\nLightGBM group importance (Gain, 13-variable pool):")
log(grp_lgb.to_string(index=False))

log("\nStage A complete.")
LOG.close()
