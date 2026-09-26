"""
Stage D — re-run the ORIGINAL XGBoost top-k curve (Table 4) and XGBoost Gain
group importance (Table 5) in the current software environment, for full
numerical consistency with the refreshed Table 3 (reviewer feedback point 2/4
made the environment refresh necessary; this stage just extends that refresh
to the XGBoost-anchored tables that were otherwise going to be left stale).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare, dataset1_feature_to_group
from src.models import build_pipeline
from src.evaluation import compute_metrics
from src.feature_selection import importance_from_pipeline, group_importance_table, evaluate_performance_vs_k

OUT = Path(__file__).parent / "conference_paper_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageD_log.txt", "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


df_raw = pd.read_excel(DATA_RAW)
full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_all_full, y_all_full = prepare(df_raw, full_drop_cols, TARGET_COL)
y_0_full = y_all_full - 1

cat_full = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_all_full.columns]
num_full = [c for c in NUMERICAL_COLS if c in X_all_full.columns]

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_all_full, y_0_full, test_size=TEST_SIZE, stratify=y_0_full, random_state=RANDOM_STATE
)

pipe_xgb_full = build_pipeline("xgboost", cat_full, num_full, RANDOM_STATE, imbalance_strategy="smote_moderate")
pipe_xgb_full.fit(X_train1, y_train1)
importance_xgb = importance_from_pipeline(pipe_xgb_full)
ordered_xgb = importance_xgb.index.tolist()
log("XGBoost feature importance ranking (13-variable pool):")
for i, (feat, imp) in enumerate(importance_xgb.items(), 1):
    log(f"  {i:2d}. {feat:20s} {imp:.4f}")


def _build_and_eval_xgb(subset):
    cat_sub = [c for c in cat_full if c in subset]
    num_sub = [c for c in num_full if c in subset]
    p = build_pipeline("xgboost", cat_sub, num_sub, random_state=RANDOM_STATE,
                        imbalance_strategy="smote_moderate")
    p.fit(X_train1[subset], y_train1)
    y_pred = p.predict(X_test1[subset]) + 1
    y_proba = p.predict_proba(X_test1[subset])
    y_true = y_test1 + 1
    acc = (y_pred == y_true).mean()
    m = compute_metrics(y_true, y_pred, y_proba)
    return {"accuracy": acc, "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}


ks1 = [3, 5, 7, 9, 11, 13]
df_k_xgb = evaluate_performance_vs_k(_build_and_eval_xgb, ordered_xgb, ks1)
df_k_xgb.to_csv(OUT / "xgboost_topk_curve.csv", index=False)
log("\nXGBoost top-k curve:")
log(df_k_xgb.to_string(index=False))

grp_xgb = group_importance_table(importance_xgb, dataset1_feature_to_group)
grp_xgb.to_csv(OUT / "xgboost_group_importance.csv", index=False)
log("\nXGBoost group importance (Gain, 13-variable pool):")
log(grp_xgb.to_string(index=False))

log("\nStage D complete.")
LOG.close()
