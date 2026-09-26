"""
Stage B — repeated stratified 5-fold x 10-repeat CV for the 6-model main
comparison (11-feature configuration, smote_moderate), to report Macro-F1 as
mean +/- std instead of a single train/test split point estimate (reviewer
feedback point 4). Uses src.evaluation.repeated_stratified_recall exactly as
already used elsewhere in the codebase (pages/2_Huan_Luyen.py).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE,
)
from src.preprocessing import prepare
from src.models import build_pipeline
from src.evaluation import repeated_stratified_recall

OUT = Path(__file__).parent / "conference_paper_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageB_log.txt", "w", encoding="utf-8")


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1

cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]

model_keys = ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
summary_rows = []

for key in model_keys:
    t0 = time.time()
    try:
        pipe = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy="smote_moderate")
        per_class_df, summary = repeated_stratified_recall(
            pipe, X_all, y_0, n_splits=5, n_repeats=10, random_state=RANDOM_STATE,
        )
        dt = time.time() - t0
        summary_rows.append({"model": key, "macro_f1_mean": summary["macro_f1_mean"],
                              "macro_f1_std": summary["macro_f1_std"],
                              "n_splits": summary["n_splits"], "n_repeats": summary["n_repeats"],
                              "seconds": dt, "status": "OK"})
        log(f"{key:16s} OK   macroF1 = {summary['macro_f1_mean']:.4f} +/- {summary['macro_f1_std']:.4f}"
            f"   ({dt:.1f}s)")
        per_class_df.to_csv(OUT / f"repeatedcv_perclass_{key}.csv", index=False)
    except Exception as e:
        dt = time.time() - t0
        summary_rows.append({"model": key, "macro_f1_mean": float("nan"),
                              "macro_f1_std": float("nan"), "n_splits": 5, "n_repeats": 10,
                              "seconds": dt, "status": f"ERROR: {e}"})
        log(f"{key:16s} ERROR: {e}  ({dt:.1f}s)")
        log(traceback.format_exc())

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT / "repeatedcv_summary.csv", index=False)
log("\nSaved repeatedcv_summary.csv")
log(df_summary.to_string(index=False))
LOG.close()
