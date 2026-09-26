"""
Independent verification: is CatBoost--none really the best model, and is
"not weighting" really the reason -- or is it a confound with CatBoost's
native categorical handling vs. everyone else's label-encoding (via
CreditPreprocessor), now that ORGNBR (134 categories) and PARENTORGNBR
(31 categories) are treated as categorical?

2x2 design for LightGBM: {label-encoded, native-categorical} x {balanced, none}
compared against CatBoost's native-categorical {balanced, none}.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare
from src.models import build_pipeline

OUT = Path(__file__).parent / "verify_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "verify_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1

log(f"CATEGORICAL_COLS ({len(CATEGORICAL_COLS)}): {CATEGORICAL_COLS}")
log(f"NUMERICAL_COLS ({len(NUMERICAL_COLS)}): {NUMERICAL_COLS}")
for c in CATEGORICAL_COLS:
    log(f"  {c}: {X_all[c].nunique()} unique values")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
)
log(f"\nTrain {len(X_train)}  Test {len(X_test)}")

results = []


def eval_pipe(name, pipe, Xtr, ytr, Xte, yte, sample_weight=None):
    if sample_weight is not None:
        pipe.fit(Xtr, ytr, classifier__sample_weight=sample_weight)
    else:
        pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    acc = (y_pred == yte).mean()
    f1m = f1_score(yte, y_pred, average="macro")
    f1w = f1_score(yte, y_pred, average="weighted")
    results.append({"config": name, "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w})
    log(f"  {name:45s} acc={acc:.4f}  macroF1={f1m:.4f}  weightedF1={f1w:.4f}")
    return pipe, y_pred


# ---------------------------------------------------------------------------
# 1) CatBoost, native categorical, none vs balanced (reproduce paper claim)
# ---------------------------------------------------------------------------
log("\n=== 1) CatBoost (native categorical handling) ===")
pipe_cb_none = build_pipeline("catboost", CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy="none")
pipe_cb_none, pred_cb_none = eval_pipe("CatBoost native-cat + none", pipe_cb_none, X_train, y_train, X_test, y_test)

pipe_cb_bal = build_pipeline("catboost", CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy="class_weight")
pipe_cb_bal, pred_cb_bal = eval_pipe("CatBoost native-cat + balanced", pipe_cb_bal, X_train, y_train, X_test, y_test)

# ---------------------------------------------------------------------------
# 2) LightGBM, label-encoded (current pipeline), balanced vs none
# ---------------------------------------------------------------------------
log("\n=== 2) LightGBM (label-encoded via CreditPreprocessor, current pipeline) ===")
pipe_lgb_bal = build_pipeline("lightgbm", CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy="class_weight")
pipe_lgb_bal, pred_lgb_bal = eval_pipe("LightGBM label-enc + balanced", pipe_lgb_bal, X_train, y_train, X_test, y_test)

pipe_lgb_none = build_pipeline("lightgbm", CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, imbalance_strategy="none")
pipe_lgb_none, pred_lgb_none = eval_pipe("LightGBM label-enc + none", pipe_lgb_none, X_train, y_train, X_test, y_test)

# ---------------------------------------------------------------------------
# 3) LightGBM, NATIVE categorical (bypass label-encoding), balanced vs none
# ---------------------------------------------------------------------------
log("\n=== 3) LightGBM (native pandas-category handling, no label-encoding) ===")
from lightgbm import LGBMClassifier


def make_native_cat_frame(X):
    Xc = X.copy()
    for c in CATEGORICAL_COLS:
        Xc[c] = Xc[c].astype(str).astype("category")
    return Xc


def fit_eval_lgb_native(class_weight, name):
    Xtr_n = make_native_cat_frame(X_train)
    Xte_n = make_native_cat_frame(X_test)
    # align category dtype codes between train/test (use train's categories)
    for c in CATEGORICAL_COLS:
        cats = Xtr_n[c].cat.categories
        Xte_n[c] = pd.Categorical(Xte_n[c], categories=cats)

    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xtr_num = pd.DataFrame(scaler.fit_transform(num_imputer.fit_transform(Xtr_n[NUMERICAL_COLS])),
                            columns=NUMERICAL_COLS, index=Xtr_n.index)
    Xte_num = pd.DataFrame(scaler.transform(num_imputer.transform(Xte_n[NUMERICAL_COLS])),
                            columns=NUMERICAL_COLS, index=Xte_n.index)
    Xtr_final = pd.concat([Xtr_num, Xtr_n[CATEGORICAL_COLS]], axis=1)
    Xte_final = pd.concat([Xte_num, Xte_n[CATEGORICAL_COLS]], axis=1)

    clf = LGBMClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1, class_weight=class_weight,
    )
    clf.fit(Xtr_final, y_train, categorical_feature=CATEGORICAL_COLS)
    y_pred = clf.predict(Xte_final)
    acc = (y_pred == y_test).mean()
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")
    results.append({"config": name, "accuracy": acc, "macro_f1": f1m, "weighted_f1": f1w})
    log(f"  {name:45s} acc={acc:.4f}  macroF1={f1m:.4f}  weightedF1={f1w:.4f}")
    return clf, y_pred


_, pred_lgb_nat_bal = fit_eval_lgb_native("balanced", "LightGBM native-cat + balanced")
_, pred_lgb_nat_none = fit_eval_lgb_native(None, "LightGBM native-cat + none")

# ---------------------------------------------------------------------------
# Summary table + confusion matrices for the two headline configs
# ---------------------------------------------------------------------------
log("\n=== SUMMARY ===")
df_res = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
log(df_res.to_string(index=False))
df_res.to_csv(OUT / "verify_summary.csv", index=False)

log("\n=== Confusion matrix: CatBoost native-cat + none ===")
cm1 = confusion_matrix(y_test, pred_cb_none)
log(str(cm1))
log(classification_report(y_test, pred_cb_none, digits=3))

log("\n=== Confusion matrix: LightGBM label-enc + balanced (paper's comparator) ===")
cm2 = confusion_matrix(y_test, pred_lgb_bal)
log(str(cm2))
log(classification_report(y_test, pred_lgb_bal, digits=3))

log("\nDone.")
LOG.close()
