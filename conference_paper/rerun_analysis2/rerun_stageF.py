"""
Stage F -- SEBL2026 revision, round 2 (class-weighting methodology + new
diagnostics). Single-split (fast) experiments:

1. Table 3 refresh: 6-model comparison using class_weight (not SMOTE).
2. Confusion matrix + per-class Precision/Recall/F1 for LightGBM and XGBoost.
3. CatBoost ablation re-check under the new base strategy.
4. DAYS_TO_MATURITY ablation (XGBoost + LightGBM, 11 -> 10 features).
5. Reference-date (d0) sensitivity check (LightGBM, 11-feature config).
6. Top-k curve (XGBoost + LightGBM), 13-variable pool, class_weight.
7. Gain group importance (XGBoost + LightGBM), class_weight.
8. SHAP (XGBoost + LightGBM), 11-feature main config, class_weight.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import prepare, parse_dates, engineer_features, clean, dataset1_feature_to_group
from src.models import build_pipeline, compute_sample_weights
from src.evaluation import compute_metrics
from src.feature_selection import importance_from_pipeline, group_importance_table, evaluate_performance_vs_k
from src.explainability import compute_shap_values, _mean_abs_importance

OUT = Path(__file__).parent / "conference_paper_out2"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "stageF_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


def fit_predict(model_key, cat_cols, num_cols, X_train, y_train, X_test, y_test, strategy="class_weight"):
    pipe = build_pipeline(model_key, cat_cols, num_cols, RANDOM_STATE, imbalance_strategy=strategy)
    use_sw = model_key == "xgboost" and strategy == "class_weight"
    if use_sw:
        pipe.fit(X_train, y_train, classifier__sample_weight=compute_sample_weights(y_train))
    else:
        pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test) + 1
    y_proba = pipe.predict_proba(X_test)
    y_true = y_test + 1
    acc = (y_pred == y_true).mean()
    m = compute_metrics(y_true, y_pred, y_proba)
    return pipe, y_true, y_pred, acc, m


# ---------------------------------------------------------------------------
# Load data (main 11-feature config)
# ---------------------------------------------------------------------------
df_raw = pd.read_excel(DATA_RAW)
X_all, y_all = prepare(df_raw, DROP_COLS, TARGET_COL)
y_0 = y_all - 1
cat11 = [c for c in CATEGORICAL_COLS if c in X_all.columns]
num11 = [c for c in NUMERICAL_COLS if c in X_all.columns]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
)
log(f"11-feature config: {len(cat11)} cat + {len(num11)} num = {len(cat11)+len(num11)}")
log(f"Train {len(X_train)}  Test {len(X_test)}")

# ---------------------------------------------------------------------------
# 1) Table 3 refresh -- class_weight strategy, 6 models
# ---------------------------------------------------------------------------
log("\n=== F.1: Table 3 refresh (class_weight) ===")
model_keys = ["logistic", "decision_tree", "random_forest", "xgboost", "lightgbm", "catboost"]
rows = []
pipes = {}
for key in model_keys:
    pipe, y_true, y_pred, acc, m = fit_predict(key, cat11, num11, X_train, y_train, X_test, y_test, "class_weight")
    pipes[key] = pipe
    rows.append({"model": key, "accuracy": acc, "f1_macro": m["f1_macro"],
                 "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]})
    log(f"  {key:16s} acc={acc:.4f}  macroF1={m['f1_macro']:.4f}  "
        f"weightedF1={m['f1_weighted']:.4f}  rocauc={m['roc_auc']:.4f}")
pd.DataFrame(rows).to_csv(OUT / "table3_class_weight.csv", index=False)

# ---------------------------------------------------------------------------
# 2) Confusion matrix + per-class metrics -- LightGBM and XGBoost
# ---------------------------------------------------------------------------
log("\n=== F.2: Confusion matrix + per-class metrics ===")
for key in ["lightgbm", "xgboost"]:
    pipe = pipes[key]
    y_pred = pipe.predict(X_test) + 1
    y_true_arr = y_test + 1
    cm = confusion_matrix(y_true_arr, y_pred, labels=[1, 2, 3, 4, 5])
    log(f"\n{key} confusion matrix (rows=true, cols=pred), labels 1..5:")
    log(str(cm))
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                 columns=[f"pred_{i}" for i in range(1, 6)]).to_csv(OUT / f"confusion_{key}.csv")
    report = classification_report(y_true_arr, y_pred, labels=[1, 2, 3, 4, 5], output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).T
    df_report.to_csv(OUT / f"perclass_{key}.csv")
    log(f"\n{key} per-class report:")
    log(df_report.to_string())

# ---------------------------------------------------------------------------
# 3) CatBoost ablation re-check (auto_class_weights None vs Balanced)
# ---------------------------------------------------------------------------
log("\n=== F.3: CatBoost ablation (re-check under class_weight base) ===")
cb_rows = []
for strat in ["class_weight", "none"]:
    _, y_true_cb, y_pred_cb, acc_cb, m_cb = fit_predict("catboost", cat11, num11, X_train, y_train, X_test, y_test, strat)
    auto_cw = "None" if strat == "none" else "Balanced"
    cb_rows.append({"strategy": strat, "auto_class_weights": auto_cw, "accuracy": acc_cb,
                     "f1_macro": m_cb["f1_macro"], "f1_weighted": m_cb["f1_weighted"], "roc_auc": m_cb["roc_auc"]})
    log(f"  strategy={strat:14s} auto_cw={auto_cw:9s} acc={acc_cb:.4f} macroF1={m_cb['f1_macro']:.4f} "
        f"weightedF1={m_cb['f1_weighted']:.4f} rocauc={m_cb['roc_auc']:.4f}")
pd.DataFrame(cb_rows).to_csv(OUT / "catboost_ablation_cw.csv", index=False)

# ---------------------------------------------------------------------------
# 4) DAYS_TO_MATURITY ablation -- XGBoost & LightGBM, 11 -> 10 features
# ---------------------------------------------------------------------------
log("\n=== F.4: DAYS_TO_MATURITY ablation (remove from 11-feature config) ===")
num10 = [c for c in num11 if c != "DAYS_TO_MATURITY"]
abl_rows = []
for key in ["xgboost", "lightgbm"]:
    _, _, _, acc10, m10 = fit_predict(key, cat11, num10, X_train, y_train, X_test, y_test, "class_weight")
    full_row = next(r for r in rows if r["model"] == key)
    abl_rows.append({"model": key, "macroF1_with_dtm": full_row["f1_macro"], "macroF1_without_dtm": m10["f1_macro"],
                      "delta": full_row["f1_macro"] - m10["f1_macro"], "rocauc_without_dtm": m10["roc_auc"]})
    log(f"  {key:10s} with={full_row['f1_macro']:.4f}  without={m10['f1_macro']:.4f}  "
        f"delta={full_row['f1_macro']-m10['f1_macro']:.4f}  rocauc_without={m10['roc_auc']:.4f}")
pd.DataFrame(abl_rows).to_csv(OUT / "dtm_ablation.csv", index=False)

# ---------------------------------------------------------------------------
# 5) Reference-date sensitivity -- LightGBM, 11-feature config
# ---------------------------------------------------------------------------
log("\n=== F.5: Reference-date (d0) sensitivity ===")


def prepare_with_d0(df_raw_, d0):
    df = parse_dates(df_raw_)
    df = df.copy()
    ref = pd.Timestamp(d0)
    df["LOAN_TENURE_DAYS"] = (df["NGAYDENHAN"] - df["OPEN_DATE"]).dt.days.clip(lower=0)
    df["DAYS_TO_MATURITY"] = (df["NGAYDENHAN"] - ref).dt.days
    df["UTIL_RATE"] = np.where(df["BASE_BAL"] > 0, df["CURR_BAL"] / df["BASE_BAL"], 0.0)
    df["UTIL_RATE"] = df["UTIL_RATE"].clip(0, 10)
    y = df[TARGET_COL].values.copy()
    df2 = clean(df, drop_cols=DROP_COLS + [TARGET_COL])
    return df2, y


d0_candidates = [
    datetime(2021, 6, 30), datetime(2021, 12, 31), datetime(2022, 6, 30),
    datetime(2020, 12, 31), datetime(2023, 12, 31),
]
d0_rows = []
for d0 in d0_candidates:
    Xd, yd = prepare_with_d0(df_raw, d0)
    yd0 = yd - 1
    Xtr, Xte, ytr, yte = train_test_split(Xd, yd0, test_size=TEST_SIZE, stratify=yd0, random_state=RANDOM_STATE)
    _, _, _, acc_d, m_d = fit_predict("lightgbm", cat11, num11, Xtr, ytr, Xte, yte, "class_weight")
    d0_rows.append({"d0": d0.strftime("%Y-%m-%d"), "accuracy": acc_d, "f1_macro": m_d["f1_macro"],
                     "f1_weighted": m_d["f1_weighted"], "roc_auc": m_d["roc_auc"]})
    log(f"  d0={d0.strftime('%Y-%m-%d')}  macroF1={m_d['f1_macro']:.4f}  rocauc={m_d['roc_auc']:.4f}")
pd.DataFrame(d0_rows).to_csv(OUT / "d0_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 6) Top-k curve (XGBoost + LightGBM), 13-variable pool, class_weight
# ---------------------------------------------------------------------------
log("\n=== F.6: Top-k curve (class_weight, 13-variable pool) ===")
full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X_full, y_full = prepare(df_raw, full_drop_cols, TARGET_COL)
y0_full = y_full - 1
cat_full = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_full.columns]
num_full = [c for c in NUMERICAL_COLS if c in X_full.columns]
Xtr1, Xte1, ytr1, yte1 = train_test_split(X_full, y0_full, test_size=TEST_SIZE, stratify=y0_full, random_state=RANDOM_STATE)

ks = [3, 5, 7, 9, 11, 13]
for key in ["xgboost", "lightgbm"]:
    pipe_full = build_pipeline(key, cat_full, num_full, RANDOM_STATE, imbalance_strategy="class_weight")
    if key == "xgboost":
        pipe_full.fit(Xtr1, ytr1, classifier__sample_weight=compute_sample_weights(ytr1))
    else:
        pipe_full.fit(Xtr1, ytr1)
    imp = importance_from_pipeline(pipe_full)
    ordered = imp.index.tolist()
    log(f"\n{key} importance ranking (13-pool, class_weight): {ordered}")

    def _build_eval(subset, key=key):
        cs = [c for c in cat_full if c in subset]
        ns = [c for c in num_full if c in subset]
        p = build_pipeline(key, cs, ns, RANDOM_STATE, imbalance_strategy="class_weight")
        if key == "xgboost":
            p.fit(Xtr1[subset], ytr1, classifier__sample_weight=compute_sample_weights(ytr1))
        else:
            p.fit(Xtr1[subset], ytr1)
        yp = p.predict(Xte1[subset]) + 1
        ypr = p.predict_proba(Xte1[subset])
        yt = yte1 + 1
        acc_ = (yp == yt).mean()
        mm = compute_metrics(yt, yp, ypr)
        return {"accuracy": acc_, "f1_macro": mm["f1_macro"], "f1_weighted": mm["f1_weighted"], "roc_auc": mm["roc_auc"]}

    df_k = evaluate_performance_vs_k(_build_eval, ordered, ks)
    df_k.to_csv(OUT / f"topk_{key}_cw.csv", index=False)
    log(f"{key} top-k curve (class_weight):")
    log(df_k.to_string(index=False))

    grp = group_importance_table(imp, dataset1_feature_to_group)
    grp.to_csv(OUT / f"group_importance_{key}_cw.csv", index=False)
    log(f"{key} group importance (Gain, class_weight):")
    log(grp.to_string(index=False))

# ---------------------------------------------------------------------------
# 8) SHAP (XGBoost + LightGBM), 11-feature main config, class_weight
# ---------------------------------------------------------------------------
log("\n=== F.8: SHAP (class_weight, 11-feature main config) ===")
for key in ["xgboost", "lightgbm"]:
    pipe = pipes[key]
    shap_values, feature_names, _ = compute_shap_values(pipe, X_test, key, max_samples=10000)
    mean_abs = _mean_abs_importance(shap_values, class_idx=None)
    s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    s.rename("mean_abs_shap").reset_index().rename(columns={"index": "feature"}).to_csv(
        OUT / f"shap_{key}_cw.csv", index=False)
    log(f"\n{key} SHAP (class_weight):")
    log(s.to_string())

    df_imp = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    df_imp["group"] = df_imp["feature"].map(dataset1_feature_to_group)
    grp = (df_imp.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index())
    grp["pct"] = (grp["importance"] / grp["importance"].sum() * 100).round(1)
    grp.to_csv(OUT / f"shap_group_{key}_cw.csv", index=False)
    log(f"{key} SHAP group importance (class_weight):")
    log(grp.to_string(index=False))

log("\nStage F complete.")
LOG.close()
