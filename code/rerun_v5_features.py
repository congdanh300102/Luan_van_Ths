"""
Feature-level analyses (RQ2/RQ3) recomputed under BOTH imbalance treatments,
so that the paper's feature conclusions are reported for the configuration the
model comparison actually selects (unweighted), not only for the class-weighted
configuration Table 10 shows to be inferior.

Produces, for XGBoost and LightGBM, under {class_weight, none}:
  topk_<model>_<treatment>.csv            Macro-F1 / ROC-AUC vs. number of features
  gain_group_<model>_<treatment>.csv      Gain-based business-group importance
  shap_group_<model>_<treatment>.csv      SHAP-based business-group importance
  shap_feature_<model>_<treatment>.csv    SHAP mean |phi| per feature
  dtm_ablation.csv                        DAYS_TO_MATURITY removal, both treatments
  d0_sensitivity.csv                      reference-date invariance, both treatments
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import (
    prepare, parse_dates, clean, dataset1_feature_to_group,
)
from src.models import build_pipeline
from src.evaluation import compute_metrics
from src.feature_selection import (
    importance_from_pipeline, group_importance_table, evaluate_performance_vs_k,
)
from src.explainability import compute_shap_values, _mean_abs_importance

OUT = Path(__file__).parent / "rerun_v5_out"
OUT.mkdir(exist_ok=True)
LOG = open(OUT / "features_log.txt", "w", encoding="utf-8")


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


TREATMENTS = [("class_weight", "balanced"), ("none", "none")]
MODELS = ["xgboost", "lightgbm"]

df_raw = pd.read_excel(DATA_RAW)

# 11-feature main configuration -----------------------------------------------
X11, y11 = prepare(df_raw, DROP_COLS, TARGET_COL)
y11_0 = y11 - 1
cat11 = [c for c in CATEGORICAL_COLS if c in X11.columns]
num11 = [c for c in NUMERICAL_COLS if c in X11.columns]
Xtr11, Xte11, ytr11, yte11 = train_test_split(
    X11, y11_0, test_size=TEST_SIZE, stratify=y11_0, random_state=RANDOM_STATE)

# 13-variable pool (adds SEX and LOAIKH) --------------------------------------
pool_drop = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
X13, y13 = prepare(df_raw, pool_drop, TARGET_COL)
y13_0 = y13 - 1
cat13 = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X13.columns]
num13 = [c for c in NUMERICAL_COLS if c in X13.columns]
Xtr13, Xte13, ytr13, yte13 = train_test_split(
    X13, y13_0, test_size=TEST_SIZE, stratify=y13_0, random_state=RANDOM_STATE)


def evaluate(pipe, X_te, y_te):
    y_pred = pipe.predict(X_te) + 1
    y_proba = pipe.predict_proba(X_te)
    y_true = y_te + 1
    m = compute_metrics(y_true, y_pred, y_proba)
    return {"accuracy": float((y_pred == y_true).mean()), "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}


# ---------------------------------------------------------------------------
# 1) DAYS_TO_MATURITY ablation, both treatments
# ---------------------------------------------------------------------------
log("=== 1) DAYS_TO_MATURITY ablation (11 -> 10 features) ===")
num10 = [c for c in num11 if c != "DAYS_TO_MATURITY"]
abl = []
for key in MODELS:
    for strat, tag in TREATMENTS:
        p_full = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
        p_full.fit(Xtr11, ytr11)
        m_full = evaluate(p_full, Xte11, yte11)
        p_abl = build_pipeline(key, cat11, num10, RANDOM_STATE, imbalance_strategy=strat)
        p_abl.fit(Xtr11, ytr11)
        m_abl = evaluate(p_abl, Xte11, yte11)
        delta = m_full["f1_macro"] - m_abl["f1_macro"]
        abl.append({"model": key, "treatment": tag,
                    "f1_11feat": m_full["f1_macro"], "f1_10feat": m_abl["f1_macro"],
                    "delta": delta, "pct_change": -delta / m_full["f1_macro"] * 100,
                    "auc_10feat": m_abl["roc_auc"]})
        log(f"  {key:9s} {tag:9s} 11f={m_full['f1_macro']:.4f}  10f={m_abl['f1_macro']:.4f}  "
            f"delta={-delta:+.4f} ({-delta/m_full['f1_macro']*100:+.1f}%)  auc10={m_abl['roc_auc']:.4f}")
pd.DataFrame(abl).to_csv(OUT / "dtm_ablation.csv", index=False)

# ---------------------------------------------------------------------------
# 2) Reference-date sensitivity, both treatments (LightGBM)
# ---------------------------------------------------------------------------
log("\n=== 2) Reference-date (d0) sensitivity, LightGBM ===")


def prepare_with_d0(raw, d0):
    df = parse_dates(raw).copy()
    ref = pd.Timestamp(d0)
    df["LOAN_TENURE_DAYS"] = (df["NGAYDENHAN"] - df["OPEN_DATE"]).dt.days.clip(lower=0)
    df["DAYS_TO_MATURITY"] = (df["NGAYDENHAN"] - ref).dt.days
    df["UTIL_RATE"] = np.where(df["BASE_BAL"] > 0, df["CURR_BAL"] / df["BASE_BAL"], 0.0).clip(0, 10)
    y = df[TARGET_COL].values.copy()
    return clean(df, drop_cols=DROP_COLS + [TARGET_COL]), y


d0_rows = []
for d0 in [datetime(2020, 12, 31), datetime(2021, 6, 30), datetime(2021, 12, 31),
           datetime(2022, 6, 30), datetime(2023, 12, 31)]:
    Xd, yd = prepare_with_d0(df_raw, d0)
    yd0 = yd - 1
    Xa, Xb, ya, yb = train_test_split(Xd, yd0, test_size=TEST_SIZE, stratify=yd0,
                                      random_state=RANDOM_STATE)
    for strat, tag in TREATMENTS:
        p = build_pipeline("lightgbm", cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
        p.fit(Xa, ya)
        m = evaluate(p, Xb, yb)
        d0_rows.append({"d0": d0.strftime("%Y-%m-%d"), "treatment": tag, **m})
        log(f"  d0={d0:%Y-%m-%d}  {tag:9s} macroF1={m['f1_macro']:.4f}  auc={m['roc_auc']:.4f}")
pd.DataFrame(d0_rows).to_csv(OUT / "d0_sensitivity.csv", index=False)

# ---------------------------------------------------------------------------
# 3) Top-k curve + Gain group importance, 13-variable pool, both treatments
# ---------------------------------------------------------------------------
log("\n=== 3) Top-k curve and Gain group importance (13-variable pool) ===")
KS = [3, 5, 7, 9, 11, 13]
for key in MODELS:
    for strat, tag in TREATMENTS:
        t0 = time.time()
        p_full = build_pipeline(key, cat13, num13, RANDOM_STATE, imbalance_strategy=strat)
        p_full.fit(Xtr13, ytr13)
        imp = importance_from_pipeline(p_full)
        ordered = imp.index.tolist()
        log(f"\n  {key} / {tag} Gain ranking: {ordered}")

        def _build_eval(subset, key=key, strat=strat):
            cs = [c for c in cat13 if c in subset]
            ns = [c for c in num13 if c in subset]
            p = build_pipeline(key, cs, ns, RANDOM_STATE, imbalance_strategy=strat)
            p.fit(Xtr13[subset], ytr13)
            return evaluate(p, Xte13[subset], yte13)

        df_k = evaluate_performance_vs_k(_build_eval, ordered, KS)
        df_k.to_csv(OUT / f"topk_{key}_{tag}.csv", index=False)
        log(df_k.to_string(index=False))

        grp = group_importance_table(imp, dataset1_feature_to_group)
        grp.to_csv(OUT / f"gain_group_{key}_{tag}.csv", index=False)
        log(grp.to_string(index=False))
        log(f"  ({time.time()-t0:.0f}s)")

# ---------------------------------------------------------------------------
# 4) SHAP, 11-feature main configuration, both treatments
# ---------------------------------------------------------------------------
log("\n=== 4) SHAP (11-feature main configuration) ===")
for key in MODELS:
    for strat, tag in TREATMENTS:
        p = build_pipeline(key, cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
        p.fit(Xtr11, ytr11)
        shap_values, feature_names, _ = compute_shap_values(p, Xte11, key, max_samples=10000)
        mean_abs = _mean_abs_importance(shap_values, class_idx=None)
        s = (pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
             .rename("mean_abs_shap"))
        s.reset_index().rename(columns={"index": "feature"}).to_csv(
            OUT / f"shap_feature_{key}_{tag}.csv", index=False)
        log(f"\n  {key} / {tag} SHAP mean |phi|:")
        log(s.round(4).to_string())

        d = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        d["group"] = d["feature"].map(dataset1_feature_to_group)
        grp = d.groupby("group")["importance"].sum().sort_values(ascending=False).reset_index()
        grp["pct"] = (grp["importance"] / grp["importance"].sum() * 100).round(1)
        grp.to_csv(OUT / f"shap_group_{key}_{tag}.csv", index=False)
        log(grp.to_string(index=False))

# ---------------------------------------------------------------------------
# 5) Embedded (in-fold) feature selection, k = 9, both treatments
#    The ranking is recomputed from each training fold alone, so the selected
#    subset can never have been informed by the fold's test partition.
# ---------------------------------------------------------------------------
log("\n=== 5) Embedded in-fold feature selection (LightGBM, k = 9, 13-variable pool) ===")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import f1_score

K_FIXED = 9
X13_arr = X13.reset_index(drop=True)
emb_rows = []
for strat, tag in TREATMENTS:
    t0 = time.time()
    fixed_scores, embedded_scores = [], []
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    for tr_idx, te_idx in rcv.split(X13_arr, y13_0):
        X_tr, X_te = X13_arr.iloc[tr_idx], X13_arr.iloc[te_idx]
        y_tr, y_te = y13_0[tr_idx], y13_0[te_idx]

        rank_pipe = build_pipeline("lightgbm", cat13, num13, RANDOM_STATE,
                                   imbalance_strategy=strat)
        rank_pipe.fit(X_tr, y_tr)
        top = importance_from_pipeline(rank_pipe).index.tolist()[:K_FIXED]

        cs = [c for c in cat13 if c in top]
        ns = [c for c in num13 if c in top]
        p = build_pipeline("lightgbm", cs, ns, RANDOM_STATE, imbalance_strategy=strat)
        p.fit(X_tr[top], y_tr)
        embedded_scores.append(f1_score(y_te, p.predict(X_te[top]), average="macro"))

        pf = build_pipeline("lightgbm", cat11, num11, RANDOM_STATE, imbalance_strategy=strat)
        pf.fit(X11.reset_index(drop=True).iloc[tr_idx], y11_0[tr_idx])
        fixed_scores.append(f1_score(y11_0[te_idx],
                                     pf.predict(X11.reset_index(drop=True).iloc[te_idx]),
                                     average="macro"))
    e, f = np.array(embedded_scores), np.array(fixed_scores)
    emb_rows.append({"treatment": tag,
                     "embedded_k9_mean": e.mean(), "embedded_k9_std": e.std(ddof=1),
                     "fixed_11_mean": f.mean(), "fixed_11_std": f.std(ddof=1)})
    log(f"  {tag:9s} embedded k=9: {e.mean():.4f} +/- {e.std(ddof=1):.4f}   "
        f"fixed 11: {f.mean():.4f} +/- {f.std(ddof=1):.4f}   ({time.time()-t0:.0f}s)")
pd.DataFrame(emb_rows).to_csv(OUT / "embedded_topk.csv", index=False)

log("\nDone.")
LOG.close()
