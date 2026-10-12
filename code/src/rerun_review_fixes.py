"""Chạy lại và kiểm chứng các mục trong bản rà soát luận văn (bản 20260920).

  M   Bộ B, bài toán 5 lớp — chạy lại toàn bộ bảng chính sau khi sửa lỗi điền
      khuyết ba biến cơ cấu nợ (mục C.1): 6 mô hình trên tập kiểm định nội bộ
      và tập ngoài thời gian, hiệu năng theo nhóm, ma trận nhầm lẫn, Top-k,
      Gain theo nhóm nghiệp vụ, SHAP (cỡ mẫu thống nhất 5.000 quan sát).
  S1  Kiểm chứng C.5 — mã chi nhánh xử lý như biến phân loại thay vì biến số.
  S2  Kiểm chứng C.7 — SMOTE-NC thay cho SMOTE trên dữ liệu hỗn hợp.
  S3  Kiểm chứng C.6 — One-Hot cho hai biến định danh nhiều danh mục của bộ A.
  I   Kiểm chứng A.4 — Information Value tính CHỈ trên tập huấn luyện thay vì
      toàn bộ mẫu, cho cả hai bộ.
  T   Xác nhận bài toán chuyển nhóm không đổi sau khi sửa C.1.

Chạy:  cd code && py -m src.rerun_review_fixes [M] [S1] [S2] [S3] [I] [T]
Output: code/results/rev2_*.csv
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config.config import (
    DATA_RAW, TARGET_COL, RANDOM_STATE, TEST_SIZE,
    DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY,
    TRANSITION_ID_COL, TRANSITION_SMOTE_STRATEGY,
    TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
)
from src.preprocessing import parse_dates, engineer_features
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, feature_to_group,
    TRANSITION_LABEL_COL, RESTRUCTURE_COLS,
)
from src.models import build_pipeline
from src.iv_analysis import compute_iv_table
from src.evaluation import repeated_stratified_binary_eval

RES = Path(__file__).parent.parent / "results"
MODELS = {"Logistic Regression": "logistic", "Decision Tree": "decision_tree",
          "Random Forest": "random_forest", "XGBoost": "xgboost",
          "LightGBM": "lightgbm", "CatBoost": "catboost"}
BRANCH_COL = "Mã chi nhánh TCTD"
SHAP_N = 5000

CAT_A = ["MJACCTTYPCD", "CURRMIACCTTYPCD", "MUCDICHVAY"]
NUM_A = ["BASE_BAL", "CURR_BAL", "LAISUAT", "ORGNBR", "PARENTORGNBR",
         "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY", "UTIL_RATE"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def ordinal_mae(cm):
    cm = np.asarray(cm, float)
    k = cm.shape[0]
    return float((cm * np.abs(np.arange(k)[:, None] - np.arange(k))).sum() / cm.sum())


def evaluate(pipe, XX, yy):
    pred = np.asarray(pipe.predict(XX)).ravel().astype(int)
    proba = pipe.predict_proba(XX)
    cm = confusion_matrix(yy, pred, labels=[0, 1, 2, 3, 4])
    pr, rc, f1, sup = precision_recall_fscore_support(
        yy, pred, labels=[0, 1, 2, 3, 4], zero_division=0)
    m = {"accuracy": (pred == yy).mean(),
         "f1_macro": f1_score(yy, pred, average="macro"),
         "f1_weighted": f1_score(yy, pred, average="weighted"),
         "roc_auc_macro": roc_auc_score(yy, proba, multi_class="ovr", average="macro"),
         "balanced_accuracy": rc.mean(), "ordinal_mae": ordinal_mae(cm)}
    return m, cm, (pr, rc, f1, sup)


def load_b(branch_as_categorical=False):
    df = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    num_raw, cat_raw = build_raw_feature_pool(df)
    de, eng = engineer_business_features(df)
    num_p, cat_p = num_raw + eng, list(cat_raw)
    if branch_as_categorical and BRANCH_COL in num_p:
        num_p = [c for c in num_p if c != BRANCH_COL]
        cat_p = cat_p + [BRANCH_COL]
        de[BRANCH_COL] = de[BRANCH_COL].astype(str)
    X, y = prepare_credit_info(de, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)

    dh, _ = engineer_business_features(load_credit_info(DATA_CREDIT_INFO_TEST))
    if branch_as_categorical:
        dh[BRANCH_COL] = dh[BRANCH_COL].astype(str)
    Xh, yh = prepare_credit_info(dh, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)
    return X, y, Xh, yh, num_p, cat_p


# ── M ────────────────────────────────────────────────────────────────────────
def part_m():
    X, y, Xh, yh, num_p, cat_p = load_b()
    log(f"M: kiểm tra điền 0 — trung vị ba biến cơ cấu nay bằng "
        f"{[float(X[c].median()) for c in RESTRUCTURE_COLS]}, "
        f"tỷ lệ thiếu {[round(X[c].isna().mean()*100, 2) for c in RESTRUCTURE_COLS]}")
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    rows, byclass, cms = [], [], []
    best = (None, -1)
    for name, key in MODELS.items():
        t0 = time.time()
        p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr, ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            m, cm, (pr, rc, f1, sup) = evaluate(p, XX, yy)
            rows.append({"model": name, "split": split, **m})
            for i in range(5):
                byclass.append({"model": name, "split": split, "nhom": i + 1,
                                "support": int(sup[i]), "precision": pr[i],
                                "recall": rc[i], "f1": f1[i]})
            cms.append(pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                                    columns=[f"pred_{i}" for i in range(1, 6)])
                       .assign(model=name, split=split))
        log(f"M  {name}: val={rows[-2]['f1_macro']:.4f} ngoài TG={rows[-1]['f1_macro']:.4f} "
            f"({time.time()-t0:.0f}s)")
        if rows[-2]["f1_macro"] > best[1]:
            best = (key, rows[-2]["f1_macro"], p, name)
    pd.DataFrame(rows).to_csv(RES / "rev2_b_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(byclass).to_csv(RES / "rev2_b_by_class.csv", index=False, encoding="utf-8-sig")
    pd.concat(cms).to_csv(RES / "rev2_b_confusion.csv", encoding="utf-8-sig")

    key, _f, pipe, name = best
    log(f"M: mô hình tham chiếu = {name}")

    # Gain theo biến và theo nhóm nghiệp vụ
    prep, clf = pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]
    names = prep.get_feature_names()
    gain = pd.Series(clf.feature_importances_, index=names).astype(float)
    gain = gain / gain.sum() * 100

    # SHAP trên cỡ mẫu thống nhất
    Xt = pd.DataFrame(prep.transform(Xva), columns=names)
    Xs = Xt.sample(min(SHAP_N, len(Xt)), random_state=RANDOM_STATE)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xs))
    imp = np.abs(sv).mean(axis=(0, 2)) if sv.ndim == 3 else np.abs(sv).mean(axis=0)
    shp = pd.Series(imp, index=names)
    d = pd.DataFrame({"gain_pct": gain, "mean_abs_shap": shp,
                      "shap_pct": shp / shp.sum() * 100})
    d["group"] = [feature_to_group(f) for f in d.index]
    d.sort_values("gain_pct", ascending=False).to_csv(
        RES / "rev2_b_gain_shap_feature.csv", encoding="utf-8-sig")
    g = d.groupby("group")[["gain_pct", "shap_pct"]].sum().sort_values("gain_pct", ascending=False)
    g.to_csv(RES / "rev2_b_gain_shap_group.csv", encoding="utf-8-sig")
    from scipy import stats
    log(f"M SHAP n={len(Xs)}; nhóm nghiệp vụ:\n{g.round(1)}\n"
        f"Spearman biến rho={stats.spearmanr(d.gain_pct, d.shap_pct).statistic:.3f} "
        f"p={stats.spearmanr(d.gain_pct, d.shap_pct).pvalue:.4g} | "
        f"nhóm rho={stats.spearmanr(g.gain_pct, g.shap_pct).statistic:.3f} "
        f"p={stats.spearmanr(g.gain_pct, g.shap_pct).pvalue:.4g}")

    # Top-k theo thứ tự Gain
    order = gain.sort_values(ascending=False).index.tolist()
    rows = []
    for k in [3, 5, 8, 11, 14, 17, 20]:
        sel = order[:k]
        ck, nk = [c for c in cat_p if c in sel], [c for c in num_p if c in sel]
        p = build_pipeline(key, ck, nk, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr[nk + ck], ytr)
        m, _cm, _ = evaluate(p, Xva[nk + ck], yva)
        rows.append({"k": k, "q_k_pct": gain.sort_values(ascending=False)[:k].sum(), **m})
        log(f"M  Top-{k}: Macro-F1={m['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_b_topk.csv", index=False, encoding="utf-8-sig")


# ── S1/S2/S3 ─────────────────────────────────────────────────────────────────
def part_s1():
    rows = []
    for label, flag in [("Mã chi nhánh là biến số (cấu hình chính)", False),
                        ("Mã chi nhánh là biến phân loại", True)]:
        X, y, Xh, yh, num_p, cat_p = load_b(branch_as_categorical=flag)
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
        for name in ["LightGBM", "XGBoost", "Logistic Regression"]:
            p = build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE,
                               imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
            p.fit(Xtr, ytr)
            for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
                m, _c, _ = evaluate(p, XX, yy)
                rows.append({"cau_hinh": label, "model": name, "split": split, **m})
            log(f"S1 {label} | {name}: val={rows[-2]['f1_macro']:.4f} "
                f"ngoài TG={rows[-1]['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_s1_branch_encoding.csv", index=False,
                              encoding="utf-8-sig")


def part_s2():
    rows = []
    X, y, Xh, yh, num_p, cat_p = load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    for label, strat in [("SMOTE (cấu hình chính)", "custom"), ("SMOTE-NC", "smote_nc")]:
        for name in ["LightGBM", "XGBoost"]:
            p = build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE,
                               imbalance_strategy=strat,
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
            p.fit(Xtr, ytr)
            for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
                m, _c, _ = evaluate(p, XX, yy)
                rows.append({"bo": "B", "chien_luoc": label, "model": name, "split": split, **m})
            log(f"S2 B {label} | {name}: val={rows[-2]['f1_macro']:.4f} "
                f"ngoài TG={rows[-1]['f1_macro']:.4f}")

    dfa = engineer_features(parse_dates(pd.read_excel(DATA_RAW)))
    ya = dfa[TARGET_COL].astype(int).values - 1
    Xa = dfa[CAT_A + NUM_A]
    Xatr, Xate, yatr, yate = train_test_split(Xa, ya, test_size=TEST_SIZE, stratify=ya,
                                              random_state=RANDOM_STATE)
    for label, strat in [("SMOTE (cấu hình chính)", "smote_moderate"), ("SMOTE-NC", "smote_nc")]:
        p = build_pipeline("xgboost", CAT_A, NUM_A, random_state=RANDOM_STATE,
                           imbalance_strategy=strat)
        p.fit(Xatr, yatr)
        m, _c, _ = evaluate(p, Xate, yate)
        rows.append({"bo": "A", "chien_luoc": label, "model": "XGBoost",
                     "split": "validation", **m})
        log(f"S2 A {label} | XGBoost: {m['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_s2_smotenc.csv", index=False, encoding="utf-8-sig")


def part_s3():
    """One-Hot cho hai biến định danh nhiều danh mục của bộ A."""
    dfa = engineer_features(parse_dates(pd.read_excel(DATA_RAW)))
    ya = dfa[TARGET_COL].astype(int).values - 1
    Xa = dfa[CAT_A + NUM_A]
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=TEST_SIZE, stratify=ya,
                                          random_state=RANDOM_STATE)
    rows = []
    # Cấu hình chính: LabelEncoder (nằm trong CreditPreprocessor)
    for name in ["XGBoost", "Logistic Regression"]:
        p = build_pipeline(MODELS[name], CAT_A, NUM_A, random_state=RANDOM_STATE,
                           imbalance_strategy="smote_moderate")
        p.fit(Xtr, ytr)
        m, _c, _ = evaluate(p, Xte, yte)
        rows.append({"ma_hoa": "LabelEncoder (cấu hình chính)", "model": name, **m})
        log(f"S3 LabelEncoder | {name}: {m['f1_macro']:.4f}")

    # One-Hot học trên tập huấn luyện cho hai biến nhiều danh mục
    hi = ["CURRMIACCTTYPCD", "MUCDICHVAY"]
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20)
    enc.fit(Xtr[hi].astype(str))
    def oh(d):
        arr = enc.transform(d[hi].astype(str))
        cols = [f"OH_{i}" for i in range(arr.shape[1])]
        out = d.drop(columns=hi).copy()
        out[cols] = arr
        return out, cols
    Xtr_o, ohcols = oh(Xtr)
    Xte_o, _ = oh(Xte)
    cat_o = [c for c in CAT_A if c not in hi]
    num_o = NUM_A + ohcols
    log(f"S3 One-Hot sinh {len(ohcols)} cột (min_frequency=20)")
    for name in ["XGBoost", "Logistic Regression"]:
        p = build_pipeline(MODELS[name], cat_o, num_o, random_state=RANDOM_STATE,
                           imbalance_strategy="smote_moderate")
        p.fit(Xtr_o, ytr)
        m, _c, _ = evaluate(p, Xte_o, yte)
        rows.append({"ma_hoa": "One-Hot (2 biến nhiều danh mục)", "model": name, **m})
        log(f"S3 One-Hot | {name}: {m['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_s3_onehot.csv", index=False, encoding="utf-8-sig")


# ── I ────────────────────────────────────────────────────────────────────────
def part_i():
    """IV tính trên toàn mẫu so với chỉ trên tập huấn luyện."""
    out = []
    # Bộ B
    df = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    num_raw, cat_raw = build_raw_feature_pool(df)
    de, eng = engineer_business_features(df)
    num_p, cat_p = num_raw + eng, cat_raw
    idx = np.arange(len(de))
    ytmp = de[CREDIT_INFO_TARGET_COL].astype(int).values
    itr, _ite = train_test_split(idx, test_size=TEST_SIZE, stratify=ytmp,
                                 random_state=RANDOM_STATE)
    cols = num_p + cat_p + [CREDIT_INFO_TARGET_COL]
    full = compute_iv_table(de[cols].copy(), CREDIT_INFO_TARGET_COL, good_class=1)
    trn = compute_iv_table(de.iloc[itr][cols].copy(), CREDIT_INFO_TARGET_COL, good_class=1)
    k = full.columns[0]
    m = full[[k, "iv"]].merge(trn[[k, "iv"]], on=k, suffixes=("_toan_mau", "_chi_train"))
    m["bo"] = "B"
    out.append(m)
    # Bộ A
    dfa = engineer_features(parse_dates(pd.read_excel(DATA_RAW)))
    ia = np.arange(len(dfa))
    itr2, _ = train_test_split(ia, test_size=TEST_SIZE,
                               stratify=dfa[TARGET_COL].astype(int).values,
                               random_state=RANDOM_STATE)
    colsa = CAT_A + NUM_A + ["SEX", "LOAIKH", TARGET_COL]
    colsa = [c for c in colsa if c in dfa.columns]
    fa = compute_iv_table(dfa[colsa].copy(), TARGET_COL, good_class=1)
    ta = compute_iv_table(dfa.iloc[itr2][colsa].copy(), TARGET_COL, good_class=1)
    ka = fa.columns[0]
    ma = fa[[ka, "iv"]].merge(ta[[ka, "iv"]], on=ka, suffixes=("_toan_mau", "_chi_train"))
    ma["bo"] = "A"
    out.append(ma)
    r = pd.concat(out, ignore_index=True)
    r["chenh_lech"] = (r.iv_chi_train - r.iv_toan_mau).abs()
    r = r.sort_values("chenh_lech", ascending=False)
    r.to_csv(RES / "rev2_iv_train_only.csv", index=False, encoding="utf-8-sig")
    log(f"I: IV toàn mẫu so với chỉ tập huấn luyện\n{r.head(12).round(4).to_string()}")
    log(f"I: chênh lệch tuyệt đối lớn nhất {r.chenh_lech.max():.4f}; "
        f"không biến nào đổi mức cảnh báo 1,0: "
        f"{((r.iv_toan_mau > 1) == (r.iv_chi_train > 1)).all()}")


# ── T ────────────────────────────────────────────────────────────────────────
def part_t():
    """Xác nhận bài toán chuyển nhóm không đổi sau khi sửa C.1."""
    r1, r2, r3 = (load_credit_info(f) for f in
                  (DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2))
    rows = []
    for nm, (a, b) in {"cap1": (r1, r2), "cap2": (r2, r3)}.items():
        dm = build_transition_dataset(a, b, id_col=TRANSITION_ID_COL,
                                      target_col=CREDIT_INFO_TARGET_COL)
        num_raw, cat_raw = build_raw_feature_pool(dm)
        de, eng = engineer_business_features(dm)
        num_p, cat_p = num_raw + eng, cat_raw
        X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
        for name in ["XGBoost", "LightGBM"]:
            strat = "none" if name == "XGBoost" else "custom"
            kw = {"random_state": RANDOM_STATE, "imbalance_strategy": strat}
            if strat == "custom":
                kw["custom_smote_strategy"] = TRANSITION_SMOTE_STRATEGY
            if name == "XGBoost":
                kw["eval_metric"] = "logloss"
            pipe = build_pipeline(MODELS[name], cat_p, num_p, **kw)
            _pr, s = repeated_stratified_binary_eval(
                pipe, X, y, n_splits=TRANSITION_CV_SPLITS,
                n_repeats=TRANSITION_CV_REPEATS, random_state=RANDOM_STATE)
            rows.append({"cap": nm, "model": name, "chien_luoc": strat, **s})
            log(f"T {nm} {name}: AP={s['pr_auc_mean']:.4f}±{s['pr_auc_std']:.4f} "
                f"ROC-AUC={s['roc_auc_mean']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_transition_confirm.csv", index=False,
                              encoding="utf-8-sig")


# ── G ────────────────────────────────────────────────────────────────────────
GRID_B = {
    "Logistic Regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "Decision Tree": [{"max_depth": d, "min_samples_leaf": l}
                      for d in (5, 8, 12, 20) for l in (5, 20)],
    "Random Forest": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
                      for n in (200, 400) for d in (10, 20, None) for l in (5, 15)],
    "XGBoost": [{"max_depth": d, "learning_rate": r, "n_estimators": n}
                for d in (3, 6, 9) for r in (0.03, 0.1) for n in (200, 400)],
    "LightGBM": [{"max_depth": d, "learning_rate": r, "n_estimators": n}
                 for d in (3, 6, 9) for r in (0.03, 0.1) for n in (200, 400)],
    "CatBoost": [{"depth": d, "learning_rate": r, "iterations": n}
                 for d in (4, 6, 8) for r in (0.03, 0.1) for n in (200, 400)],
}


def part_g():
    """Chạy lại lưới siêu tham số của bộ B sau khi sửa C.1 (60 cấu hình)."""
    X, y, _Xh, _yh, num_p, cat_p = load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    rows = []
    for name, grid in GRID_B.items():
        key = MODELS[name]
        t0 = time.time()
        # cấu hình mặc định (giá trị trong models.py), không nằm trong lưới
        p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr, ytr)
        default_f1 = evaluate(p, Xva, yva)[0]["f1_macro"]
        scores = []
        for params in grid:
            p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                               imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY,
                               model_params=params)
            p.fit(Xtr, ytr)
            scores.append(evaluate(p, Xva, yva)[0]["f1_macro"])
        s = np.array(scores)
        rows.append({"bo": "B", "model": name, "n_cau_hinh": len(grid),
                     "f1_mac_dinh": default_f1, "f1_tot_nhat": s.max(),
                     "cai_thien_pct": (s.max() / default_f1 - 1) * 100,
                     "f1_nho_nhat": s.min(), "bien_do": s.max() - s.min(),
                     "do_lech_chuan": s.std(ddof=0)})
        log(f"G {name}: mặc định={default_f1:.4f} tốt nhất={s.max():.4f} "
            f"nhỏ nhất={s.min():.4f} biên độ={s.max()-s.min():.4f} ({time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(RES / "rev2_b_grid.csv", index=False, encoding="utf-8-sig")


def part_imb():
    """Chạy lại so sánh bốn chiến lược mất cân bằng của bộ B sau khi sửa C.1."""
    X, y, _Xh, _yh, num_p, cat_p = load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    rows = []
    for label, strat in [("SMOTE", "custom"), ("Không xử lý", "none"),
                         ("Kết hợp", "smote_class_weight"), ("Trọng số lớp", "class_weight")]:
        p = build_pipeline("lightgbm", cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy=strat,
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr, ytr)
        m, _cm, (pr, rc, f1, sup) = evaluate(p, Xva, yva)
        rows.append({"chien_luoc": label, **m,
                     **{f"recall_n{i+1}": rc[i] for i in range(5)}})
        log(f"IMB {label}: F1m={m['f1_macro']:.4f} ROC={m['roc_auc_macro']:.4f} "
            f"R2={rc[1]:.3f} R3={rc[2]:.3f} R4={rc[3]:.3f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_b_imbalance.csv", index=False, encoding="utf-8-sig")


def part_leak():
    """Chạy lại thí nghiệm loại biến lãi của bộ B sau khi sửa C.1."""
    X, y, Xh, yh, num_p, cat_p = load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    off, on = "Lãi chưa thu hạch toán ngoại bảng", "Lãi phải thu hạch toán nội bảng"
    rows = []
    for label, drop in [("Đầy đủ 20 biến", []), ("Bỏ lãi ngoại bảng", [off]),
                        ("Bỏ cả hai biến lãi", [off, on])]:
        num = [c for c in num_p if c not in drop]
        p = build_pipeline("lightgbm", cat_p, num, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr[num + cat_p], ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xh, yh)]:
            m, _c, _ = evaluate(p, XX[num + cat_p], yy)
            rows.append({"cau_hinh": label, "split": split, **m})
        log(f"LEAK {label}: val={rows[-2]['f1_macro']:.4f} "
            f"ngoài TG={rows[-1]['f1_macro']:.4f} ordMAE={rows[-2]['ordinal_mae']:.3f}")
    pd.DataFrame(rows).to_csv(RES / "rev2_b_leak.csv", index=False, encoding="utf-8-sig")


def main():
    w = set(a.upper() for a in sys.argv[1:]) or {"M", "S1", "S2", "S3", "I", "T"}
    for k, f in [("LEAK", part_leak), ("IMB", part_imb), ("G", part_g)]:
        if k in w:
            f()
    for k, f in [("I", part_i), ("M", part_m), ("S1", part_s1), ("S2", part_s2),
                 ("S3", part_s3), ("T", part_t)]:
        if k in w:
            f()
    log("HOÀN TẤT")


if __name__ == "__main__":
    main()
