"""Số liệu bổ sung cho bản rút gọn/chỉnh sửa luận văn.

Mỗi phần (A1..T2) tạo một nhóm số liệu mà bản chỉnh sửa cần nhưng chưa có
trong code/results:

  A1  Bộ A, cấu hình chính của luận văn (3 biến phân loại + 8 biến số,
      smote_moderate): sáu mô hình, kèm Balanced Accuracy, Ordinal MAE và
      ma trận nhầm lẫn — gồm cả Logistic Regression.
  A2  Bộ A: Gain và SHAP tính trên CÙNG một mô hình XGBoost 11 biến.
  A3  Bộ A: bốn chiến lược mất cân bằng trên đúng pipeline của A1, để bảng
      chiến lược và bảng mô hình dùng chung một lần chạy.
  A4  Bộ A: Top-k (tập 13 biến) xếp theo Gain của XGBoost 13 biến.
  B1  Bộ B: Ordinal MAE / Balanced Accuracy suy từ ma trận nhầm lẫn đã lưu.
  B2  Bộ B: độ nhạy khi điền 0 (thay vì trung vị) cho ba biến cơ cấu nợ.
  B3  Bộ B: rà soát rò rỉ — tỷ lệ lãi ngoại bảng > 0 theo nhóm nợ.
  T1  Chuyển nhóm cặp 1: kiểm định t hiệu chỉnh (Nadeau–Bengio) giữa các
      mô hình trên PR-AUC từng phần kiểm định.
  T2  Chuyển nhóm cặp 1: SHAP trên mẫu phân tầng theo nhãn.
  S   Spearman kèm p-value cho đối chiếu SHAP nhóm giữa hai bộ.

Chạy:  cd code && py -m src.generate_thesis_revision
Output: code/results/revision_*.csv
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
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, recall_score)
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, RANDOM_STATE, TEST_SIZE,
    DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, CREDIT_INFO_TARGET_COL,
    CREDIT_INFO_SMOTE_STRATEGY, TRANSITION_ID_COL,
)
from src.preprocessing import parse_dates, engineer_features
from src.models import build_pipeline
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL,
)

RES = Path(__file__).parent.parent / "results"

# Cấu hình chính của luận văn cho bộ A (Mục 2.3.4): 3 phân loại + 8 số.
CAT_A = ["MJACCTTYPCD", "CURRMIACCTTYPCD", "MUCDICHVAY"]
NUM_A = ["BASE_BAL", "CURR_BAL", "LAISUAT", "ORGNBR", "PARENTORGNBR",
         "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY", "UTIL_RATE"]
CAT_A13 = CAT_A + ["SEX", "LOAIKH"]
GROUPS_A = {
    "Dư nợ và hạn mức sử dụng": ["BASE_BAL", "CURR_BAL", "UTIL_RATE"],
    "Lãi suất": ["LAISUAT"],
    "Kỳ hạn khoản vay": ["LOAN_TENURE_DAYS", "DAYS_TO_MATURITY"],
    "Đơn vị/Chi nhánh": ["ORGNBR", "PARENTORGNBR"],
    "Sản phẩm và mục đích vay": ["MJACCTTYPCD", "CURRMIACCTTYPCD", "MUCDICHVAY"],
    "Nhân khẩu học có IV thấp": ["SEX", "LOAIKH"],
}
MODELS = {"Logistic Regression": "logistic", "Decision Tree": "decision_tree",
          "Random Forest": "random_forest", "XGBoost": "xgboost",
          "LightGBM": "lightgbm", "CatBoost": "catboost"}
RESTRUCT = ["Số lần cơ cấu lại thời hạn trả nợ", "Số tiền nợ gốc cơ cấu", "Số tiền nợ lãi cơ cấu"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ordinal_mae_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    k = cm.shape[0]
    dist = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    return float((cm * dist).sum() / cm.sum())


def metrics_5class(y_true, y_pred, proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "ordinal_mae": ordinal_mae_from_cm(cm),
        **{f"recall_n{c + 1}": r for c, r in
           enumerate(recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None))},
    }, cm


def load_a():
    df = engineer_features(parse_dates(pd.read_excel(DATA_RAW)))
    y = df[TARGET_COL].astype(int).values - 1
    X = df[CAT_A13 + NUM_A]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def part_a(Xtr, Xte, ytr, yte):
    # A1 — sáu mô hình, cấu hình chính
    rows, cms = [], []
    for name, key in MODELS.items():
        t0 = time.time()
        pipe = build_pipeline(key, CAT_A, NUM_A, random_state=RANDOM_STATE,
                              imbalance_strategy="smote_moderate")
        pipe.fit(Xtr[CAT_A + NUM_A], ytr)
        pred = np.asarray(pipe.predict(Xte[CAT_A + NUM_A])).ravel().astype(int)
        m, cm = metrics_5class(yte, pred, pipe.predict_proba(Xte[CAT_A + NUM_A]))
        rows.append({"model": name, **m})
        cms.append(pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                                columns=[f"pred_{i}" for i in range(1, 6)]).assign(model=name))
        log(f"A1 {name}: F1m={m['f1_macro']:.4f} ordMAE={m['ordinal_mae']:.4f} ({time.time()-t0:.0f}s)")
    pd.DataFrame(rows).to_csv(RES / "revision_a_models.csv", index=False, encoding="utf-8-sig")
    pd.concat(cms).to_csv(RES / "revision_a_confusion.csv", encoding="utf-8-sig")

    # A2 — Gain và SHAP trên cùng XGBoost 11 biến
    pipe = build_pipeline("xgboost", CAT_A, NUM_A, random_state=RANDOM_STATE,
                          imbalance_strategy="smote_moderate")
    pipe.fit(Xtr[CAT_A + NUM_A], ytr)
    prep, clf = pipe.named_steps["preprocessor"], pipe.named_steps["classifier"]
    names = prep.get_feature_names()
    gain = clf.get_booster().get_score(importance_type="total_gain")
    gain = np.array([gain.get(f"f{i}", 0.0) for i in range(len(names))])
    Xt = pd.DataFrame(prep.transform(Xte[CAT_A + NUM_A]), columns=names)
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
    if sv.ndim == 3 and sv.shape[0] == len(names):  # (feat, n, class) hiếm gặp
        sv = np.moveaxis(sv, 0, 1)
    if sv.ndim == 3 and sv.shape[1] != len(names):  # (class, n, feat)
        sv = np.moveaxis(sv, 0, -1)
    shap_imp = np.abs(sv).mean(axis=(0, 2)) if sv.shape[1] == len(names) else np.abs(sv).mean(axis=0)
    feat = pd.DataFrame({"feature": names, "gain_total": gain, "mean_abs_shap": shap_imp})
    feat["gain_pct"] = feat["gain_total"] / feat["gain_total"].sum() * 100
    feat["shap_pct"] = feat["mean_abs_shap"] / feat["mean_abs_shap"].sum() * 100
    feat["group"] = feat["feature"].map({c: g for g, cs in GROUPS_A.items() for c in cs})
    feat.sort_values("shap_pct", ascending=False).to_csv(
        RES / "revision_a_gain_vs_shap_feature.csv", index=False, encoding="utf-8-sig")
    grp = feat.groupby("group")[["gain_pct", "shap_pct"]].sum().sort_values("shap_pct", ascending=False)
    grp.to_csv(RES / "revision_a_gain_vs_shap_group.csv", encoding="utf-8-sig")
    rf = stats.spearmanr(feat["gain_pct"], feat["shap_pct"])
    rg = stats.spearmanr(grp["gain_pct"], grp["shap_pct"])
    pd.DataFrame([{"level": "feature", "n": len(feat), "rho": rf.statistic, "p": rf.pvalue},
                  {"level": "group", "n": len(grp), "rho": rg.statistic, "p": rg.pvalue}]
                 ).to_csv(RES / "revision_a_gain_vs_shap_spearman.csv", index=False)
    log(f"A2 group Gain vs SHAP:\n{grp}\nrho_feat={rf.statistic:.3f} p={rf.pvalue:.4f}")

    # A3 — bốn chiến lược, cùng pipeline
    rows = []
    for label, strat in [("SMOTE", "smote_moderate"), ("Không xử lý", "none"),
                         ("Kết hợp", "smote_class_weight"), ("Trọng số lớp", "class_weight")]:
        p = build_pipeline("xgboost", CAT_A, NUM_A, random_state=RANDOM_STATE, imbalance_strategy=strat)
        p.fit(Xtr[CAT_A + NUM_A], ytr)
        pred = np.asarray(p.predict(Xte[CAT_A + NUM_A])).ravel().astype(int)
        m, _ = metrics_5class(yte, pred, p.predict_proba(Xte[CAT_A + NUM_A]))
        rows.append({"strategy": label, **m})
    pd.DataFrame(rows).to_csv(RES / "revision_a_imbalance.csv", index=False, encoding="utf-8-sig")
    log(f"A3:\n{pd.DataFrame(rows)[['strategy', 'f1_macro', 'roc_auc_macro']]}")

    # A4 — Top-k trên tập 13 biến, thứ tự theo Gain của XGBoost 13 biến
    p13 = build_pipeline("xgboost", CAT_A13, NUM_A, random_state=RANDOM_STATE,
                         imbalance_strategy="smote_moderate")
    p13.fit(Xtr[CAT_A13 + NUM_A], ytr)
    n13 = p13.named_steps["preprocessor"].get_feature_names()
    g13 = p13.named_steps["classifier"].get_booster().get_score(importance_type="total_gain")
    order = [n13[i] for i in np.argsort([-g13.get(f"f{i}", 0.0) for i in range(len(n13))])]
    rows = []
    for k in [3, 5, 7, 9, 11, 13]:
        sel = order[:k]
        cat_k, num_k = [c for c in CAT_A13 if c in sel], [c for c in NUM_A if c in sel]
        p = build_pipeline("xgboost", cat_k, num_k, random_state=RANDOM_STATE,
                           imbalance_strategy="smote_moderate")
        p.fit(Xtr[cat_k + num_k], ytr)
        pred = np.asarray(p.predict(Xte[cat_k + num_k])).ravel().astype(int)
        m, _ = metrics_5class(yte, pred, p.predict_proba(Xte[cat_k + num_k]))
        rows.append({"k": k, "features": ";".join(sel), **m})
    pd.DataFrame(rows).to_csv(RES / "revision_a_topk.csv", index=False, encoding="utf-8-sig")
    log(f"A4:\n{pd.DataFrame(rows)[['k', 'f1_macro', 'roc_auc_macro']]}")


def part_b1():
    rows = []
    for split in ["validation", "holdout"]:
        for key, name in [("logistic", "Logistic Regression"), ("decision_tree", "Decision Tree"),
                          ("random_forest", "Random Forest"), ("xgboost", "XGBoost"),
                          ("lightgbm", "LightGBM"), ("catboost", "CatBoost")]:
            f = RES / f"credit_info_confusion_{key}_{split}.csv"
            cm = pd.read_csv(f, index_col=0).select_dtypes("number").values
            cm = cm[:5, :5]
            rec = np.diag(cm) / cm.sum(axis=1)
            rows.append({"split": split, "model": name, "ordinal_mae": ordinal_mae_from_cm(cm),
                         "balanced_accuracy": rec.mean(),
                         "share_off_by_2plus": float(sum(cm[i, j] for i in range(5) for j in range(5)
                                                         if abs(i - j) >= 2) / cm.sum())})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "revision_b_ordinal.csv", index=False, encoding="utf-8-sig")
    log(f"B1:\n{df}")


def load_b():
    df_raw = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    num_raw, cat_raw = build_raw_feature_pool(df_raw)
    df_eng, eng = engineer_business_features(df_raw)
    num_p, cat_p = num_raw + eng, cat_raw
    X, y = prepare_credit_info(df_eng, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)
    df_ho = load_credit_info(DATA_CREDIT_INFO_TEST)
    df_ho_eng, _ = engineer_business_features(df_ho)
    Xho, yho = prepare_credit_info(df_ho_eng, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)
    return df_raw, X, y, Xho, yho, num_p, cat_p


def part_b2_b3(df_raw, X, y, Xho, yho, num_p, cat_p):
    # B3 — rà soát rò rỉ: lãi ngoại bảng / nội bảng dương theo nhóm nợ
    g = df_raw[CREDIT_INFO_TARGET_COL]
    rows = []
    for col in ["Lãi chưa thu hạch toán ngoại bảng", "Lãi phải thu hạch toán nội bảng"]:
        pos = (df_raw[col].fillna(0) > 0)
        for grp in sorted(g.dropna().unique()):
            mask = g == grp
            rows.append({"bien": col, "nhom": int(grp), "n": int(mask.sum()),
                         "ty_le_duong_pct": pos[mask].mean() * 100})
        rows.append({"bien": col, "nhom": "tong", "n": len(g), "ty_le_duong_pct": pos.mean() * 100,
                     "ty_le_nhom2_5_trong_so_duong_pct": (g[pos] >= 2).mean() * 100})
    pd.DataFrame(rows).to_csv(RES / "revision_b_offbalance_leakage.csv", index=False, encoding="utf-8-sig")
    log(f"B3:\n{pd.DataFrame(rows)}")
    for c in RESTRUCT:
        log(f"  {c}: thiếu {X[c].isna().mean()*100:.2f}%, trung vị phần quan sát = {X[c].median():.2f}")

    # B2 — điền trung vị (hiện tại) vs điền 0 cho ba biến cơ cấu
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    rows = []
    for variant in ["median (hiện tại)", "điền 0"]:
        def fix(d):
            d = d.copy()
            if variant == "điền 0":
                d[RESTRUCT] = d[RESTRUCT].fillna(0)
            return d
        for name, key in [("LightGBM", "lightgbm"), ("XGBoost", "xgboost"), ("Random Forest", "random_forest")]:
            p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE, imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
            p.fit(fix(Xtr), ytr)
            for split, XX, yy in [("validation", Xva, yva), ("holdout", Xho, yho)]:
                pred = np.asarray(p.predict(fix(XX))).ravel().astype(int)
                m, _ = metrics_5class(yy, pred, p.predict_proba(fix(XX)))
                rows.append({"variant": variant, "model": name, "split": split, **m})
            log(f"B2 {variant} {name}: val F1m={rows[-2]['f1_macro']:.4f} ho F1m={rows[-1]['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "revision_b_restructure_impute.csv", index=False, encoding="utf-8-sig")


def corrected_ttest(a, b, test_train_ratio=0.25):
    """Kiểm định t hiệu chỉnh cho kiểm định chéo lặp (Nadeau & Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = d.var(ddof=1)
    t = d.mean() / np.sqrt((1 / k + test_train_ratio) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return d.mean(), t, p


def part_t1():
    out = []
    best = {"XGBoost": "none", "LightGBM": "custom", "Random Forest": "none",
            "CatBoost": "none", "Decision Tree": "custom", "Logistic Regression": "none"}
    for pair, f in [("cap1", "transition_cv_per_repeat.csv"), ("cap2", "transition_pair2_cv_per_repeat.csv")]:
        df = pd.read_csv(RES / f)
        if pair == "cap2" and (RES / "transition_pair2_catboost_per_repeat.csv").exists():
            cb = pd.read_csv(RES / "transition_pair2_catboost_per_repeat.csv")
            df = pd.concat([df[df["model"] != "CatBoost"], cb], ignore_index=True)
        ref = df[(df["model"] == "XGBoost") & (df["strategy"] == "none")]["pr_auc"].values
        for m, s in best.items():
            if m == "XGBoost":
                continue
            other = df[(df["model"] == m) & (df["strategy"] == s)]["pr_auc"].values
            if len(other) != len(ref):
                continue
            md, t, p = corrected_ttest(ref, other)
            w = stats.wilcoxon(ref, other)
            out.append({"pair": pair, "comparator": f"{m} ({s})", "mean_diff_pr_auc": md,
                        "t_corrected": t, "p_corrected": p, "p_wilcoxon_naive": w.pvalue})
    df = pd.DataFrame(out)
    for pair in df["pair"].unique():
        idx = df["pair"] == pair
        ps = df.loc[idx, "p_corrected"].values
        order = np.argsort(ps)
        holm = np.empty_like(ps)
        running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, (len(ps) - rank) * ps[i]))
            holm[i] = running
        df.loc[idx, "p_holm"] = holm
    df.to_csv(RES / "revision_transition_tests.csv", index=False, encoding="utf-8-sig")
    log(f"T1:\n{df}")


def part_t2():
    df_t = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    df_t1 = load_credit_info(DATA_CREDIT_INFO_TEST)
    dm = build_transition_dataset(df_t, df_t1, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL)
    num_raw, cat_raw = build_raw_feature_pool(dm)
    de, eng = engineer_business_features(dm)
    num_p, cat_p = num_raw + eng, cat_raw
    X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    p = build_pipeline("xgboost", cat_p, num_p, random_state=RANDOM_STATE,
                       imbalance_strategy="none", eval_metric="logloss")
    p.fit(Xtr, ytr)
    prep, clf = p.named_steps["preprocessor"], p.named_steps["classifier"]
    names = prep.get_feature_names()
    Xt = pd.DataFrame(prep.transform(Xva), columns=names)
    rng = np.random.RandomState(RANDOM_STATE)
    pos_idx = np.where(yva == 1)[0]
    neg_idx = rng.choice(np.where(yva == 0)[0], 500 - len(pos_idx), replace=False)
    rows = []
    sv = np.asarray(shap.TreeExplainer(clf).shap_values(Xt))
    for lab, idx in [("duong (xau di)", pos_idx), ("am", neg_idx),
                     ("phan tang 500", np.concatenate([pos_idx, neg_idx]))]:
        imp = np.abs(sv[idx]).mean(axis=0)
        for f, v in zip(names, imp):
            rows.append({"mau": lab, "n": len(idx), "feature": f, "mean_abs_shap": v})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "revision_transition_shap_stratified.csv", index=False, encoding="utf-8-sig")
    top = df[df["mau"] == "phan tang 500"].sort_values("mean_abs_shap", ascending=False).head(10)
    log(f"T2 (n_pos={len(pos_idx)}):\n{top}")


def part_s():
    a = {"Kỳ hạn": 39.9, "Dư nợ": 27.3, "Chi nhánh": 15.8, "Lãi suất": 8.6, "Sản phẩm": 8.4}
    b = {"Kỳ hạn": 38.2, "Dư nợ": 44.5, "Chi nhánh": 5.6, "Lãi suất": 6.4, "Sản phẩm": 4.1}
    k = list(a)
    r = stats.spearmanr([a[x] for x in k], [b[x] for x in k])
    # p chính xác bằng hoán vị (n=5 → 120 hoán vị)
    import itertools
    ra = stats.rankdata([a[x] for x in k])
    rb = stats.rankdata([b[x] for x in k])
    perms = [stats.spearmanr(ra, np.array(pp)).statistic for pp in itertools.permutations(rb)]
    p_exact = np.mean(np.abs(perms) >= abs(r.statistic) - 1e-12)
    log(f"S: rho={r.statistic:.3f} p_asym={r.pvalue:.4f} p_exact={p_exact:.4f}")
    pd.DataFrame([{"rho": r.statistic, "p_asymptotic": r.pvalue, "p_exact_permutation": p_exact}]
                 ).to_csv(RES / "revision_spearman_ab.csv", index=False)


def part_leak_a(Xtr, Xte, ytr, yte):
    """Rà soát DAYS_TO_MATURITY (bộ A): loại biến và tỷ lệ đã đáo hạn theo nhóm."""
    rows = []
    for label, drop in [("Cấu hình chính (11 biến)", []),
                        ("Bỏ DAYS_TO_MATURITY", ["DAYS_TO_MATURITY"]),
                        ("Bỏ DAYS_TO_MATURITY và LOAN_TENURE_DAYS", ["DAYS_TO_MATURITY", "LOAN_TENURE_DAYS"])]:
        num = [c for c in NUM_A if c not in drop]
        p = build_pipeline("xgboost", CAT_A, num, random_state=RANDOM_STATE, imbalance_strategy="smote_moderate")
        p.fit(Xtr[CAT_A + num], ytr)
        pred = np.asarray(p.predict(Xte[CAT_A + num])).ravel().astype(int)
        m, _ = metrics_5class(yte, pred, p.predict_proba(Xte[CAT_A + num]))
        rows.append({"cau_hinh": label, **m})
    pd.DataFrame(rows).to_csv(RES / "revision_a_dtm_ablation.csv", index=False, encoding="utf-8-sig")
    log(f"L-A:\n{pd.DataFrame(rows)[['cau_hinh', 'f1_macro', 'roc_auc_macro', 'ordinal_mae']]}")
    df = engineer_features(parse_dates(pd.read_excel(DATA_RAW)))
    cutoff = df["OPEN_DATE"].max()
    share = (df.assign(da_dao_han=df["NGAYDENHAN"] < cutoff)
             .groupby(TARGET_COL)["da_dao_han"].mean() * 100)
    share.rename("ty_le_dao_han_truoc_ngay_mo_cuoi_pct").to_csv(
        RES / "revision_a_matured_share.csv", encoding="utf-8-sig")
    log(f"L-A share đáo hạn trước {cutoff.date()}:\n{share}")


def part_leak_b(df_raw, X, y, Xho, yho, num_p, cat_p):
    """Rà soát hai biến lãi (bộ B): loại biến và đo lại Macro-F1."""
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    off, on = "Lãi chưa thu hạch toán ngoại bảng", "Lãi phải thu hạch toán nội bảng"
    rows = []
    for label, drop in [("Đầy đủ 20 biến", []), ("Bỏ lãi ngoại bảng", [off]),
                        ("Bỏ cả hai biến lãi", [off, on])]:
        num = [c for c in num_p if c not in drop]
        p = build_pipeline("lightgbm", cat_p, num, random_state=RANDOM_STATE, imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr[num + cat_p], ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xho, yho)]:
            pred = np.asarray(p.predict(XX[num + cat_p])).ravel().astype(int)
            m, _ = metrics_5class(yy, pred, p.predict_proba(XX[num + cat_p]))
            rows.append({"cau_hinh": label, "split": split, **m})
        log(f"L-B {label}: val={rows[-2]['f1_macro']:.4f} ho={rows[-1]['f1_macro']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "revision_b_interest_ablation.csv", index=False, encoding="utf-8-sig")


def main():
    which = set(sys.argv[1:]) or {"A", "B1", "B2", "T1", "T2", "S"}
    if "LA" in which:
        part_leak_a(*load_a())
    if "LB" in which:
        part_leak_b(*load_b())
    if "S" in which:
        part_s()
    if "T1" in which:
        part_t1()
    if "B1" in which:
        part_b1()
    if "A" in which:
        part_a(*load_a())
    if "B2" in which:
        part_b2_b3(*load_b())
    if "T2" in which:
        part_t2()
    log("HOÀN TẤT")


if __name__ == "__main__":
    main()
