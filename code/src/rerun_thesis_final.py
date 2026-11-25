"""Chạy lại TOÀN BỘ số liệu của bản luận văn sửa (tháng 11/2026) trên một máy,
một lần chạy, chỉ dùng dữ liệu THẬT:

  * bộ A  — Data_credit_rating_VN.xlsx
  * bộ B  — hai kỳ 20260430 và 20260531 (KHÔNG dùng tệp 20260630: mọi phiên bản
            của tệp này đều là dữ liệu mô phỏng, xem generate_synthetic_period.py
            trong lịch sử git).

Các phần (chạy độc lập, có thể song song bằng nhiều tiến trình):

  A   Bộ A: 6 mô hình + ma trận nhầm lẫn; Gain/SHAP trên cùng XGBoost 11 biến;
      4 chiến lược mất cân bằng; Top-k (13 biến); loại DAYS_TO_MATURITY; lưới
      65 cấu hình; One-Hot; SMOTE-NC; độ nhạy theo ngày tham chiếu d0.
  B   Bộ B (bài toán 5 nhóm): rerun_iceir.task1 (6 mô hình, theo nhóm, ma trận
      nhầm lẫn, Gain/SHAP, Top-k, loại biến lãi, 4 chiến lược, SMOTE-NC, mã chi
      nhánh, IV, PSI); lưới 60 cấu hình cho cả 6 mô hình; điền khuyết biến cơ
      cấu; tách tập ngoài thời gian theo tập con (review_iceir.m1).
  I   IV (hàm đã sửa cách chia khoảng) toàn mẫu và chỉ tập huấn luyện, hai bộ.
  T   Chuyển nhóm cặp 04->05: rerun_iceir.task2 (CV 5x10, kiểm định t hiệu chỉnh
      + Holm, Recall@k, AP theo loại, SHAP phân tầng, loại biến lãi) và ma trận
      chuyển nhóm / khoản vay bị loại.
  S   Spearman kèm p tiệm cận VÀ p hoán vị chính xác cho mọi phép đối chiếu
      thứ hạng nhóm (chạy sau A, B).

Chạy:  cd code && python3 -m src.rerun_thesis_final A|B|I|T|S
Output: code/results/thesis_final/
"""
import itertools
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, TARGET_COL, RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN,
    DATA_CREDIT_INFO_TEST, CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY,
    TRANSITION_ID_COL,
)
import src.preprocessing as prepmod
from src.models import build_pipeline
from src.credit_info_preprocessing import load_credit_info, build_transition_dataset
import src.rerun_iceir as ice
import src.review_iceir as rev
import src.generate_thesis_revision as gtr
import src.rerun_review_fixes as rrf

OUT = Path(__file__).parent.parent / "results" / "thesis_final"
OUT.mkdir(parents=True, exist_ok=True)
# Chuyển hướng mọi tệp kết quả của các module dùng lại sang thư mục mới.
ice.OUT = rev.OUT = gtr.RES = rrf.RES = OUT

CAT_A, NUM_A = gtr.CAT_A, gtr.NUM_A
MODELS = gtr.MODELS


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── Lưới siêu tham số bộ A (65 cấu hình, Bảng P5.1) ─────────────────────────
GRID_A = {
    "Logistic Regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "Decision Tree": [{"max_depth": d, "min_samples_leaf": l} for d in (5, 10, 15, 20) for l in (5, 10, 20)],
    "Random Forest": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
                      for n in (100, 300, 500) for d in (10, 15, 20) for l in (5, 10)],
    "XGBoost": [{"max_depth": d, "learning_rate": r, "n_estimators": 400} for d in (4, 6, 8) for r in (0.03, 0.05, 0.1)]
               + [{"n_estimators": n} for n in (200, 600)],
    "LightGBM": [{"max_depth": d, "learning_rate": r, "n_estimators": 400} for d in (4, 6, 8) for r in (0.03, 0.05, 0.1)]
                + [{"n_estimators": n} for n in (200, 600)],
    "CatBoost": [{"depth": d, "learning_rate": r, "iterations": 400} for d in (4, 6, 8) for r in (0.03, 0.05, 0.1)],
}


def _f1(p, X, y):
    return rrf.evaluate(p, X, y)[0]["f1_macro"]


def grid_rows(bo, grid, fit_eval):
    rows = []
    for name, cfgs in grid.items():
        t0 = time.time()
        default = fit_eval(name, None)
        s = np.array([fit_eval(name, prm) for prm in cfgs])
        rows.append({"bo": bo, "model": name, "n_cau_hinh": len(cfgs), "f1_mac_dinh": default,
                     "f1_tot_nhat": s.max(), "cai_thien_pct": (s.max() / default - 1) * 100,
                     "f1_nho_nhat": s.min(), "bien_do": s.max() - s.min(), "do_lech_chuan": s.std(ddof=0)})
        log(f"lưới {bo} {name}: mặc định {default:.4f} tốt nhất {s.max():.4f} biên độ {s.max()-s.min():.4f} "
            f"({time.time()-t0:.0f}s)")
        pd.DataFrame(rows).to_csv(OUT / f"grid_{bo}.csv", index=False, encoding="utf-8-sig")


# ── A ────────────────────────────────────────────────────────────────────────
def part_a():
    Xtr, Xte, ytr, yte = gtr.load_a()
    gtr.part_a(Xtr, Xte, ytr, yte)          # revision_a_models/confusion/gain_vs_shap/imbalance/topk
    gtr.part_leak_a(Xtr, Xte, ytr, yte)     # revision_a_dtm_ablation, revision_a_matured_share
    rrf.part_s3()                           # rev2_s3_onehot
    # SMOTE-NC bộ A (phần A của rrf.part_s2, tách riêng để không chạy lại phần B)
    rows = []
    for label, strat in [("SMOTE", "smote_moderate"), ("SMOTE-NC", "smote_nc")]:
        p = build_pipeline("xgboost", CAT_A, NUM_A, random_state=RANDOM_STATE, imbalance_strategy=strat)
        p.fit(Xtr[CAT_A + NUM_A], ytr)
        rows.append({"chien_luoc": label, **rrf.evaluate(p, Xte[CAT_A + NUM_A], yte)[0]})
    pd.DataFrame(rows).to_csv(OUT / "a_smotenc.csv", index=False, encoding="utf-8-sig")
    log(f"A SMOTE-NC: {[round(r['f1_macro'], 4) for r in rows]}")

    # Lưới 65 cấu hình
    def fe(name, prm):
        p = build_pipeline(MODELS[name], CAT_A, NUM_A, random_state=RANDOM_STATE,
                           imbalance_strategy="smote_moderate", model_params=prm)
        return _f1(p.fit(Xtr[CAT_A + NUM_A], ytr), Xte[CAT_A + NUM_A], yte)
    grid_rows("A", GRID_A, fe)

    # Độ nhạy theo ngày tham chiếu d0
    raw = pd.read_excel(DATA_RAW)
    rows = []
    for d0 in ["2020-12-31", "2021-12-31", "2023-12-31"]:
        prepmod.REFERENCE_DATE = pd.Timestamp(d0)
        df = prepmod.engineer_features(prepmod.parse_dates(raw))
        yy = df[TARGET_COL].astype(int).values - 1
        a, b, ya, yb = train_test_split(df[CAT_A + NUM_A], yy, test_size=TEST_SIZE, stratify=yy,
                                        random_state=RANDOM_STATE)
        for name in ["XGBoost", "Logistic Regression"]:
            p = build_pipeline(MODELS[name], CAT_A, NUM_A, random_state=RANDOM_STATE, imbalance_strategy="smote_moderate")
            rows.append({"d0": d0, "model": name, "f1_macro": _f1(p.fit(a, ya), b, yb)})
    prepmod.REFERENCE_DATE = pd.Timestamp("2021-12-31")
    pd.DataFrame(rows).to_csv(OUT / "a_d0_sensitivity.csv", index=False, encoding="utf-8-sig")
    log(f"A d0: {rows}")


# ── B ────────────────────────────────────────────────────────────────────────
def part_b():
    ice.task1()                                 # t1_*
    gtr.part_b2_b3(*gtr.load_b())               # revision_b_offbalance_leakage, revision_b_restructure_impute
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    rev.m1(a, b)                                # m1_oot_subsets
    X, y, _Xh, _yh, num_p, cat_p = rrf.load_b()
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    def fe(name, prm):
        p = build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE, imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY, model_params=prm)
        return _f1(p.fit(Xtr, ytr), Xva, yva)
    grid_rows("B", rrf.GRID_B, fe)


# ── I ────────────────────────────────────────────────────────────────────────
def part_i():
    rrf.part_i()                                # rev2_iv_train_only (hàm IV đã sửa)


# ── T ────────────────────────────────────────────────────────────────────────
def part_t():
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    dm = build_transition_dataset(a, b, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL)
    from src.credit_info_preprocessing import TRANSITION_GRP_T1_COL
    pd.crosstab(dm[CREDIT_INFO_TARGET_COL].astype(int), dm[TRANSITION_GRP_T1_COL].astype(int)).to_csv(
        OUT / "t2_matrix.csv", encoding="utf-8-sig")
    kept = set(dm[TRANSITION_ID_COL])
    ex = a[~a[TRANSITION_ID_COL].isin(kept)][CREDIT_INFO_TARGET_COL].astype(int).value_counts().sort_index()
    allg = a[CREDIT_INFO_TARGET_COL].astype(int).value_counts().sort_index()
    pd.DataFrame({"bi_loai": ex, "tong": allg, "ty_le_pct": ex / allg * 100}).to_csv(
        OUT / "t2_exclusion.csv", encoding="utf-8-sig")
    new = (~b[TRANSITION_ID_COL].isin(set(a[TRANSITION_ID_COL]))).sum()
    log(f"T: khớp {len(dm)}, bị loại {int(ex.sum())}, khoản vay mới ở kỳ 05: {int(new)}")
    ice.task2()                                 # t2_*


# ── X: kiểm tra phụ + hình ───────────────────────────────────────────────────
def part_x():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.model_selection import RepeatedStratifiedKFold
    from src.credit_info_preprocessing import (build_raw_feature_pool, engineer_business_features,
                                               prepare_credit_info, TRANSITION_LABEL_COL)
    a, b = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)
    rows = []
    for lab, d in [("20260430", a), ("20260531", b)]:
        de, _ = engineer_business_features(d)
        rows.append({"ky": lab, "n": len(d), "tuoi_khoan_vay_trung_vi": float(de["ENG_LOAN_AGE_DAYS"].median()),
                     "thieu_eng_pct": float(de[["ENG_LOAN_AGE_DAYS", "ENG_DAYS_TO_MATURITY",
                                                "ENG_TENOR_ACTUAL_DAYS"]].isna().mean().max() * 100)})
    pd.DataFrame(rows).to_csv(OUT / "x_loan_age.csv", index=False, encoding="utf-8-sig")
    log(f"X tuổi khoản vay: {rows}")

    dm = build_transition_dataset(a, b, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL)
    nr, cr = build_raw_feature_pool(dm); de, eng = engineer_business_features(dm)
    num, cat = nr + eng, cr
    X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num, cat, label_offset=0)

    # Hình PR + Gain từng biến của XGBoost (chia 80/20)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    p = build_pipeline("xgboost", cat, num, random_state=RANDOM_STATE, imbalance_strategy="none",
                       eval_metric="logloss").fit(Xtr, ytr)
    s = p.predict_proba(Xva)[:, 1]
    pr, rc, _ = precision_recall_curve(yva, s); ap = average_precision_score(yva, s)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rc, pr, lw=2, label=f"XGBoost (AP = {ap:.4f})")
    ax.axhline(yva.mean(), ls="--", c="grey", label=f"Ngẫu nhiên (AP = {yva.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right"); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(OUT / "eda_transition_pr_curve.png", dpi=200); plt.close(fig)
    names = p.named_steps["preprocessor"].get_feature_names()
    g = p.named_steps["classifier"].get_booster().get_score(importance_type="total_gain")
    pd.DataFrame({"feature": names, "gain": [g.get(f"f{i}", 0.0) for i in range(len(names))]}).to_csv(
        OUT / "x_transition_gain.csv", index=False, encoding="utf-8-sig")
    log(f"X PR: AP {ap:.4f}, dự báo dương tại 0,5: {int((s >= .5).sum())}, đúng {int(((s >= .5) & (yva == 1)).sum())}")

    # Độ nhạy: hợp đồng rời danh mục từ nhóm >= 2 coi là sự kiện (như review_iceir M6)
    exits = a[~a[TRANSITION_ID_COL].isin(b[TRANSITION_ID_COL])].copy()
    de_e, _ = engineer_business_features(exits)
    Xe = de_e[num + cat]; ye = (de_e[CREDIT_INFO_TARGET_COL].astype(int) >= 2).astype(int).values
    Xa = pd.concat([X, Xe]); ya = np.r_[y, ye]
    aps = []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE).split(Xa, ya):
        q = build_pipeline("xgboost", cat, num, random_state=RANDOM_STATE, imbalance_strategy="none", eval_metric="logloss")
        q.fit(Xa.iloc[tr], ya[tr]); aps.append(average_precision_score(ya[te], q.predict_proba(Xa.iloc[te])[:, 1]))
    m6 = {"n": len(ya), "events": int(ya.sum()), "added_exit_events": int(ye.sum()), "base_rate": ya.mean(),
          "ap_mean": float(np.mean(aps)), "ap_std": float(np.std(aps, ddof=1))}
    pd.DataFrame([m6]).to_csv(OUT / "x_exits_sensitivity.csv", index=False, encoding="utf-8-sig")
    log(f"X M6: {m6}")


# ── R: điền trung vị so với điền 0 cho ba biến cơ cấu nợ ─────────────────────
def part_r():
    """gtr.part_b2_b3 không còn so sánh được vì engineer_business_features đã điền 0
    từ trước; ở đây trả các ô vốn thiếu trong dữ liệu thô về NaN cho biến thể
    "trung vị", để pipeline điền trung vị phần quan sát được như cách làm cũ."""
    from src.credit_info_preprocessing import RESTRUCTURE_COLS
    X, y, Xh, yh, num_p, cat_p = rrf.load_b()
    raw_t, raw_h = load_credit_info(DATA_CREDIT_INFO_TRAIN), load_credit_info(DATA_CREDIT_INFO_TEST)

    def to_median(X_, raw):
        X_ = X_.copy()
        for c in RESTRUCTURE_COLS:
            X_.loc[raw.loc[X_.index, c].isna().values, c] = np.nan
        return X_
    Xm, Xhm = to_median(X, raw_t), to_median(Xh, raw_h)
    log(f"R: tỷ lệ thiếu sau khi trả về NaN: {Xm[RESTRUCTURE_COLS].isna().mean().round(4).to_dict()}")
    rows = []
    for variant, XX, XH in [("trung vị", Xm, Xhm), ("điền 0", X, Xh)]:
        Xtr, Xva, ytr, yva = train_test_split(XX, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
        for name in ["LightGBM", "XGBoost", "Random Forest"]:
            p = build_pipeline(MODELS[name], cat_p, num_p, random_state=RANDOM_STATE, imbalance_strategy="custom",
                               custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY).fit(Xtr, ytr)
            rows.append({"variant": variant, "model": name, "validation": _f1(p, Xva, yva), "holdout": _f1(p, XH, yh)})
            log(f"R {variant} {name}: {rows[-1]}")
    pd.DataFrame(rows).to_csv(OUT / "r_restructure_impute.csv", index=False, encoding="utf-8-sig")


# ── S ────────────────────────────────────────────────────────────────────────
def spearman_exact(x, y):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rho = stats.spearmanr(rx, ry).statistic
    perms = np.array([stats.spearmanr(rx, np.array(p)).statistic for p in itertools.permutations(ry)])
    return rho, stats.spearmanr(x, y).pvalue, float(np.mean(np.abs(perms) >= abs(rho) - 1e-12))


def part_s():
    ga = pd.read_csv(OUT / "revision_a_gain_vs_shap_group.csv", index_col=0)
    ga = ga.drop(index=[i for i in ga.index if "Nhân khẩu" in str(i)], errors="ignore")
    gb = pd.read_csv(OUT / "t1_gain_shap_group.csv").set_index("group")
    rows = []
    for lab, g in [("A Gain vs SHAP (nhóm)", ga), ("B Gain vs SHAP (nhóm)", gb)]:
        rho, pa, pe = spearman_exact(g["gain_pct"].values, g["shap_pct"].values)
        rows.append({"phep_so": lab, "n": len(g), "rho": rho, "p_tiem_can": pa, "p_hoan_vi": pe})
    mp = {"Kỳ hạn khoản vay": "Đặc trưng kỹ thuật (engineered)", "Dư nợ và hạn mức sử dụng": "Dư nợ & Thanh toán",
          "Đơn vị/Chi nhánh": "Chi nhánh", "Lãi suất": "Lãi suất", "Sản phẩm và mục đích vay": "Hình thức & Mục đích vay"}
    xa = [ga.loc[k, "shap_pct"] for k in mp]; xb = [gb.loc[v, "shap_pct"] for v in mp.values()]
    rho, pa, pe = spearman_exact(xa, xb)
    rows.append({"phep_so": "SHAP nhóm A vs B", "n": 5, "rho": rho, "p_tiem_can": pa, "p_hoan_vi": pe})
    pd.DataFrame(rows).to_csv(OUT / "spearman_all.csv", index=False, encoding="utf-8-sig")
    log(f"S:\n{pd.DataFrame(rows)}")


if __name__ == "__main__":
    want = [x.upper() for x in sys.argv[1:]] or ["A", "B", "I", "T", "S"]
    for k, f in [("A", part_a), ("B", part_b), ("I", part_i), ("T", part_t), ("X", part_x), ("R", part_r), ("S", part_s)]:
        if k in want:
            log(f"=== Phần {k} ===")
            f()
    log("HOÀN TẤT")
