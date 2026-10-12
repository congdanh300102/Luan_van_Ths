"""Chạy lại các phân tích bị ảnh hưởng bởi lỗi phân tích cú pháp ngày ở hai kỳ
20260531 và 20260630 (xem _parse_yyyymmdd trong credit_info_preprocessing.py).

Phạm vi ảnh hưởng của lỗi: mọi phân tích lấy ĐẶC TRƯNG từ kỳ 20260531 hoặc
20260630 đều có ba biến ENG_LOAN_AGE_DAYS, ENG_DAYS_TO_MATURITY và
ENG_TENOR_ACTUAL_DAYS rỗng 100%. Cụ thể:

  H   Đánh giá ngoài thời gian của bài toán 5 lớp trên kỳ 20260531.  (bị ảnh hưởng)
  T2  Toàn bộ cặp 2 của bài toán chuyển nhóm + chuyển giao cặp 1 -> cặp 2. (bị ảnh hưởng)

  Cặp 1 KHÔNG bị ảnh hưởng: toàn bộ đặc trưng lấy từ kỳ 20260430, kỳ 20260531
  chỉ đóng góp cột nhóm nợ dùng để tạo nhãn. Script vẫn chạy lại cặp 1 để xác
  nhận kết quả trùng khớp với bản đã báo cáo.

Chạy:  cd code && py -m src.rerun_after_datefix [H] [T]
Output: code/results/fixed_*.csv
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY, TRANSITION_ID_COL,
    TRANSITION_SMOTE_STRATEGY, TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
    RANDOM_STATE, TEST_SIZE,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL,
)
from src.models import build_pipeline, compute_sample_weights
from src.evaluation import repeated_stratified_binary_eval, recall_at_k

RES = Path(__file__).parent.parent / "results"
MODELS = {"Logistic Regression": "logistic", "Decision Tree": "decision_tree",
          "Random Forest": "random_forest", "XGBoost": "xgboost",
          "LightGBM": "lightgbm", "CatBoost": "catboost"}

# Cấu hình mô hình/chiến lược tốt nhất của cặp 1, dùng lại nguyên vẹn cho cặp 2
# để bảng độ ổn định so sánh đúng một cấu hình trên hai cặp.
PAIR1_BEST = {"XGBoost": "none", "LightGBM": "custom", "Random Forest": "none",
              "Decision Tree": "custom", "Logistic Regression": "none",
              "CatBoost": "none"}  # CatBoost cuối vì chậm hơn hẳn (~80s/lần khớp)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ordinal_mae_from_cm(cm):
    cm = np.asarray(cm, dtype=float)
    k = cm.shape[0]
    d = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :])
    return float((cm * d).sum() / cm.sum())


# ══════════════════════════════════════════════════════════════════════════════
# H — bài toán 5 lớp: tập kiểm định và kỳ sau 20260531
# ══════════════════════════════════════════════════════════════════════════════
def part_h():
    log("H: tải kỳ 20260430 và 20260531…")
    df_tr = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    num_raw, cat_raw = build_raw_feature_pool(df_tr)
    df_tr_e, eng = engineer_business_features(df_tr)
    num_p, cat_p = num_raw + eng, cat_raw
    X, y = prepare_credit_info(df_tr_e, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)

    df_ho_e, _ = engineer_business_features(load_credit_info(DATA_CREDIT_INFO_TEST))
    Xho, yho = prepare_credit_info(df_ho_e, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1)
    log(f"H: ENG thiếu ở kỳ sau sau khi sửa: "
        f"{ {c: round(Xho[c].isna().mean()*100, 2) for c in eng} }")

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                          random_state=RANDOM_STATE)
    rows, byclass, cms = [], [], []
    for name, key in MODELS.items():
        t0 = time.time()
        p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(Xtr, ytr)
        for split, XX, yy in [("validation", Xva, yva), ("holdout", Xho, yho)]:
            pred = np.asarray(p.predict(XX)).ravel().astype(int)
            proba = p.predict_proba(XX)
            cm = confusion_matrix(yy, pred, labels=[0, 1, 2, 3, 4])
            pr, rc, f1, sup = precision_recall_fscore_support(
                yy, pred, labels=[0, 1, 2, 3, 4], zero_division=0)
            rows.append({
                "model": name, "split": split,
                "accuracy": (pred == yy).mean(),
                "f1_macro": f1_score(yy, pred, average="macro"),
                "f1_weighted": f1_score(yy, pred, average="weighted"),
                "roc_auc_macro": roc_auc_score(yy, proba, multi_class="ovr", average="macro"),
                "balanced_accuracy": rc.mean(),
                "ordinal_mae": ordinal_mae_from_cm(cm),
            })
            for i in range(5):
                byclass.append({"model": name, "split": split, "nhom": i + 1,
                                "support": int(sup[i]), "precision": pr[i],
                                "recall": rc[i], "f1": f1[i]})
            cms.append(pd.DataFrame(cm, index=[f"true_{i}" for i in range(1, 6)],
                                    columns=[f"pred_{i}" for i in range(1, 6)])
                       .assign(model=name, split=split))
        log(f"H  {name}: val F1m={rows[-2]['f1_macro']:.4f} | "
            f"kỳ sau F1m={rows[-1]['f1_macro']:.4f} ({time.time()-t0:.0f}s)")

    pd.DataFrame(rows).to_csv(RES / "fixed_b_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(byclass).to_csv(RES / "fixed_b_by_class.csv", index=False, encoding="utf-8-sig")
    pd.concat(cms).to_csv(RES / "fixed_b_confusion.csv", encoding="utf-8-sig")
    log("H: đã lưu fixed_b_*.csv")


# ══════════════════════════════════════════════════════════════════════════════
# T — bài toán chuyển nhóm: cặp 1 (xác nhận) và cặp 2 (chạy lại)
# ══════════════════════════════════════════════════════════════════════════════
def _kwargs(strategy, key):
    kw = {"random_state": RANDOM_STATE, "imbalance_strategy": strategy}
    if strategy in ("custom", "smote_class_weight"):
        kw["custom_smote_strategy"] = TRANSITION_SMOTE_STRATEGY
    if key == "catboost":
        kw["loss_function"] = "Logloss"
    if key == "xgboost":
        kw["eval_metric"] = "logloss"
    return kw


def _sw(strategy, key):
    return compute_sample_weights if (strategy == "class_weight" and key == "xgboost") else None


def build_pair(df_t, df_t1, name):
    dm = build_transition_dataset(df_t, df_t1, id_col=TRANSITION_ID_COL,
                                  target_col=CREDIT_INFO_TARGET_COL)
    num_raw, cat_raw = build_raw_feature_pool(dm)
    de, eng = engineer_business_features(dm)
    num_p, cat_p = num_raw + eng, cat_raw
    X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
    miss = {c: round(X[c].isna().mean() * 100, 2) for c in eng}
    log(f"{name}: {len(dm):,} khớp, {int(y.sum())} sự kiện ({y.mean()*100:.3f}%), ENG thiếu {miss}")
    return dm, X, y, num_p, cat_p


def part_t():
    raws = {k: load_credit_info(f) for k, f in
            [("20260430", DATA_CREDIT_INFO_TRAIN), ("20260531", DATA_CREDIT_INFO_TEST),
             ("20260630", DATA_CREDIT_INFO_TEST2)]}
    # Cặp 2 chạy trước: đây là cặp bị lỗi ngày làm hỏng, cũng là cặp quyết định
    # kết luận về độ ổn định. Cặp 1 chạy sau chỉ để xác nhận không đổi.
    pairs = {"cap2": build_pair(raws["20260531"], raws["20260630"], "cặp 2"),
             "cap1": build_pair(raws["20260430"], raws["20260531"], "cặp 1")}

    # T1 — kiểm định chéo 5x10 cho 6 cấu hình trên từng cặp
    rows, per_repeat = [], []
    for pname, (dm, X, y, num_p, cat_p) in pairs.items():
        for mname, strategy in PAIR1_BEST.items():
            key = MODELS[mname]
            t0 = time.time()
            pipe = build_pipeline(key, cat_p, num_p, **_kwargs(strategy, key))
            pr, summary = repeated_stratified_binary_eval(
                pipe, X, y, n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
                random_state=RANDOM_STATE, sample_weight_fn=_sw(strategy, key))
            pr["cap"], pr["model"], pr["strategy"] = pname, mname, strategy
            per_repeat.append(pr)
            rows.append({"cap": pname, "model": mname, "strategy": strategy, **summary})
            log(f"T {pname} {mname} ({strategy}): AP={summary['pr_auc_mean']:.4f}"
                f"±{summary['pr_auc_std']:.4f} ROC-AUC={summary['roc_auc_mean']:.4f}"
                f" ({time.time()-t0:.0f}s)")
            pd.DataFrame(rows).to_csv(RES / "fixed_transition_cv.csv", index=False,
                                      encoding="utf-8-sig")
    pd.concat(per_repeat, ignore_index=True).to_csv(
        RES / "fixed_transition_cv_per_repeat.csv", index=False, encoding="utf-8-sig")

    # T2 — chuyển giao hai chiều giữa hai cặp
    rows = []
    for tr, te in [("cap1", "cap2"), ("cap2", "cap1")]:
        _d, Xtr, ytr, num_p, cat_p = pairs[tr]
        _d2, Xte, yte, *_r = pairs[te]
        p = build_pipeline("xgboost", cat_p, num_p, **_kwargs("none", "xgboost"))
        p.fit(Xtr, ytr)
        s = p.predict_proba(Xte[Xtr.columns])[:, 1]
        pred = (s >= 0.5).astype(int)
        tp = int(((pred == 1) & (yte == 1)).sum())
        rows.append({"huan_luyen": tr, "cham_diem": te, "n_test": len(yte),
                     "n_pos": int(yte.sum()), "ap": average_precision_score(yte, s),
                     "ap_ngau_nhien": float(yte.mean()), "roc_auc": roc_auc_score(yte, s),
                     "recall_pos_nguong_0_5": tp / max(int(yte.sum()), 1),
                     "precision_pos_nguong_0_5": tp / max(int(pred.sum()), 1)})
        log(f"T chuyển giao {tr}->{te}: AP={rows[-1]['ap']:.4f} "
            f"(ngẫu nhiên {rows[-1]['ap_ngau_nhien']:.4f}) ROC-AUC={rows[-1]['roc_auc']:.4f}")
    pd.DataFrame(rows).to_csv(RES / "fixed_transition_transfer.csv", index=False,
                              encoding="utf-8-sig")

    # T3 — Recall@k trên một lần chia 80/20 minh họa, mỗi cặp dùng XGBoost none
    rows = []
    for pname, (dm, X, y, num_p, cat_p) in pairs.items():
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=RANDOM_STATE)
        p = build_pipeline("xgboost", cat_p, num_p, **_kwargs("none", "xgboost"))
        p.fit(Xtr, ytr)
        s = p.predict_proba(Xva)[:, 1]
        rk = recall_at_k(yva, s)
        rk["cap"] = pname
        rk["ap"] = average_precision_score(yva, s)
        rk["roc_auc"] = roc_auc_score(yva, s)
        rk["n_pos"] = int(yva.sum())
        rows.append(rk)
        log(f"T Recall@k {pname}: AP={rk['ap'].iloc[0]:.4f}, "
            f"ROC-AUC={rk['roc_auc'].iloc[0]:.4f}, n_pos={int(yva.sum())}\n{rk}")
    pd.concat(rows, ignore_index=True).to_csv(RES / "fixed_transition_recall_at_k.csv",
                                              index=False, encoding="utf-8-sig")
    log("T: đã lưu fixed_transition_*.csv")


def main():
    which = set(a.upper() for a in sys.argv[1:]) or {"H", "T"}
    if "H" in which:
        part_h()
    if "T" in which:
        part_t()
    log("HOÀN TẤT")


if __name__ == "__main__":
    main()
