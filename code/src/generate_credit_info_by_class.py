"""Bổ sung Precision/Recall/F1 theo từng nhóm nợ cho **cả 6 mô hình** của bộ B.

`generate_credit_info_report.py` chỉ lưu chi tiết theo nhóm cho mô hình tốt
nhất (credit_info_best_model_by_class_*.csv). Luận văn còn cần so sánh Recall
nhóm 3/4 giữa các mô hình, nên script này tái lập đúng phần huấn luyện cấu
hình mặc định của script đó (cùng split, cùng RANDOM_STATE, cùng chiến lược
mất cân bằng) và ghi thêm:

    code/results/credit_info_by_class_all_models_validation.csv
    code/results/credit_info_by_class_all_models_holdout.csv

Chạy:  cd code && python3 -m src.generate_credit_info_by_class
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, transform_credit_info,
)
from src.models import build_pipeline, available_models
from src.evaluation import compute_metrics

RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Đang tải dữ liệu train (20260430) và test (20260531)…")
    df_train_raw = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    df_test_raw = load_credit_info(DATA_CREDIT_INFO_TEST)

    num_raw, cat_raw = build_raw_feature_pool(df_train_raw)
    df_train_eng, eng_cols = engineer_business_features(df_train_raw)
    num_p, cat_p = num_raw + eng_cols, cat_raw

    X_all, y_all = prepare_credit_info(df_train_eng, CREDIT_INFO_TARGET_COL, num_p, cat_p)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE,
    )

    df_test_eng, _ = engineer_business_features(df_test_raw)
    df_test_valid = df_test_eng.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()
    X_holdout = transform_credit_info(df_test_valid, num_p, cat_p)
    y_holdout = df_test_valid[CREDIT_INFO_TARGET_COL].astype(int).values - 1
    log(f"Train {X_train.shape} | Validation {X_val.shape} | Kỳ sau {X_holdout.shape}")

    rows_v, rows_h = [], []
    for label, key in available_models().items():
        t0 = time.time()
        pipe = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                              imbalance_strategy="custom",
                              custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        pipe.fit(X_train, y_train)

        for tag, X, y, sink in [("validation", X_val, y_val, rows_v),
                                ("holdout", X_holdout, y_holdout, rows_h)]:
            m = compute_metrics(y + 1, pipe.predict(X) + 1, pipe.predict_proba(X))
            rep = m["report"]
            for cls in [k for k in rep if k.startswith("Nhóm")]:
                sink.append({
                    "model": label, "model_key": key, "nhom": cls,
                    "support": int(rep[cls]["support"]),
                    "precision": rep[cls]["precision"],
                    "recall": rep[cls]["recall"],
                    "f1": rep[cls]["f1-score"],
                })
            sink.append({
                "model": label, "model_key": key, "nhom": "accuracy",
                "support": int(sum(rep[k]["support"] for k in rep if k.startswith("Nhóm"))),
                "precision": float("nan"), "recall": float("nan"),
                "f1": float(rep["accuracy"]),
            })
        log(f"  {label} xong ({time.time()-t0:.1f}s)")

    for tag, rows in [("validation", rows_v), ("holdout", rows_h)]:
        out = RESULTS_DIR / f"credit_info_by_class_all_models_{tag}.csv"
        pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
        log(f"Đã lưu {out.name}")

    log("HOÀN TẤT.")


if __name__ == "__main__":
    main()
