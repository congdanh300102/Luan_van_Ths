"""Ma trận nhầm lẫn cho bộ B — phục vụ phân tích đánh đổi giữa các nhóm nợ.

Precision/Recall theo từng lớp không cho biết hồ sơ bị phân loại sai đi **vào
nhóm nào**. Với bài toán nợ xấu, điều đó quan trọng: một hồ sơ nhóm 4 bị đoán
thành nhóm 3 vẫn được gắn cờ nợ xấu (nhóm 3-5 đều là NPL theo Thông tư 11),
còn bị đoán thành nhóm 1 thì bị bỏ sót hoàn toàn.

Script tái lập đúng phần huấn luyện cấu hình mặc định của
generate_credit_info_report.py và ghi ma trận nhầm lẫn cho cả 6 mô hình:

    code/results/credit_info_confusion_<model_key>_<tang>.csv

Chạy:  cd code && python3 -m src.generate_credit_info_confusion
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
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

RESULTS_DIR = Path(__file__).parent.parent / "results"
LABELS = [1, 2, 3, 4, 5]
NAMES = [f"Nhóm {i}" for i in LABELS]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Đang tải dữ liệu…")
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
    log(f"Validation {X_val.shape} | Kỳ sau {X_holdout.shape}")

    for label, key in available_models().items():
        t0 = time.time()
        pipe = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                              imbalance_strategy="custom",
                              custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        pipe.fit(X_train, y_train)

        for tag, X, y in [("validation", X_val, y_val), ("holdout", X_holdout, y_holdout)]:
            cm = confusion_matrix(y + 1, pipe.predict(X) + 1, labels=LABELS)
            df = pd.DataFrame(cm, index=[f"Thực tế {n}" for n in NAMES],
                              columns=[f"Dự báo {n}" for n in NAMES])
            out = RESULTS_DIR / f"credit_info_confusion_{key}_{tag}.csv"
            df.to_csv(out, encoding="utf-8-sig")
        log(f"  {label} xong ({time.time()-t0:.1f}s)")

    log("HOÀN TẤT.")


if __name__ == "__main__":
    main()
