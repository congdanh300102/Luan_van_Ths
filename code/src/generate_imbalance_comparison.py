"""Script bổ sung cho luận văn — so sánh chiến lược KẾT HỢP SMOTE + trọng số
lớp với các chiến lược đơn lẻ (không xử lý / chỉ SMOTE / chỉ class weight),
trên mô hình tham chiếu của bài toán phân loại năm nhóm nợ TẠI MỘT THỜI ĐIỂM
(bộ A và bộ B) — khác với bài toán dự báo chuyển nhóm nợ ở
generate_transition_report.py, vốn đã có sẵn so sánh 4 chiến lược riêng.

Mô hình tham chiếu:
  - Bộ A: XGBoost — cấu hình mặc định của mô-đun phân tích đặc trưng (Mục
    sec:feature-a-eda Chương 3 / cumulative-importance Chương 4).
  - Bộ B: LightGBM — mô hình đạt Macro-F1 cao nhất trên tập kiểm định, được
    chọn cho các phân tích đặc trưng/SHAP tiếp theo (Mục 4.4 Chương 4).

Không phải trang Streamlit — chạy độc lập:

    cd code && py -m src.generate_imbalance_comparison

Output:
  code/results/imbalance_comparison_a.csv
  code/results/imbalance_comparison_b.csv
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
    DATA_RAW, TARGET_COL, DROP_COLS, CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE,
    DATA_CREDIT_INFO_TRAIN, CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY,
)
from src.preprocessing import parse_dates, engineer_features, clean
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info,
)
from src.imbalance_benchmark import benchmark_strategies

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Bốn chiến lược so sánh — nhãn hiển thị giống nhau cho cả hai bộ để dễ đối
# chiếu chéo trong luận văn; khoá "custom"/"smote_class_weight" của bộ B dùng
# CREDIT_INFO_SMOTE_STRATEGY (truyền riêng), không dùng mốc _SMOTE_MODERATE
# calibrate cho bộ A.
STRATEGIES_A = {
    "Không xử lý":                    "none",
    "Class weight":                   "class_weight",
    "SMOTE":                          "smote_moderate",
    "SMOTE + Class weight (kết hợp)": "smote_class_weight",
}
STRATEGIES_B = {
    "Không xử lý":                    "none",
    "Class weight":                   "class_weight",
    "SMOTE":                          "custom",
    "SMOTE + Class weight (kết hợp)": "smote_class_weight",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # ── Bộ A ─────────────────────────────────────────────────────────────────
    log("Bộ A — đang tải và tiền xử lý…")
    df_a_raw = pd.read_excel(DATA_RAW)
    df_a = parse_dates(df_a_raw)
    df_a = engineer_features(df_a)
    y_a = df_a[TARGET_COL].astype(int).values - 1
    X_a = clean(df_a, drop_cols=DROP_COLS + [TARGET_COL])
    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(
        X_a, y_a, test_size=TEST_SIZE, stratify=y_a, random_state=RANDOM_STATE,
    )
    log(f"Bộ A: train={X_a_train.shape}, test={X_a_test.shape}")

    t0 = time.time()
    df_cmp_a = benchmark_strategies(
        "xgboost", CATEGORICAL_COLS, NUMERICAL_COLS,
        X_a_train, y_a_train, X_a_test, y_a_test,
        STRATEGIES_A, random_state=RANDOM_STATE,
    )
    df_cmp_a.insert(0, "bo_du_lieu", "A")
    df_cmp_a.insert(1, "mo_hinh", "XGBoost")
    df_cmp_a.to_csv(RESULTS_DIR / "imbalance_comparison_a.csv", index=False, encoding="utf-8-sig")
    log(f"Bộ A — so sánh chiến lược ({time.time()-t0:.1f}s):\n{df_cmp_a}")

    # ── Bộ B ─────────────────────────────────────────────────────────────────
    log("Bộ B — đang tải và tiền xử lý…")
    df_b_raw = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    num_raw, cat_raw = build_raw_feature_pool(df_b_raw)
    df_b_eng, eng_cols = engineer_business_features(df_b_raw)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    X_b_all, y_b_all = prepare_credit_info(
        df_b_eng, CREDIT_INFO_TARGET_COL, num_p, cat_p, label_offset=1,
    )
    X_b_train, X_b_test, y_b_train, y_b_test = train_test_split(
        X_b_all, y_b_all, test_size=TEST_SIZE, stratify=y_b_all, random_state=RANDOM_STATE,
    )
    log(f"Bộ B: train={X_b_train.shape}, test={X_b_test.shape}")

    t0 = time.time()
    df_cmp_b = benchmark_strategies(
        "lightgbm", cat_p, num_p,
        X_b_train, y_b_train, X_b_test, y_b_test,
        STRATEGIES_B, random_state=RANDOM_STATE,
        custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY,
    )
    df_cmp_b.insert(0, "bo_du_lieu", "B")
    df_cmp_b.insert(1, "mo_hinh", "LightGBM")
    df_cmp_b.to_csv(RESULTS_DIR / "imbalance_comparison_b.csv", index=False, encoding="utf-8-sig")
    log(f"Bộ B — so sánh chiến lược ({time.time()-t0:.1f}s):\n{df_cmp_b}")

    log("HOÀN TẤT. Đã lưu code/results/imbalance_comparison_a.csv và imbalance_comparison_b.csv")


if __name__ == "__main__":
    main()
