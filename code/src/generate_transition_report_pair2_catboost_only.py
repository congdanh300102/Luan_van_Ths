"""Chạy lại RIÊNG 2 cấu hình CatBoost (none, class_weight) của cặp kỳ 2
(20260531 -> 20260630) sau khi sửa lỗi Windows file-locking
(allow_writing_files=False trong src/models.py). 20 cấu hình còn lại (5 mô
hình x 4 chiến lược) đã chạy xong ổn định ở hai lần chạy trước (kết quả trùng
khớp), không cần chạy lại — tránh tốn thêm ~45-60 phút.

Ghi CSV riêng (transition_pair2_catboost.csv); một script khác sẽ gộp với
kết quả 20 cấu hình đã có (log) để tạo transition_pair2_cv_summary.csv đầy đủ.
"""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.config import (
    RANDOM_STATE, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2,
    CREDIT_INFO_TARGET_COL, TRANSITION_ID_COL, TRANSITION_SMOTE_STRATEGY,
    TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL,
)
from src.models import build_pipeline
from src.evaluation import repeated_stratified_binary_eval

RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_kwargs(strategy: str) -> dict:
    kwargs = {"random_state": RANDOM_STATE, "imbalance_strategy": strategy, "loss_function": "Logloss"}
    return kwargs


def main():
    log("Đang tải dữ liệu kỳ T (20260531) và T+1 (20260630, THẬT)…")
    df_t_raw = load_credit_info(DATA_CREDIT_INFO_TEST)
    df_t1_raw = load_credit_info(DATA_CREDIT_INFO_TEST2)
    df_merged = build_transition_dataset(
        df_t_raw, df_t1_raw, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    num_raw, cat_raw = build_raw_feature_pool(df_merged)
    df_eng, eng_cols = engineer_business_features(df_merged)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    X_all, y_all = prepare_credit_info(df_eng, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
    log(f"X_all: {X_all.shape}, positive rate: {y_all.mean() * 100:.3f}%")

    rows, per_repeat_frames = [], []
    for strategy in ["none", "class_weight"]:
        t0 = time.time()
        pipe = build_pipeline("catboost", cat_p, num_p, **_build_kwargs(strategy))
        per_repeat_df, summary = repeated_stratified_binary_eval(
            pipe, X_all, y_all,
            n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
            random_state=RANDOM_STATE, sample_weight_fn=None,
        )
        per_repeat_df["model"] = "CatBoost"
        per_repeat_df["model_key"] = "catboost"
        per_repeat_df["strategy"] = strategy
        per_repeat_frames.append(per_repeat_df)
        row = {"model": "CatBoost", "model_key": "catboost", "strategy": strategy, **summary}
        rows.append(row)
        log(f"  [{strategy}] CatBoost: PR-AUC={summary['pr_auc_mean']:.4f}"
            f"±{summary['pr_auc_std']:.4f} | ROC-AUC={summary['roc_auc_mean']:.4f}"
            f" | Recall(+)={summary['recall_pos_mean']:.3f} ({time.time()-t0:.1f}s)")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "transition_pair2_catboost.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_repeat_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "transition_pair2_catboost_per_repeat.csv", index=False, encoding="utf-8-sig")
    log("HOÀN TẤT CatBoost cặp 2.")


if __name__ == "__main__":
    main()
