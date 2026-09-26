"""Minh họa mở rộng bài toán dự báo chuyển nhóm nợ sang HAI cặp kỳ quan sát,
dùng cho Phụ lục của luận văn — KHÔNG phải bằng chứng chính (bằng chứng
chính vẫn là cặp kỳ THẬT duy nhất 20260430→20260531 ở generate_transition_
report.py, Chương 3-4).

Cặp 1 (20260430→20260531): THẬT — không chạy lại, lấy thẳng kết quả tốt
nhất đã có ở results/transition_cv_summary.csv.
Cặp 2 (20260531→20260630): kỳ 20260630 là MÔ PHỎNG (xem generate_synthetic_
period.py) — nhãn chuyển nhóm sinh ra chỉ phụ thuộc nhóm nợ hiện tại, không
phụ thuộc các đặc trưng còn lại. Vì vậy chênh lệch hiệu năng giữa 2 cặp (nếu
có) phản ánh GIỚI HẠN CỦA CƠ CHẾ MÔ PHỎNG, không phải bằng chứng mô hình bất
ổn theo thời gian — chỉ dùng để minh họa pipeline đã sẵn sàng tiếp nhận
thêm kỳ dữ liệu thật khi có.

Chạy: cd code && py -m src.generate_transition_multiperiod_demo
(cần chạy generate_synthetic_period.py và generate_transition_report.py trước)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.config import (
    RANDOM_STATE, DATA_CREDIT_INFO_TEST, CREDIT_INFO_TARGET_COL,
    TRANSITION_ID_COL, TRANSITION_SMOTE_STRATEGY,
    TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS, DATA_CREDIT_INFO_SIM,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, TRANSITION_LABEL_COL,
)
from src.models import build_pipeline, compute_sample_weights
from src.evaluation import repeated_stratified_binary_eval

RESULTS_DIR = Path(__file__).parent.parent / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_kwargs(strategy: str, model_key: str) -> dict:
    kwargs = {"random_state": RANDOM_STATE, "imbalance_strategy": strategy}
    if strategy in ("custom", "smote_class_weight"):
        kwargs["custom_smote_strategy"] = TRANSITION_SMOTE_STRATEGY
    if model_key == "catboost":
        kwargs["loss_function"] = "Logloss"
    if model_key == "xgboost":
        kwargs["eval_metric"] = "logloss"
    return kwargs


def _sample_weight_fn(strategy: str, model_key: str):
    if strategy == "class_weight" and model_key == "xgboost":
        return compute_sample_weights
    return None


def _eval_pair(df_t_raw, df_t1_raw, model_key, strategy, label):
    df_merged = build_transition_dataset(
        df_t_raw, df_t1_raw, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    num_raw, cat_raw = build_raw_feature_pool(df_merged)
    df_eng, eng_cols = engineer_business_features(df_merged)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    X_all, y_all = prepare_credit_info(df_eng, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)

    log(f"  [{label}] {len(df_merged):,} khoản vay khớp, "
        f"{int(y_all.sum())}/{len(y_all)} sự kiện dương ({y_all.mean() * 100:.3f}%)")

    pipe = build_pipeline(model_key, cat_p, num_p, **_build_kwargs(strategy, model_key))
    per_repeat_df, summary = repeated_stratified_binary_eval(
        pipe, X_all, y_all,
        n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
        random_state=RANDOM_STATE, sample_weight_fn=_sample_weight_fn(strategy, model_key),
    )
    return {
        "pair": label, "n_total": len(df_merged), "n_pos": int(y_all.sum()),
        "positive_rate_pct": round(y_all.mean() * 100, 4),
        "model": model_key, "strategy": strategy, **summary,
    }


def main():
    cv_summary_path = RESULTS_DIR / "transition_cv_summary.csv"
    if not cv_summary_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {cv_summary_path} — chạy generate_transition_report.py trước.")
    cv_summary = pd.read_csv(cv_summary_path)
    best_row = cv_summary.sort_values("pr_auc_mean", ascending=False).iloc[0]
    best_key, best_strategy = best_row["model_key"], best_row["strategy"]
    log(f"Cấu hình tốt nhất theo cặp thật (không chạy lại): {best_row['model']} "
        f"— chiến lược {best_strategy} (PR-AUC={best_row['pr_auc_mean']:.4f})")

    log("Đang tải 20260531 (thật) và 20260630 (MÔ PHỎNG)…")
    df_531 = load_credit_info(DATA_CREDIT_INFO_TEST)
    if not DATA_CREDIT_INFO_SIM.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {DATA_CREDIT_INFO_SIM} — chạy generate_synthetic_period.py trước.")
    df_630_sim = load_credit_info(DATA_CREDIT_INFO_SIM)

    log("Đang đánh giá cặp 2 (531→630, kỳ sau MÔ PHỎNG) bằng đúng cấu hình tốt nhất của cặp 1…")
    row2 = _eval_pair(df_531, df_630_sim, best_key, best_strategy, "531_to_630_sim")

    row1 = {
        "pair": "430_to_531_real", "n_total": None, "n_pos": None,
        "positive_rate_pct": None, "model": best_row["model"], "strategy": best_strategy,
        "pr_auc_mean": best_row["pr_auc_mean"], "pr_auc_std": best_row["pr_auc_std"],
        "roc_auc_mean": best_row.get("roc_auc_mean"), "roc_auc_std": best_row.get("roc_auc_std"),
        "recall_pos_mean": best_row.get("recall_pos_mean"), "recall_pos_std": best_row.get("recall_pos_std"),
    }
    out_df = pd.DataFrame([row1, row2])
    out_df.to_csv(RESULTS_DIR / "transition_multiperiod_demo.csv", index=False, encoding="utf-8-sig")
    log(f"Đã ghi results/transition_multiperiod_demo.csv:\n{out_df}")
    log("LƯU Ý: cặp 531→630 dùng kỳ sau MÔ PHỎNG (nhãn chỉ phụ thuộc nhóm nợ "
        "hiện tại). Chênh lệch PR-AUC giữa 2 cặp (nếu có) phản ánh giới hạn "
        "của cơ chế mô phỏng, KHÔNG phải bằng chứng mô hình bất ổn theo thời "
        "gian — chỉ minh họa rằng pipeline đã sẵn sàng tiếp nhận thêm kỳ dữ "
        "liệu thật khi ngân hàng cung cấp.")


if __name__ == "__main__":
    main()
