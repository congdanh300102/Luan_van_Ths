"""Script tái lập kết quả cho luận văn — Bài toán bổ sung: dự báo chuyển
nhóm nợ trong 1 tháng (bộ B, ghép 20260430 → 20260507).

Khác với generate_credit_info_report.py (phân loại NHÓM NỢ HIỆN TẠI tại mỗi
kỳ báo cáo), script này ghép hai kỳ báo cáo của cùng một danh mục khoản vay
qua khoá "Số khế ước" để xây nhãn CHUYỂN NHÓM thật: đặc trưng lấy tại kỳ T
(20260430), nhãn xác định từ kỳ T+1 tháng (20260507) — đúng thiết kế "dùng
thông tin hiện có để dự báo diễn biến tương lai" mà phân loại nhóm nợ hiện
tại (đã được CIC báo cáo) không làm được.

Không phải trang Streamlit — chạy độc lập:

    cd code && py -m src.generate_transition_report

Output:
  code/results/transition_*.csv                — các bảng số liệu
  luanvan_latex/figures/eda_transition_*.png    — các hình minh hoạ
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve

from config.config import (
    RANDOM_STATE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, TRANSITION_ID_COL, TRANSITION_SMOTE_STRATEGY,
    TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, feature_to_group,
    build_transition_dataset, transition_exclusion_summary,
    TRANSITION_LABEL_COL,
)
from src.iv_analysis import compute_iv_table
from src.models import build_pipeline, available_models, compute_sample_weights
from src.evaluation import (
    compute_binary_metrics, recall_at_k, repeated_stratified_binary_eval,
)
from src.explainability import compute_shap_values

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent.parent / "luanvan_latex" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["none", "class_weight", "custom"]
BINARY_LABELS = {0: "Không xấu đi", 1: "Xấu đi"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _build_kwargs(strategy: str, model_key: str) -> dict:
    kwargs = {"random_state": RANDOM_STATE, "imbalance_strategy": strategy}
    if strategy == "custom":
        kwargs["custom_smote_strategy"] = TRANSITION_SMOTE_STRATEGY
    if model_key == "catboost":
        kwargs["loss_function"] = "Logloss"
    if model_key == "xgboost":
        kwargs["eval_metric"] = "logloss"
    return kwargs


def _strategies_for(model_key: str) -> list:
    # Với CatBoost, mọi imbalance_strategy khác "none" đều ánh xạ về
    # auto_class_weights="Balanced" (không resample — xem models.py) nên
    # "class_weight" và "custom" cho ra đúng 1 pipeline giống hệt nhau; chạy
    # cả hai là lãng phí ~27s/fit × 100 lần lặp mà không thêm thông tin.
    if model_key == "catboost":
        return ["none", "class_weight"]
    return STRATEGIES


def _sample_weight_fn(strategy: str, model_key: str):
    # XGBoost không có class_weight built-in — "class_weight" strategy chỉ có
    # tác dụng nếu truyền sample_weight thủ công qua compute_sample_weights().
    if strategy == "class_weight" and model_key == "xgboost":
        return compute_sample_weights
    return None


def main():
    # ── 1. Load + ghép 2 kỳ báo cáo ─────────────────────────────────────────
    log("Đang tải dữ liệu kỳ T (20260430) và T+1 tháng (20260507)…")
    df_t_raw = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    df_t1_raw = load_credit_info(DATA_CREDIT_INFO_TEST)
    log(f"Kỳ T: {df_t_raw.shape}, Kỳ T+1: {df_t1_raw.shape}")

    df_merged = build_transition_dataset(
        df_t_raw, df_t1_raw, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    log(f"Số khoản vay khớp được ở cả hai kỳ: {len(df_merged):,} "
        f"/ {len(df_t_raw):,} ({len(df_merged) / len(df_t_raw) * 100:.1f}%)")

    excl_summary = transition_exclusion_summary(
        df_t_raw, df_merged, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    excl_summary.to_csv(RESULTS_DIR / "transition_exclusion_summary.csv",
                        index=False, encoding="utf-8-sig")
    log(f"Đối chiếu khoản vay bị loại (n={excl_summary.attrs.get('n_excluded')}) "
        f"vs. được giữ (n={excl_summary.attrs.get('n_matched')}):\n{excl_summary}")

    # ── 2. Phân bố nhãn chuyển nhóm ──────────────────────────────────────────
    n_total = len(df_merged)
    n_pos = int(df_merged[TRANSITION_LABEL_COL].sum())
    label_dist = pd.DataFrame([
        {"nhan": "Không xấu đi (0)", "so_luong": n_total - n_pos,
         "ty_le_pct": round((n_total - n_pos) / n_total * 100, 4)},
        {"nhan": "Xấu đi (1)", "so_luong": n_pos,
         "ty_le_pct": round(n_pos / n_total * 100, 4)},
    ])
    label_dist.to_csv(RESULTS_DIR / "transition_label_distribution.csv",
                      index=False, encoding="utf-8-sig")

    trans_matrix = pd.crosstab(
        df_merged[CREDIT_INFO_TARGET_COL].astype(int),
        df_merged["TRANSITION_GRP_T1"].astype(int),
        rownames=["Nhóm nợ tại T"], colnames=["Nhóm nợ tại T+1"],
    )
    trans_matrix.to_csv(RESULTS_DIR / "transition_matrix.csv", encoding="utf-8-sig")
    log(f"Phân bố nhãn: {n_pos}/{n_total} ({n_pos/n_total*100:.3f}%) chuyển sang nhóm cao hơn.")
    log(f"Ma trận chuyển nhóm:\n{trans_matrix}")

    # ── 3. Tập đặc trưng (tái dùng nguyên taxonomy bộ B, chỉ từ kỳ T) ───────
    num_raw, cat_raw = build_raw_feature_pool(df_merged)
    df_eng, eng_cols = engineer_business_features(df_merged)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    log(f"Tập đặc trưng: {len(num_p)} numerical (gồm {len(eng_cols)} engineered) "
        f"+ {len(cat_p)} categorical = {len(num_p) + len(cat_p)} tổng — "
        f"toàn bộ lấy từ kỳ T, không có cột nào của kỳ T+1.")

    X_all, y_all = prepare_credit_info(
        df_eng, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0,
    )
    log(f"X_all: {X_all.shape}, positive rate: {y_all.mean() * 100:.3f}%")

    # ── 4. Information Value so với nhãn chuyển nhóm ────────────────────────
    log("Đang tính Information Value so với TRANSITION_WORSENED…")
    iv_input = df_eng[num_p + cat_p + [TRANSITION_LABEL_COL]].copy()
    iv_table = compute_iv_table(iv_input, TRANSITION_LABEL_COL, good_class=0)
    iv_table.to_csv(RESULTS_DIR / "transition_iv_table.csv", index=False, encoding="utf-8-sig")
    log(f"IV top 10:\n{iv_table.head(10)}")

    # ── 5. So sánh 6 mô hình × 3 chiến lược mất cân bằng (Repeated CV) ──────
    log(f"Đang đánh giá {len(STRATEGIES)} chiến lược × 6 mô hình bằng "
        f"RepeatedStratifiedKFold ({TRANSITION_CV_SPLITS}×{TRANSITION_CV_REPEATS})…")
    models = available_models()
    cv_summary_rows, cv_per_repeat_frames = [], []

    for label, key in models.items():
        for strategy in _strategies_for(key):
            t0 = time.time()
            pipe = build_pipeline(key, cat_p, num_p, **_build_kwargs(strategy, key))
            per_repeat_df, summary = repeated_stratified_binary_eval(
                pipe, X_all, y_all,
                n_splits=TRANSITION_CV_SPLITS, n_repeats=TRANSITION_CV_REPEATS,
                random_state=RANDOM_STATE,
                sample_weight_fn=_sample_weight_fn(strategy, key),
            )
            per_repeat_df["model"] = label
            per_repeat_df["model_key"] = key
            per_repeat_df["strategy"] = strategy
            cv_per_repeat_frames.append(per_repeat_df)

            row = {"model": label, "model_key": key, "strategy": strategy, **summary}
            cv_summary_rows.append(row)
            log(f"  [{strategy}] {label}: PR-AUC={summary['pr_auc_mean']:.4f}"
                f"±{summary['pr_auc_std']:.4f} | ROC-AUC={summary['roc_auc_mean']:.4f}"
                f" | Recall(+)={summary['recall_pos_mean']:.3f} ({time.time()-t0:.1f}s)")

    cv_summary_df = pd.DataFrame(cv_summary_rows).sort_values("pr_auc_mean", ascending=False)
    cv_summary_df.to_csv(RESULTS_DIR / "transition_cv_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_per_repeat_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "transition_cv_per_repeat.csv", index=False, encoding="utf-8-sig")

    best_row = cv_summary_df.iloc[0]
    best_label, best_key, best_strategy = best_row["model"], best_row["model_key"], best_row["strategy"]
    log(f"Mô hình tốt nhất theo PR-AUC trung bình (CV): {best_label} — chiến lược {best_strategy} "
        f"(PR-AUC={best_row['pr_auc_mean']:.4f}±{best_row['pr_auc_std']:.4f})")

    # So sánh riêng 3 chiến lược mất cân bằng (đối với mô hình tốt nhất về model_key)
    strat_compare = cv_summary_df[cv_summary_df["model_key"] == best_key].sort_values(
        "pr_auc_mean", ascending=False)
    strat_compare.to_csv(RESULTS_DIR / "transition_imbalance_strategy_comparison.csv",
                         index=False, encoding="utf-8-sig")

    # ── 6. Một lần chia 80/20 minh hoạ (ma trận nhầm lẫn, PR curve, SHAP) ───
    # Chỉ mang tính minh hoạ/chẩn đoán — bằng chứng chính về hiệu năng là CV
    # ở bước 5, vì 1 lần chia đơn với ~53 sự kiện dương trên tập kiểm định
    # không đủ ổn định để làm căn cứ chọn mô hình.
    log("Đang huấn luyện lại mô hình tốt nhất trên 1 lần chia 80/20 để minh hoạ…")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE,
    )
    best_pipe = build_pipeline(best_key, cat_p, num_p, **_build_kwargs(best_strategy, best_key))
    sw_fn = _sample_weight_fn(best_strategy, best_key)
    if sw_fn is not None:
        best_pipe.fit(X_train, y_train, classifier__sample_weight=sw_fn(y_train))
    else:
        best_pipe.fit(X_train, y_train)

    y_pred_v = best_pipe.predict(X_val)
    y_proba_v = best_pipe.predict_proba(X_val)[:, 1]
    m_v = compute_binary_metrics(y_val, y_pred_v, y_proba_v)
    log(f"Kết quả minh hoạ (tập kiểm định 20%, n_pos={int(y_val.sum())}): "
        f"PR-AUC={m_v['pr_auc']:.4f}, ROC-AUC={m_v['roc_auc']:.4f}, "
        f"Recall(+)={m_v['recall_pos']:.3f}, Precision(+)={m_v['precision_pos']:.3f}")

    rk_df = recall_at_k(y_val, y_proba_v)
    rk_df.to_csv(RESULTS_DIR / "transition_recall_at_k.csv", index=False, encoding="utf-8-sig")
    log(f"Recall@k:\n{rk_df}")

    # Ma trận nhầm lẫn (matplotlib, nhất quán với các hình EDA khác của bộ B)
    cm = confusion_matrix(y_val, y_pred_v, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([BINARY_LABELS[0], BINARY_LABELS[1]])
    ax.set_yticks([0, 1]); ax.set_yticklabels([BINARY_LABELS[0], BINARY_LABELS[1]])
    ax.set_xlabel("Dự báo"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn — {best_label} ({best_strategy})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_val, y_proba_v)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#c0392b", linewidth=2)
    ax.axhline(y_val.mean(), color="gray", linestyle="--",
              label=f"Baseline ngẫu nhiên (prevalence={y_val.mean()*100:.2f}%)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — {best_label} ({best_strategy}), PR-AUC={m_v['pr_auc']:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_pr_curve.png", dpi=150)
    plt.close(fig)

    # Phân tán PR-AUC qua các lần lặp CV (mức độ bất ổn của ước lượng)
    cv_all = pd.concat(cv_per_repeat_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_models = cv_all[cv_all["strategy"] == best_strategy]
    order = (plot_models.groupby("model")["pr_auc"].mean()
            .sort_values(ascending=False).index.tolist())
    data = [plot_models.loc[plot_models["model"] == m, "pr_auc"].values for m in order]
    ax.boxplot(data, labels=order, showmeans=True)
    ax.set_ylabel("PR-AUC")
    ax.set_title(f"Phân tán PR-AUC qua {TRANSITION_CV_SPLITS}×{TRANSITION_CV_REPEATS} lần lặp CV "
                f"— chiến lược {best_strategy}")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_cv_spread.png", dpi=150)
    plt.close(fig)

    # Recall@k
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(rk_df["top_k_pct"].astype(str) + "%", rk_df["recall_at_k"], color="#2980b9")
    ax.set_xlabel("Top-k% hồ sơ rủi ro cao nhất")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall@k — {best_label} ({best_strategy})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_recall_at_k.png", dpi=150)
    plt.close(fig)
    log("Đã lưu 4 hình vào luanvan_latex/figures/")

    # ── 7. SHAP cho mô hình tốt nhất ─────────────────────────────────────────
    log("Đang tính SHAP values cho mô hình tốt nhất…")
    try:
        shap_values, feat_names, X_disp = compute_shap_values(
            best_pipe, X_val, best_key, max_samples=500)
        # shap_values: (n, n_features, n_classes) — lớp 1 = "Xấu đi"
        class_idx = 1 if shap_values.shape[-1] > 1 else 0
        shap_imp = np.abs(shap_values[:, :, class_idx]).mean(axis=0)
        shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": shap_imp}).sort_values(
            "mean_abs_shap", ascending=False)
        shap_df.to_csv(RESULTS_DIR / "transition_shap_top.csv", index=False, encoding="utf-8-sig")
        log(f"SHAP top 15 (lớp 'Xấu đi'):\n{shap_df.head(15)}")
    except Exception as e:
        log(f"Bỏ qua SHAP (lỗi: {e})")

    log("HOÀN TẤT. Bảng số liệu đã lưu vào code/results/transition_*.csv")
    log("Hình đã lưu vào luanvan_latex/figures/eda_transition_*.png")


if __name__ == "__main__":
    main()
