"""Script tái lập kết quả cho luận văn — cặp kỳ quan sát THẬT thứ hai của bài
toán chuyển nhóm nợ: 20260531 (T) -> 20260630 (T+1), bổ sung ngày 2026-09-12
khi ngân hàng cung cấp thêm kỳ báo cáo thứ ba.

Trước khi có file này, luận văn chỉ có MỘT cặp kỳ quan sát thật
(20260430->20260531, xem generate_transition_report.py) và một minh hoạ dùng
kỳ MÔ PHỎNG (generate_transition_multiperiod_demo.py). Script này thay minh
hoạ mô phỏng bằng bằng chứng thật thứ hai, cho phép đánh giá trực tiếp độ ổn
định của mô hình/chiến lược qua hai cặp kỳ độc lập.

Chạy đúng lại toàn bộ quy trình đã dùng cho cặp 1 (build_transition_dataset,
6 mô hình x chiến lược mất cân bằng, Repeated CV 5x10) trên cặp 2, PLUS một
kiểm tra chuyển giao ngoài thời gian (out-of-time transfer): huấn luyện đúng
một lần trên toàn bộ dữ liệu cặp 1 bằng cấu hình tốt nhất của cặp 1, rồi chấm
điểm trực tiếp trên dữ liệu cặp 2 mà KHÔNG huấn luyện lại — đây là phép kiểm
định nghiêm ngặt nhất về khả năng khái quát hoá theo thời gian của một mô
hình cụ thể, không chỉ của một họ cấu hình.

Không phải trang Streamlit — chạy độc lập:

    cd code && py -m src.generate_transition_report_pair2

Output:
  code/results/transition_pair2_*.csv              — bảng số liệu cặp 2
  code/results/transition_pairs_stability.csv       — so sánh cặp 1 vs cặp 2
  code/results/transition_oot_transfer.csv          — chuyển giao ngoài mẫu
  luanvan_latex/figures/eda_transition_pair2_*.png  — hình minh hoạ cặp 2
"""
import sys
import time
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
    RANDOM_STATE, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2,
    CREDIT_INFO_TARGET_COL, TRANSITION_ID_COL, TRANSITION_SMOTE_STRATEGY,
    TRANSITION_CV_SPLITS, TRANSITION_CV_REPEATS,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset, transition_exclusion_summary,
    TRANSITION_LABEL_COL,
)
from src.models import build_pipeline, available_models, compute_sample_weights
from src.evaluation import (
    compute_binary_metrics, recall_at_k, repeated_stratified_binary_eval,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent.parent / "luanvan_latex" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["none", "class_weight", "custom", "smote_class_weight"]
BINARY_LABELS = {0: "Không xấu đi", 1: "Xấu đi"}


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


def _strategies_for(model_key: str) -> list:
    if model_key == "catboost":
        return ["none", "class_weight"]
    return STRATEGIES


def _sample_weight_fn(strategy: str, model_key: str):
    if strategy == "class_weight" and model_key == "xgboost":
        return compute_sample_weights
    return None


def main():
    # ── 1. Load + ghép kỳ 20260531 (T) và 20260630 (T+1), cả hai đều THẬT ───
    log("Đang tải dữ liệu kỳ T (20260531) và T+1 (20260630, THẬT)…")
    df_t_raw = load_credit_info(DATA_CREDIT_INFO_TEST)
    df_t1_raw = load_credit_info(DATA_CREDIT_INFO_TEST2)
    log(f"Kỳ T: {df_t_raw.shape}, Kỳ T+1: {df_t1_raw.shape}")

    df_merged = build_transition_dataset(
        df_t_raw, df_t1_raw, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    log(f"Số khoản vay khớp được ở cả hai kỳ: {len(df_merged):,} "
        f"/ {len(df_t_raw):,} ({len(df_merged) / len(df_t_raw) * 100:.1f}%)")

    excl_summary = transition_exclusion_summary(
        df_t_raw, df_merged, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    excl_summary.to_csv(RESULTS_DIR / "transition_pair2_exclusion_summary.csv",
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
    label_dist.to_csv(RESULTS_DIR / "transition_pair2_label_distribution.csv",
                      index=False, encoding="utf-8-sig")

    trans_matrix = pd.crosstab(
        df_merged[CREDIT_INFO_TARGET_COL].astype(int),
        df_merged["TRANSITION_GRP_T1"].astype(int),
        rownames=["Nhóm nợ tại T"], colnames=["Nhóm nợ tại T+1"],
    )
    trans_matrix.to_csv(RESULTS_DIR / "transition_pair2_matrix.csv", encoding="utf-8-sig")
    log(f"Phân bố nhãn: {n_pos}/{n_total} ({n_pos/n_total*100:.3f}%) chuyển sang nhóm cao hơn.")
    log(f"Ma trận chuyển nhóm (cặp 2):\n{trans_matrix}")

    # ── 3. Tập đặc trưng (đúng taxonomy bộ B, chỉ từ kỳ T=20260531) ─────────
    num_raw, cat_raw = build_raw_feature_pool(df_merged)
    df_eng, eng_cols = engineer_business_features(df_merged)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    log(f"Tập đặc trưng: {len(num_p)} numerical (gồm {len(eng_cols)} engineered) "
        f"+ {len(cat_p)} categorical = {len(num_p) + len(cat_p)} tổng.")

    X_all, y_all = prepare_credit_info(
        df_eng, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0,
    )
    log(f"X_all: {X_all.shape}, positive rate: {y_all.mean() * 100:.3f}%")

    # ── 4. So sánh 6 mô hình x chiến lược mất cân bằng (Repeated CV) ────────
    log(f"Đang đánh giá {len(STRATEGIES)} chiến lược x 6 mô hình bằng "
        f"RepeatedStratifiedKFold ({TRANSITION_CV_SPLITS}x{TRANSITION_CV_REPEATS})…")
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
    cv_summary_df.to_csv(RESULTS_DIR / "transition_pair2_cv_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(cv_per_repeat_frames, ignore_index=True).to_csv(
        RESULTS_DIR / "transition_pair2_cv_per_repeat.csv", index=False, encoding="utf-8-sig")

    best_row2 = cv_summary_df.iloc[0]
    best_label2, best_key2, best_strategy2 = best_row2["model"], best_row2["model_key"], best_row2["strategy"]
    log(f"Mô hình tốt nhất trên cặp 2 theo PR-AUC trung bình (CV): {best_label2} — chiến lược "
        f"{best_strategy2} (PR-AUC={best_row2['pr_auc_mean']:.4f}±{best_row2['pr_auc_std']:.4f})")

    # ── 5. So sánh độ ổn định cặp 1 vs cặp 2 (cùng model_key + strategy) ────
    pair1_path = RESULTS_DIR / "transition_cv_summary.csv"
    stability_rows = []
    if pair1_path.exists():
        cv1 = pd.read_csv(pair1_path)
        for _, r1 in cv1.iterrows():
            match2 = cv_summary_df[(cv_summary_df["model_key"] == r1["model_key"]) &
                                    (cv_summary_df["strategy"] == r1["strategy"])]
            if len(match2) == 0:
                continue
            r2 = match2.iloc[0]
            stability_rows.append({
                "model": r1["model"], "strategy": r1["strategy"],
                "pr_auc_pair1": r1["pr_auc_mean"], "pr_auc_pair1_std": r1["pr_auc_std"],
                "pr_auc_pair2": r2["pr_auc_mean"], "pr_auc_pair2_std": r2["pr_auc_std"],
                "delta_pr_auc": r2["pr_auc_mean"] - r1["pr_auc_mean"],
                "roc_auc_pair1": r1["roc_auc_mean"], "roc_auc_pair2": r2["roc_auc_mean"],
            })
        stability_df = pd.DataFrame(stability_rows).sort_values("pr_auc_pair1", ascending=False)
        stability_df.to_csv(RESULTS_DIR / "transition_pairs_stability.csv", index=False, encoding="utf-8-sig")
        log(f"Đã ghi transition_pairs_stability.csv ({len(stability_df)} cấu hình đối chiếu được).")
        best1_row = cv1.sort_values("pr_auc_mean", ascending=False).iloc[0]
        best1_match2 = cv_summary_df[(cv_summary_df["model_key"] == best1_row["model_key"]) &
                                      (cv_summary_df["strategy"] == best1_row["strategy"])]
        if len(best1_match2):
            b2 = best1_match2.iloc[0]
            log(f"Cấu hình tốt nhất của CẶP 1 ({best1_row['model']}-{best1_row['strategy']}) "
                f"trên CẶP 2: PR-AUC={b2['pr_auc_mean']:.4f}±{b2['pr_auc_std']:.4f} "
                f"(cặp 1: {best1_row['pr_auc_mean']:.4f}±{best1_row['pr_auc_std']:.4f})")
    else:
        log("Không tìm thấy transition_cv_summary.csv (cặp 1) — bỏ qua bước so sánh ổn định.")

    # ── 6. Một lần chia 80/20 minh hoạ trên cặp 2 (ma trận nhầm lẫn, PR, Recall@k) ─
    log("Đang huấn luyện lại mô hình tốt nhất của CẶP 2 trên 1 lần chia 80/20 để minh hoạ…")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE,
    )
    best_pipe2 = build_pipeline(best_key2, cat_p, num_p, **_build_kwargs(best_strategy2, best_key2))
    sw_fn2 = _sample_weight_fn(best_strategy2, best_key2)
    if sw_fn2 is not None:
        best_pipe2.fit(X_train, y_train, classifier__sample_weight=sw_fn2(y_train))
    else:
        best_pipe2.fit(X_train, y_train)

    y_pred_v = best_pipe2.predict(X_val)
    y_proba_v = best_pipe2.predict_proba(X_val)[:, 1]
    m_v = compute_binary_metrics(y_val, y_pred_v, y_proba_v)
    log(f"Kết quả minh hoạ cặp 2 (tập kiểm định 20%, n_pos={int(y_val.sum())}): "
        f"PR-AUC={m_v['pr_auc']:.4f}, ROC-AUC={m_v['roc_auc']:.4f}, "
        f"Recall(+)={m_v['recall_pos']:.3f}, Precision(+)={m_v['precision_pos']:.3f}")

    rk_df = recall_at_k(y_val, y_proba_v)
    rk_df.to_csv(RESULTS_DIR / "transition_pair2_recall_at_k.csv", index=False, encoding="utf-8-sig")
    log(f"Recall@k (cặp 2):\n{rk_df}")

    cm = confusion_matrix(y_val, y_pred_v, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([BINARY_LABELS[0], BINARY_LABELS[1]])
    ax.set_yticks([0, 1]); ax.set_yticklabels([BINARY_LABELS[0], BINARY_LABELS[1]])
    ax.set_xlabel("Dự báo"); ax.set_ylabel("Thực tế")
    ax.set_title(f"Ma trận nhầm lẫn — cặp 2 — {best_label2} ({best_strategy2})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_pair2_confusion_matrix.png", dpi=150)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_val, y_proba_v)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="#c0392b", linewidth=2)
    ax.axhline(y_val.mean(), color="gray", linestyle="--",
              label=f"Baseline ngẫu nhiên (prevalence={y_val.mean()*100:.2f}%)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall — cặp 2 — {best_label2} ({best_strategy2}), PR-AUC={m_v['pr_auc']:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_transition_pair2_pr_curve.png", dpi=150)
    plt.close(fig)

    # ── 7. Chuyển giao ngoài thời gian (out-of-time transfer): huấn luyện trên
    # TOÀN BỘ dữ liệu cặp 1 với cấu hình tốt nhất của cặp 1, chấm điểm trực
    # tiếp trên TOÀN BỘ dữ liệu cặp 2 — KHÔNG huấn luyện lại. Đây là phép thử
    # nghiêm ngặt nhất: một mô hình cụ thể, cố định, có khái quát hoá sang một
    # kỳ dữ liệu hoàn toàn chưa thấy hay không.
    if pair1_path.exists():
        log("Đang thực hiện kiểm tra chuyển giao ngoài thời gian (train cặp 1 -> test cặp 2)…")
        from config.config import DATA_CREDIT_INFO_TRAIN
        df_pair1_t = load_credit_info(DATA_CREDIT_INFO_TRAIN)
        df_pair1_t1 = load_credit_info(DATA_CREDIT_INFO_TEST)
        df_merged1 = build_transition_dataset(
            df_pair1_t, df_pair1_t1, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
        )
        num_raw1, cat_raw1 = build_raw_feature_pool(df_merged1)
        df_eng1, eng_cols1 = engineer_business_features(df_merged1)
        num_p1, cat_p1 = num_raw1 + eng_cols1, cat_raw1
        X_all1, y_all1 = prepare_credit_info(df_eng1, TRANSITION_LABEL_COL, num_p1, cat_p1, label_offset=0)

        best1_row = cv1.sort_values("pr_auc_mean", ascending=False).iloc[0]
        oot_key, oot_strategy = best1_row["model_key"], best1_row["strategy"]
        oot_pipe = build_pipeline(oot_key, cat_p1, num_p1, **_build_kwargs(oot_strategy, oot_key))
        sw_fn_oot = _sample_weight_fn(oot_strategy, oot_key)
        if sw_fn_oot is not None:
            oot_pipe.fit(X_all1, y_all1, classifier__sample_weight=sw_fn_oot(y_all1))
        else:
            oot_pipe.fit(X_all1, y_all1)

        # Chấm điểm trên TOÀN BỘ dữ liệu cặp 2 (không chia train/test — mô
        # hình đã cố định, không dùng bất kỳ dòng nào của cặp 2 để huấn luyện).
        common_num = [c for c in num_p1 if c in X_all.columns]
        common_cat = [c for c in cat_p1 if c in X_all.columns]
        X_pair2_for_oot = X_all[common_num + common_cat]
        y_proba_oot = oot_pipe.predict_proba(X_pair2_for_oot)[:, 1]
        y_pred_oot = oot_pipe.predict(X_pair2_for_oot)
        m_oot = compute_binary_metrics(y_all, y_pred_oot, y_proba_oot)
        oot_df = pd.DataFrame([{
            "model": best1_row["model"], "strategy": oot_strategy,
            "n_train_pair1": len(X_all1), "n_test_pair2": len(X_pair2_for_oot),
            "n_pos_pair2": int(y_all.sum()),
            "pr_auc": m_oot["pr_auc"], "roc_auc": m_oot["roc_auc"],
            "recall_pos": m_oot["recall_pos"], "precision_pos": m_oot["precision_pos"],
        }])
        oot_df.to_csv(RESULTS_DIR / "transition_oot_transfer.csv", index=False, encoding="utf-8-sig")
        log(f"Chuyển giao ngoài thời gian (huấn luyện cặp 1, chấm điểm TOÀN BỘ cặp 2, "
            f"không huấn luyện lại): PR-AUC={m_oot['pr_auc']:.4f}, ROC-AUC={m_oot['roc_auc']:.4f}, "
            f"Recall(+)={m_oot['recall_pos']:.3f}, Precision(+)={m_oot['precision_pos']:.3f}")
    else:
        log("Bỏ qua bước chuyển giao ngoài thời gian (thiếu dữ liệu cặp 1).")

    log("HOÀN TẤT. Bảng số liệu đã lưu vào code/results/transition_pair2_*.csv, "
        "transition_pairs_stability.csv, transition_oot_transfer.csv")
    log("Hình đã lưu vào luanvan_latex/figures/eda_transition_pair2_*.png")


if __name__ == "__main__":
    main()
