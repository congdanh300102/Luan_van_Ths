"""Trang 2 — Huấn luyện & Đánh giá mô hình"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config.config import (
    DATA_RAW, MODEL_DIR, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
)
from src.preprocessing import prepare
from src.models import build_pipeline, available_models, IMBALANCE_OPTIONS, compute_sample_weights
from src.evaluation import (
    compute_metrics, plot_confusion_matrix,
    plot_feature_importance, plot_model_comparison,
    cross_val_scores, repeated_stratified_recall,
    predict_with_class_weights, tune_class_boost, NHOMNO_LABELS,
)
from src.data_loader import get_raw_bytes

st.set_page_config(page_title="Huấn luyện", page_icon="🤖", layout="wide")
st.title("🤖 Huấn luyện & Đánh giá mô hình")


# ── Cache dữ liệu ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dữ liệu…")
def load_prepared(raw_bytes: bytes):
    import io
    df = pd.read_excel(io.BytesIO(raw_bytes))
    X, y = prepare(df, DROP_COLS, TARGET_COL)
    cat_p = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_p = [c for c in NUMERICAL_COLS   if c in X.columns]
    return X, y, cat_p, num_p


_raw_bytes = get_raw_bytes(DATA_RAW)
X_all, y_all, cat_cols, num_cols = load_prepared(_raw_bytes)
y_0 = y_all - 1  # 0-based cho XGBoost

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
)

# ── Sidebar: cấu hình ────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    MODEL_OPTIONS = available_models()
    default_choice = list(MODEL_OPTIONS.keys())[-1:]  # chọn model cuối trong danh sách
    selected_labels = st.multiselect(
        "Chọn mô hình",
        options=list(MODEL_OPTIONS.keys()),
        default=default_choice,
    )
    imbalance_label = st.selectbox(
        "Chiến lược xử lý imbalance",
        options=list(IMBALANCE_OPTIONS.keys()),
        index=0,
        help="SMOTE moderate: chỉ oversample minority vừa phải, không double-count với class_weight",
    )
    imbalance_key = IMBALANCE_OPTIONS[imbalance_label]

    run_cv = st.checkbox("Chạy Cross-Validation (chậm hơn)", value=False)

    run_repeated_cv = st.checkbox(
        "Repeated Stratified K-Fold cho nhóm hiếm (chậm hơn nhiều)",
        value=False,
        help=("Với nhóm nợ có rất ít quan sát (VD nhóm 3, 4), Recall đo trên "
              "1 lần train/test split có phương sai rất lớn — 1 hồ sơ sai đã "
              "làm Recall nhảy vài chục %. Lặp lại Stratified K-Fold nhiều "
              "lần và lấy trung bình cho ước lượng đáng tin cậy hơn."),
    )
    n_repeats = 3
    if run_repeated_cv:
        n_repeats = st.number_input("Số lần lặp (n_repeats)", min_value=1, max_value=20, value=3)

    st.markdown(f"**Train**: {len(X_train):,} | **Test**: {len(X_test):,}")
    st.caption(f"Imbalance ratio: 34.7x (N1=77.6% vs N3=2.2%)")

train_btn = st.button("🚀 Bắt đầu huấn luyện", type="primary",
                       disabled=len(selected_labels) == 0)

# ── Trạng thái session ────────────────────────────────────────────────────────
if "trained_results" not in st.session_state:
    st.session_state.trained_results = []
if "trained_pipelines" not in st.session_state:
    st.session_state.trained_pipelines = {}

# ── Huấn luyện ───────────────────────────────────────────────────────────────
if train_btn:
    st.session_state.trained_results = []
    st.session_state.trained_pipelines = {}

    for label in selected_labels:
        key = MODEL_OPTIONS[label]
        with st.spinner(f"Đang huấn luyện {label}…"):
            pipe = build_pipeline(key, cat_cols, num_cols, RANDOM_STATE,
                                  imbalance_strategy=imbalance_key)

            # XGBoost không có tham số class_weight built-in (khác với
            # LogisticRegression/DecisionTree/RandomForest/LightGBM) — phải
            # tự tính sample_weight và truyền qua fit() để cost-sensitive
            # thực sự có hiệu lực khi chọn chiến lược "class_weight".
            use_sample_weight = key == "xgboost" and imbalance_key == "class_weight"
            sample_weight_fn = compute_sample_weights if use_sample_weight else None

            cv_info = {}
            if run_cv:
                with st.spinner(f"  Cross-validation {label}…"):
                    cv_info = cross_val_scores(pipe, X_all, y_0, CV_FOLDS, RANDOM_STATE)

            repeated_cv_df, repeated_cv_summary = None, None
            if run_repeated_cv:
                with st.spinner(f"  Repeated Stratified K-Fold {label} ({n_repeats} lần)…"):
                    repeated_cv_df, repeated_cv_summary = repeated_stratified_recall(
                        pipe, X_all, y_0, n_splits=CV_FOLDS, n_repeats=n_repeats,
                        random_state=RANDOM_STATE, sample_weight_fn=sample_weight_fn,
                    )

            if use_sample_weight:
                pipe.fit(X_train, y_train,
                        classifier__sample_weight=compute_sample_weights(y_train))
            else:
                pipe.fit(X_train, y_train)
            y_pred  = pipe.predict(X_test) + 1
            y_proba = pipe.predict_proba(X_test)
            y_true  = y_test + 1

            metrics = compute_metrics(y_true, y_pred, y_proba)
            metrics["model"] = label
            metrics["key"]   = key

            st.session_state.trained_results.append({
                "model": label, "key": key,
                **{k: v for k, v in metrics.items() if k not in ("report", "model", "key")},
                "cv": cv_info,
                "repeated_cv_df": repeated_cv_df,
                "repeated_cv_summary": repeated_cv_summary,
                "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
            })
            st.session_state.trained_pipelines[key] = pipe

            # Lưu model
            with open(MODEL_DIR / f"model_{key}.pkl", "wb") as f:
                pickle.dump(pipe, f)

    # Lưu mô hình tốt nhất theo ROC-AUC
    best = max(st.session_state.trained_results, key=lambda x: x.get("roc_auc", 0))
    (MODEL_DIR / "best_model.txt").write_text(best["key"])
    st.success(f"✅ Hoàn tất! Mô hình tốt nhất: **{best['model']}** (ROC-AUC = {best['roc_auc']:.4f})")

# ── Hiển thị kết quả ─────────────────────────────────────────────────────────
results = st.session_state.trained_results

if results:
    # Tổng quan metrics
    st.markdown("---")
    st.subheader("📈 Kết quả đánh giá")

    metric_df = pd.DataFrame([
        {"Mô hình": r["model"],
         "Macro F1": f"{r['f1_macro']:.4f}",
         "Weighted F1": f"{r['f1_weighted']:.4f}",
         "ROC-AUC": f"{r['roc_auc']:.4f}",
         **({f"CV Macro F1": f"{r['cv']['f1_macro_mean']:.4f} ± {r['cv']['f1_macro_std']:.4f}"}
            if r.get("cv") else {}),
        }
        for r in results
    ])
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    # So sánh mô hình (chỉ khi > 1)
    if len(results) > 1:
        fig_cmp = plot_model_comparison([
            {"model": r["model"], "f1_macro": r["f1_macro"],
             "f1_weighted": r["f1_weighted"], "roc_auc": r["roc_auc"]}
            for r in results
        ])
        st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Chi tiết từng mô hình")

    tabs = st.tabs([r["model"] for r in results])
    for tab, result in zip(tabs, results):
        with tab:
            key = result["key"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Macro F1",    f"{result['f1_macro']:.4f}")
            c2.metric("Weighted F1", f"{result['f1_weighted']:.4f}")
            c3.metric("ROC-AUC",     f"{result['roc_auc']:.4f}")

            if result.get("cv"):
                st.info(
                    f"Cross-Val ({CV_FOLDS}-fold) — "
                    f"Macro F1: {result['cv']['f1_macro_mean']:.4f} ± {result['cv']['f1_macro_std']:.4f} | "
                    f"Weighted F1: {result['cv']['f1_weighted_mean']:.4f} ± {result['cv']['f1_weighted_std']:.4f}"
                )

            if result.get("repeated_cv_df") is not None:
                s = result["repeated_cv_summary"]
                st.markdown(
                    f"**📊 Repeated Stratified K-Fold ({s['n_repeats']}×{s['n_splits']}-fold) "
                    f"— Recall theo nhóm nợ**"
                )
                st.caption(
                    "Đánh giá qua nhiều lần lặp thay vì 1 lần train/test split — ổn định hơn "
                    "cho các nhóm có rất ít quan sát (VD nhóm 3, 4)."
                )
                st.dataframe(result["repeated_cv_df"], use_container_width=True, hide_index=True)
                st.caption(
                    f"Macro F1 trung bình qua các fold: "
                    f"{s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f}"
                )

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Confusion Matrix**")
                fig_cm = plot_confusion_matrix(result["y_true"], result["y_pred"])
                st.plotly_chart(fig_cm, use_container_width=True)

            with col_b:
                st.markdown("**Classification Report**")
                report_df = pd.DataFrame(result.get("report", {})).T
                numeric_cols_r = report_df.select_dtypes(include=float).columns
                st.dataframe(report_df.style.format({c: "{:.3f}" for c in numeric_cols_r}),
                             use_container_width=True)

            # Feature importance
            pipe = st.session_state.trained_pipelines.get(key)
            if pipe:
                fig_fi = plot_feature_importance(pipe, top_n=20)
                if fig_fi:
                    st.markdown("**Feature Importance**")
                    st.plotly_chart(fig_fi, use_container_width=True)

            # Hiệu chỉnh ngưỡng post-hoc cho nhóm hiếm (không cần huấn luyện lại)
            with st.expander("🎯 Hiệu chỉnh ngưỡng cho nhóm hiếm (post-hoc class-boost)"):
                st.caption(
                    "Tăng Recall nhóm hiếm bằng argmax có trọng số (dịch ranh giới quyết "
                    "định) — không cần huấn luyện lại mô hình. Boost tìm được ở đây tối ưu "
                    "ngay trên tập test hiện dùng để báo cáo nên chỉ mang tính minh hoạ "
                    "hướng cải thiện; áp dụng thực tế cần tìm boost trên tập validation "
                    "tách riêng khỏi tập test cuối cùng."
                )
                support_counts = pd.Series(y_train).value_counts()
                rarest_default = support_counts.sort_values().index[:2].tolist()
                target_classes = st.multiselect(
                    "Chọn nhóm (0-based: 0=Nhóm 1 … 4=Nhóm 5) cần tăng Recall",
                    options=sorted(np.unique(y_test)),
                    default=rarest_default,
                    key=f"boost_target_{key}",
                )
                if st.button(f"Tìm boost tối ưu", key=f"boost_btn_{key}"):
                    if not target_classes:
                        st.warning("Chọn ít nhất 1 nhóm để boost.")
                    else:
                        best_boost, best_score = tune_class_boost(
                            result["y_true"] - 1, result["y_proba"], target_classes,
                        )
                        y_pred_boosted = predict_with_class_weights(result["y_proba"], best_boost) + 1

                        boost_display = {NHOMNO_LABELS.get(c + 1, str(c)): w
                                         for c, w in best_boost.items()}
                        st.success(
                            f"Boost tối ưu {boost_display} → Macro F1 = {best_score:.4f} "
                            f"(gốc chưa boost: {result['f1_macro']:.4f})"
                        )

                        c1b, c2b = st.columns(2)
                        with c1b:
                            st.markdown("**Confusion Matrix — sau boost**")
                            st.plotly_chart(
                                plot_confusion_matrix(result["y_true"], y_pred_boosted),
                                use_container_width=True,
                            )
                        with c2b:
                            st.markdown("**Classification Report — sau boost**")
                            report_boosted = pd.DataFrame(classification_report(
                                result["y_true"], y_pred_boosted,
                                output_dict=True, zero_division=0,
                            )).T
                            num_cols_b = report_boosted.select_dtypes(include=float).columns
                            st.dataframe(
                                report_boosted.style.format({c: "{:.3f}" for c in num_cols_b}),
                                use_container_width=True,
                            )

else:
    st.info("👈 Chọn mô hình trong sidebar và nhấn **Bắt đầu huấn luyện**.")

    # Kiểm tra mô hình đã lưu
    saved = list(MODEL_DIR.glob("model_*.pkl"))
    if saved:
        st.markdown("---")
        st.markdown(f"**Mô hình đã lưu** ({len(saved)} file):")
        for p in saved:
            st.markdown(f"- `{p.name}`")
        st.markdown("Bạn có thể chuyển sang **Trang 3 — Chấm điểm** để dùng các mô hình này.")
