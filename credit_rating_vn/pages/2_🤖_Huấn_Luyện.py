"""Trang 2 — Huấn luyện & Đánh giá mô hình"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, MODEL_DIR, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS,
)
from src.preprocessing import prepare
from src.models import build_pipeline, available_models
from src.evaluation import (
    compute_metrics, plot_confusion_matrix,
    plot_feature_importance, plot_model_comparison,
    cross_val_scores,
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
    run_cv = st.checkbox("Chạy Cross-Validation (chậm hơn)", value=False)
    st.markdown(f"**Train**: {len(X_train):,} | **Test**: {len(X_test):,}")

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
            pipe = build_pipeline(key, cat_cols, num_cols, RANDOM_STATE)

            cv_info = {}
            if run_cv:
                with st.spinner(f"  Cross-validation {label}…"):
                    cv_info = cross_val_scores(pipe, X_all, y_0, CV_FOLDS, RANDOM_STATE)

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
