"""Trang 6 — So sánh mô hình cũ (Data_credit_rating_VN) vs mô hình mới (fct_l)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, MODEL_DIR, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS, RANDOM_STATE, TEST_SIZE, SCORE_BANDS,
    DATA_FCT_L, DATA_FCT_L_PROCESSED,
    FCT_L_TARGET_COL, FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS,
    FCT_L_NHOMNO_LABELS, GROUP_COLORS,
)
from src.preprocessing import prepare
from src.fct_l_preprocessing import prepare_fct_l
from src.models import available_models
from src.scoring import proba_to_score, classify_score, build_score_df, plot_score_distribution, gradient_style
from src.evaluation import compute_metrics, plot_confusion_matrix
from src.data_loader import get_raw_bytes

st.set_page_config(page_title="So sánh Model", page_icon="⚖️", layout="wide")
st.title("⚖️ So sánh Mô hình")
st.markdown(
    "So sánh hiệu năng giữa **mô hình cũ** (huấn luyện trên `Data_credit_rating_VN.xlsx`) "
    "và **mô hình FCT_L** (huấn luyện trên `fct_l.xlsx`)."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_old_model(key: str):
    p = MODEL_DIR / f"model_{key}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=False)
def load_fct_l_model(key: str):
    p = MODEL_DIR / f"model_fct_l_{key}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Đang tải tập kiểm tra (mô hình cũ)…")
def get_old_test_set(raw_bytes: bytes):
    df = pd.read_excel(io.BytesIO(raw_bytes))
    X, y = prepare(df, DROP_COLS, TARGET_COL)
    y_0 = y - 1
    _, X_test, _, y_test = train_test_split(
        X, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
    )
    return X_test, y_test


@st.cache_data(show_spinner="Đang tải tập kiểm tra (mô hình FCT_L)…")
def get_fct_l_test_set(use_30k: bool):
    if use_30k and DATA_FCT_L_PROCESSED.exists():
        df = pd.read_csv(DATA_FCT_L_PROCESSED)
    elif DATA_FCT_L.exists():
        df = pd.read_excel(DATA_FCT_L)
    else:
        return None, None
    X, y = prepare_fct_l(df, FCT_L_TARGET_COL, FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return X_test, y_test


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    avail_old = {l: k for l, k in available_models().items()
                 if (MODEL_DIR / f"model_{k}.pkl").exists()}
    avail_fctl = {l: k for l, k in available_models().items()
                  if (MODEL_DIR / f"model_fct_l_{k}.pkl").exists()}

    if avail_old:
        old_lbl = st.selectbox("Mô hình cũ (Data_credit_rating_VN)", list(avail_old.keys()))
        old_key = avail_old[old_lbl]
    else:
        st.warning("Chưa có mô hình cũ. Huấn luyện ở **Trang 2**.")
        old_lbl, old_key = None, None

    if avail_fctl:
        fctl_lbl = st.selectbox("Mô hình FCT_L (fct_l.xlsx)", list(avail_fctl.keys()))
        fctl_key = avail_fctl[fctl_lbl]
    else:
        st.warning("Chưa có mô hình FCT_L. Huấn luyện ở **Trang 5**.")
        fctl_lbl, fctl_key = None, None

    use_30k = st.checkbox("Dùng fct_l_30k (30k hồ sơ) cho tập test FCT_L", value=True)

    st.markdown("---")
    st.markdown("**Ghi chú**")
    st.caption(
        "Hai mô hình được đánh giá trên **tập test riêng biệt** tương ứng với "
        "bộ dữ liệu huấn luyện. Không thể so sánh trực tiếp trên cùng 1 tập test "
        "vì mỗi mô hình yêu cầu bộ đặc trưng khác nhau."
    )

if old_key is None and fctl_key is None:
    st.info("Chưa có mô hình nào. Vui lòng huấn luyện mô hình ở **Trang 2** và **Trang 5** trước.")
    st.stop()

# ── Load models & test sets ───────────────────────────────────────────────────
with st.spinner("Đang tải mô hình và dữ liệu…"):
    old_pipe  = load_old_model(old_key)   if old_key   else None
    fctl_pipe = load_fct_l_model(fctl_key) if fctl_key else None

    old_metrics, fctl_metrics = None, None
    old_scores_df, fctl_scores_df = None, None

    if old_pipe is not None:
        try:
            _raw_bytes = get_raw_bytes(DATA_RAW)
            X_old_test, y_old_test = get_old_test_set(_raw_bytes)
            old_pred  = old_pipe.predict(X_old_test) + 1
            old_proba = old_pipe.predict_proba(X_old_test)
            old_true  = y_old_test + 1
            old_metrics = compute_metrics(old_true, old_pred, old_proba)
            old_scores_df = build_score_df(old_proba, old_pred, old_true)
        except Exception as e:
            st.error(f"Lỗi tải mô hình cũ: {e}")

    if fctl_pipe is not None:
        try:
            X_fctl_test, y_fctl_test = get_fct_l_test_set(use_30k)
            if X_fctl_test is not None:
                fctl_pred  = fctl_pipe.predict(X_fctl_test) + 1
                fctl_proba = fctl_pipe.predict_proba(X_fctl_test)
                fctl_true  = y_fctl_test + 1
                fctl_metrics = compute_metrics(fctl_true, fctl_pred, fctl_proba)
                fctl_scores_df = build_score_df(fctl_proba, fctl_pred, fctl_true)
        except Exception as e:
            st.error(f"Lỗi tải mô hình FCT_L: {e}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_metrics, tab_scores, tab_cm, tab_detail = st.tabs([
    "📈 Chỉ số hiệu năng",
    "💳 Phân phối điểm",
    "🔢 Confusion Matrix",
    "📋 Chi tiết",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Metrics comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.subheader("So sánh chỉ số hiệu năng")

    # Summary cards
    col_a, col_sep, col_b = st.columns([5, 1, 5])

    with col_a:
        st.markdown(f"#### 🔵 Mô hình cũ — {old_lbl or 'N/A'}")
        st.caption("Dữ liệu: `Data_credit_rating_VN.xlsx` | Target: `NHOMNOMOI`")
        if old_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Macro F1",    f"{old_metrics['f1_macro']:.4f}")
            c2.metric("Weighted F1", f"{old_metrics['f1_weighted']:.4f}")
            c3.metric("ROC-AUC",     f"{old_metrics['roc_auc']:.4f}")
        else:
            st.warning("Không có dữ liệu.")

    with col_sep:
        st.markdown("<div style='text-align:center;font-size:2rem;padding-top:40px'>VS</div>",
                    unsafe_allow_html=True)

    with col_b:
        st.markdown(f"#### 🟠 Mô hình FCT_L — {fctl_lbl or 'N/A'}")
        st.caption("Dữ liệu: `fct_l.xlsx` | Target: `CLASSIFICATION`")
        if fctl_metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Macro F1",    f"{fctl_metrics['f1_macro']:.4f}")
            c2.metric("Weighted F1", f"{fctl_metrics['f1_weighted']:.4f}")
            c3.metric("ROC-AUC",     f"{fctl_metrics['roc_auc']:.4f}")
        else:
            st.warning("Không có dữ liệu.")

    st.markdown("---")

    # Bar chart comparison
    rows = []
    if old_metrics:
        rows.append({
            "Mô hình": f"Mô hình cũ ({old_lbl})",
            "Macro F1": old_metrics["f1_macro"],
            "Weighted F1": old_metrics["f1_weighted"],
            "ROC-AUC": old_metrics["roc_auc"],
        })
    if fctl_metrics:
        rows.append({
            "Mô hình": f"Mô hình FCT_L ({fctl_lbl})",
            "Macro F1": fctl_metrics["f1_macro"],
            "Weighted F1": fctl_metrics["f1_weighted"],
            "ROC-AUC": fctl_metrics["roc_auc"],
        })

    if rows:
        df_cmp = pd.DataFrame(rows)
        fig_bar = go.Figure()
        colors = ["#3498db", "#e67e22"]
        for metric in ["Macro F1", "Weighted F1", "ROC-AUC"]:
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=df_cmp["Mô hình"],
                y=df_cmp[metric],
                text=df_cmp[metric].map(lambda v: f"{v:.4f}"),
                textposition="auto",
            ))
        fig_bar.update_layout(
            barmode="group",
            title="So sánh hiệu năng hai mô hình",
            yaxis=dict(range=[0, 1]),
            height=400,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Table
        st.markdown("#### Bảng chỉ số")
        cmp_tbl = df_cmp.set_index("Mô hình")
        st.dataframe(
            cmp_tbl.style.format("{:.4f}").apply(gradient_style),
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Score distributions
# ══════════════════════════════════════════════════════════════════════════════
with tab_scores:
    st.subheader("Phân phối điểm tín dụng")

    # KPI comparison
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"#### 🔵 Mô hình cũ — {old_lbl or 'N/A'}")
        if old_scores_df is not None:
            scores = old_scores_df["diem_tin_dung"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Điểm TB",     f"{scores.mean():.0f}")
            c2.metric("Thấp nhất",   f"{scores.min()}")
            c3.metric("Cao nhất",    f"{scores.max()}")
    with col_b:
        st.markdown(f"#### 🟠 Mô hình FCT_L — {fctl_lbl or 'N/A'}")
        if fctl_scores_df is not None:
            scores = fctl_scores_df["diem_tin_dung"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Điểm TB",     f"{scores.mean():.0f}")
            c2.metric("Thấp nhất",   f"{scores.min()}")
            c3.metric("Cao nhất",    f"{scores.max()}")

    # Overlay histogram
    if old_scores_df is not None or fctl_scores_df is not None:
        fig_hist = go.Figure()
        if old_scores_df is not None:
            fig_hist.add_trace(go.Histogram(
                x=old_scores_df["diem_tin_dung"],
                name=f"Mô hình cũ ({old_lbl})",
                nbinsx=40,
                opacity=0.65,
                marker_color="#3498db",
            ))
        if fctl_scores_df is not None:
            fig_hist.add_trace(go.Histogram(
                x=fctl_scores_df["diem_tin_dung"],
                name=f"Mô hình FCT_L ({fctl_lbl})",
                nbinsx=40,
                opacity=0.65,
                marker_color="#e67e22",
            ))
        for lo, _, grade, _, color in SCORE_BANDS:
            fig_hist.add_vline(x=lo, line_dash="dash", line_color="gray",
                               line_width=1, annotation_text=grade, annotation_position="top right")
        fig_hist.update_layout(
            barmode="overlay",
            title="Phân phối điểm tín dụng (so sánh)",
            xaxis_title="Điểm tín dụng",
            yaxis_title="Số lượng hồ sơ",
            height=420,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Phân bổ hạng side-by-side
    col_a, col_b = st.columns(2)
    grade_order = [b[2] for b in SCORE_BANDS]

    with col_a:
        if old_scores_df is not None:
            st.markdown(f"**Phân bổ hạng — Mô hình cũ**")
            vc = old_scores_df["hang"].value_counts().reindex(grade_order, fill_value=0)
            grade_tbl = pd.DataFrame({
                "Hạng": vc.index,
                "Mô tả": [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc.index],
                "Số lượng": vc.values,
                "Tỷ lệ (%)": (vc.values / len(old_scores_df) * 100).round(1),
            })
            st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

    with col_b:
        if fctl_scores_df is not None:
            st.markdown(f"**Phân bổ hạng — Mô hình FCT_L**")
            vc = fctl_scores_df["hang"].value_counts().reindex(grade_order, fill_value=0)
            grade_tbl = pd.DataFrame({
                "Hạng": vc.index,
                "Mô tả": [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc.index],
                "Số lượng": vc.values,
                "Tỷ lệ (%)": (vc.values / len(fctl_scores_df) * 100).round(1),
            })
            st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

    # Boxplot điểm theo nhóm nợ thực tế
    st.markdown("#### Điểm tín dụng theo nhóm nợ thực tế")
    col_a, col_b = st.columns(2)

    with col_a:
        if old_scores_df is not None and "nhom_thuc_te" in old_scores_df.columns:
            fig = go.Figure()
            for g in sorted(old_scores_df["nhom_thuc_te"].unique()):
                data = old_scores_df.loc[old_scores_df["nhom_thuc_te"] == g, "diem_tin_dung"]
                fig.add_trace(go.Box(
                    y=data, name=f"Nhóm {g}",
                    marker_color=list(GROUP_COLORS.values())[int(g) - 1],
                    boxpoints="outliers",
                ))
            fig.update_layout(
                title=f"Mô hình cũ — {old_lbl}",
                yaxis_title="Điểm tín dụng", height=380, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        if fctl_scores_df is not None and "nhom_thuc_te" in fctl_scores_df.columns:
            fig = go.Figure()
            for g in sorted(fctl_scores_df["nhom_thuc_te"].unique()):
                data = fctl_scores_df.loc[fctl_scores_df["nhom_thuc_te"] == g, "diem_tin_dung"]
                fig.add_trace(go.Box(
                    y=data, name=f"Nhóm {g}",
                    marker_color=list(GROUP_COLORS.values())[int(g) - 1],
                    boxpoints="outliers",
                ))
            fig.update_layout(
                title=f"Mô hình FCT_L — {fctl_lbl}",
                yaxis_title="Điểm tín dụng", height=380, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Confusion matrices
# ══════════════════════════════════════════════════════════════════════════════
with tab_cm:
    st.subheader("Confusion Matrix")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**🔵 Mô hình cũ — {old_lbl or 'N/A'}**")
        if old_metrics and old_scores_df is not None:
            fig_cm = plot_confusion_matrix(old_true, old_pred)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Không có dữ liệu.")

    with col_b:
        st.markdown(f"**🟠 Mô hình FCT_L — {fctl_lbl or 'N/A'}**")
        if fctl_metrics and fctl_scores_df is not None:
            fig_cm = plot_confusion_matrix(fctl_true, fctl_pred)
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("Không có dữ liệu.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Detail
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("Chi tiết từng mô hình")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"#### 🔵 Mô hình cũ — {old_lbl or 'N/A'}")
        st.markdown("""
        | Thông tin | Giá trị |
        |-----------|---------|
        | **Bộ dữ liệu** | Data_credit_rating_VN.xlsx |
        | **Target** | NHOMNOMOI (nhóm nợ mới) |
        | **Số hồ sơ** | ~27,001 |
        | **Số đặc trưng** | 11 (8 số học + 3 phân loại) |
        | **Nguồn** | CIC / nội bộ ngân hàng |
        """)
        if old_metrics:
            report_df = pd.DataFrame(old_metrics.get("report", {})).T
            num_r = report_df.select_dtypes(include=float).columns
            st.dataframe(
                report_df.style.format({c: "{:.3f}" for c in num_r}),
                use_container_width=True,
            )

    with col_b:
        st.markdown(f"#### 🟠 Mô hình FCT_L — {fctl_lbl or 'N/A'}")
        src_name = "fct_l_30k.csv (30k)" if use_30k else "fct_l.xlsx (5.4k)"
        st.markdown(f"""
        | Thông tin | Giá trị |
        |-----------|---------|
        | **Bộ dữ liệu** | {src_name} |
        | **Target** | CLASSIFICATION (nhóm nợ) |
        | **Số hồ sơ** | {"~30,000" if use_30k else "~5,400"} |
        | **Số đặc trưng** | {len(FCT_L_NUMERICAL_COLS) + len(FCT_L_CATEGORICAL_COLS)} ({len(FCT_L_NUMERICAL_COLS)} số học + {len(FCT_L_CATEGORICAL_COLS)} phân loại) |
        | **Nguồn** | Hệ thống core banking FCT_L |
        """)
        if fctl_metrics:
            report_df = pd.DataFrame(fctl_metrics.get("report", {})).T
            num_r = report_df.select_dtypes(include=float).columns
            st.dataframe(
                report_df.style.format({c: "{:.3f}" for c in num_r}),
                use_container_width=True,
            )

    # Nhận xét tổng hợp
    st.markdown("---")
    st.markdown("### 📝 Nhận xét so sánh")
    if old_metrics and fctl_metrics:
        winner_f1  = "Mô hình cũ" if old_metrics["f1_macro"] > fctl_metrics["f1_macro"] else "Mô hình FCT_L"
        winner_auc = "Mô hình cũ" if old_metrics["roc_auc"]  > fctl_metrics["roc_auc"]  else "Mô hình FCT_L"
        diff_f1  = abs(old_metrics["f1_macro"] - fctl_metrics["f1_macro"])
        diff_auc = abs(old_metrics["roc_auc"]  - fctl_metrics["roc_auc"])

        st.info(f"""
**Macro F1**: {winner_f1} tốt hơn ({diff_f1:.4f} điểm chênh lệch)

**ROC-AUC**: {winner_auc} tốt hơn ({diff_auc:.4f} điểm chênh lệch)

**Lưu ý quan trọng**: Hai mô hình sử dụng bộ đặc trưng và tập dữ liệu hoàn toàn khác nhau,
nên kết quả so sánh phản ánh sự khác biệt về **chất lượng dữ liệu** và **đặc trưng** hơn là
về thuật toán. Mô hình có Macro F1 cao hơn sẽ phân loại tốt hơn đặc biệt ở các nhóm nợ thiểu số (N2–N5).
        """)
    else:
        st.info("Cần có đủ cả hai mô hình để hiển thị nhận xét so sánh.")
