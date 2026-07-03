"""Trang 5 — Mô hình dự báo Nhóm Nợ trên tập dữ liệu fct_l.xlsx"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

from config.config import (
    MODEL_DIR, RANDOM_STATE, TEST_SIZE, SCORE_BANDS,
    DATA_FCT_L, DATA_FCT_L_PROCESSED,
    FCT_L_TARGET_COL, FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS,
    FCT_L_SMOTE_ORIGINAL, FCT_L_SMOTE_30K, FCT_L_NHOMNO_LABELS, GROUP_COLORS,
)
from src.fct_l_preprocessing import prepare_fct_l, load_fct_l, load_fct_l_processed
from src.models import build_pipeline, available_models, IMBALANCE_OPTIONS
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_feature_importance
from src.scoring import (
    build_score_df, plot_score_distribution,
    plot_score_by_group, proba_to_score, classify_score,
)

st.set_page_config(page_title="Mô hình FCT_L", page_icon="📊", layout="wide")
st.title("📊 Mô hình Dự báo Nhóm Nợ — FCT_L")
st.markdown("Xây dựng và huấn luyện mô hình phân loại nhóm nợ **1–5** trên tập dữ liệu `fct_l.xlsx`.")


# ── Cache helpers ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dữ liệu fct_l…")
def _load_original() -> pd.DataFrame:
    if DATA_FCT_L.exists():
        return pd.read_excel(DATA_FCT_L)
    return pd.DataFrame()


@st.cache_data(show_spinner="Đang tải fct_l_30k…")
def _load_processed() -> pd.DataFrame:
    if DATA_FCT_L_PROCESSED.exists():
        return pd.read_csv(DATA_FCT_L_PROCESSED)
    return pd.DataFrame()


@st.cache_resource(show_spinner="Đang tải mô hình FCT_L…")
def _load_saved_model(key: str):
    p = MODEL_DIR / f"model_fct_l_{key}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    data_source = st.radio(
        "Nguồn dữ liệu",
        ["fct_l_30k.csv (30,000 hồ sơ — khuyến nghị)", "fct_l.xlsx (5,400 hồ sơ gốc)"],
        index=0,
        help="Tập 30k được tạo bằng phương pháp Stratified Bootstrap từ dữ liệu gốc, giữ nguyên tỉ lệ nhóm nợ.",
    )
    use_30k = "30k" in data_source

    MODEL_OPTIONS = available_models()
    model_label = st.selectbox(
        "Chọn mô hình",
        options=list(MODEL_OPTIONS.keys()),
        index=1,  # Random Forest mặc định
    )
    model_key = MODEL_OPTIONS[model_label]

    st.markdown("---")
    st.markdown("**Phân phối nhóm nợ (CLASSIFICATION)**")
    dist_info = {
        "fct_l.xlsx": {"N1": "5,247 (97.2%)", "N2": "87 (1.6%)", "N3": "11 (0.2%)", "N4": "13 (0.2%)", "N5": "23 (0.4%)"},
        "fct_l_30k" : {"N1": "~29,151 (97.2%)", "N2": "~483 (1.6%)", "N3": "~60 (0.2%)", "N4": "~72 (0.2%)", "N5": "~129 (0.4%)"},
    }
    src_key = "fct_l_30k" if use_30k else "fct_l.xlsx"
    for g, v in dist_info[src_key].items():
        color = list(GROUP_COLORS.values())[int(g[1]) - 1]
        st.markdown(
            f"<span style='background:{color};color:#fff;padding:1px 6px;border-radius:3px'>{g}</span> {v}",
            unsafe_allow_html=True,
        )

train_btn = st.button("🚀 Huấn luyện mô hình", type="primary")

# ── Load data ─────────────────────────────────────────────────────────────────
df_raw = _load_processed() if use_30k else _load_original()

if df_raw.empty:
    st.error("Không tìm thấy file dữ liệu. Kiểm tra lại đường dẫn.")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_eda, tab_train, tab_score = st.tabs(
    ["📋 Tổng quan dữ liệu", "🤖 Huấn luyện & Đánh giá", "💳 Chấm điểm tín dụng"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    df_valid = df_raw.dropna(subset=[FCT_L_TARGET_COL]).copy()
    df_valid[FCT_L_TARGET_COL] = df_valid[FCT_L_TARGET_COL].astype(int)

    r, c = df_raw.shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số hồ sơ",     f"{r:,}")
    m2.metric("Số cột",       c)
    m3.metric("Hồ sơ hợp lệ", f"{len(df_valid):,}")
    m4.metric("Số nhóm nợ",   df_valid[FCT_L_TARGET_COL].nunique())

    # Class distribution
    st.markdown("#### Phân phối nhóm nợ (CLASSIFICATION)")
    vc = df_valid[FCT_L_TARGET_COL].value_counts().sort_index()
    fig_cls = go.Figure(go.Bar(
        x=[FCT_L_NHOMNO_LABELS.get(k, str(k)) for k in vc.index],
        y=vc.values,
        marker_color=[GROUP_COLORS.get(k, "#999") for k in vc.index],
        text=[f"{v:,} ({v/len(df_valid)*100:.1f}%)" for v in vc.values],
        textposition="auto",
    ))
    fig_cls.update_layout(
        xaxis_title="Nhóm nợ", yaxis_title="Số lượng",
        height=360,
        annotations=[{
            "x": 0.5, "y": 1.08, "xref": "paper", "yref": "paper",
            "text": f"<b>Imbalance ratio: {vc.iloc[0] / vc.iloc[-1]:.0f}x</b>",
            "showarrow": False, "font": {"size": 13, "color": "red"},
        }],
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    # Feature availability
    st.markdown("#### Thống kê các biến đặc trưng được dùng")
    feat_cols = FCT_L_NUMERICAL_COLS + FCT_L_CATEGORICAL_COLS
    feat_df = pd.DataFrame({
        "Cột":     feat_cols,
        "Loại":    ["Số học"] * len(FCT_L_NUMERICAL_COLS) + ["Phân loại"] * len(FCT_L_CATEGORICAL_COLS),
        "Null (%)": [f"{df_raw[c].isnull().mean()*100:.1f}%" if c in df_raw.columns else "—" for c in feat_cols],
        "Unique":  [df_raw[c].nunique() if c in df_raw.columns else "—" for c in feat_cols],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # Numeric distributions per group
    st.markdown("#### Phân phối biến số theo nhóm nợ")
    num_avail = [c for c in FCT_L_NUMERICAL_COLS if c in df_valid.columns]
    sel_feat = st.selectbox("Chọn biến", num_avail, key="eda_feat")
    fig_box = px.box(
        df_valid.assign(Nhóm=df_valid[FCT_L_TARGET_COL].map(lambda x: f"Nhóm {x}")),
        x="Nhóm", y=sel_feat,
        color="Nhóm",
        title=f"Phân phối {sel_feat} theo nhóm nợ",
    )
    fig_box.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Train khi nhấn button
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    with st.spinner(f"Đang chuẩn bị dữ liệu và huấn luyện {model_label}…"):
        try:
            X_all, y_all = prepare_fct_l(
                df_raw, FCT_L_TARGET_COL,
                FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS,
            )

            cat_p = [c for c in FCT_L_CATEGORICAL_COLS if c in X_all.columns]
            num_p = [c for c in FCT_L_NUMERICAL_COLS   if c in X_all.columns]

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                random_state=RANDOM_STATE,
            )

            smote_strategy = FCT_L_SMOTE_30K if use_30k else FCT_L_SMOTE_ORIGINAL

            pipe = build_pipeline(
                model_key, cat_p, num_p,
                random_state=RANDOM_STATE,
                imbalance_strategy="custom",
                custom_smote_strategy=smote_strategy,
            )
            pipe.fit(X_train, y_train)

            y_pred  = pipe.predict(X_test) + 1   # 1-based
            y_proba = pipe.predict_proba(X_test)
            y_true  = y_test + 1

            metrics = compute_metrics(y_true, y_pred, y_proba)

            # Lưu model
            model_path = MODEL_DIR / f"model_fct_l_{model_key}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(pipe, f)
            # Lưu tên model tốt nhất
            (MODEL_DIR / "best_fct_l_model.txt").write_text(model_key)

            st.session_state["fct_l_pipe"]     = pipe
            st.session_state["fct_l_y_true"]   = y_true
            st.session_state["fct_l_y_pred"]   = y_pred
            st.session_state["fct_l_y_proba"]  = y_proba
            st.session_state["fct_l_metrics"]  = metrics
            st.session_state["fct_l_model_lbl"]= model_label
            st.session_state["fct_l_model_key"]= model_key
            _load_saved_model.clear()

            st.success(
                f"✅ Hoàn tất! **{model_label}** — "
                f"Macro F1: **{metrics['f1_macro']:.4f}** | "
                f"ROC-AUC: **{metrics['roc_auc']:.4f}**"
            )
        except Exception as e:
            st.error(f"Lỗi huấn luyện: {e}")

# ── Tải kết quả nếu đã có model lưu (chưa train trong session này)
if "fct_l_pipe" not in st.session_state:
    best_txt = MODEL_DIR / "best_fct_l_model.txt"
    if best_txt.exists():
        saved_key = best_txt.read_text().strip()
        saved_pipe = _load_saved_model(saved_key)
        if saved_pipe is not None:
            # Chạy lại prediction trên original data để hiển thị
            try:
                X_all, y_all = prepare_fct_l(
                    _load_original() if not use_30k else _load_processed(),
                    FCT_L_TARGET_COL, FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS,
                )
                _, X_test, _, y_test = train_test_split(
                    X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                    random_state=RANDOM_STATE,
                )
                y_pred  = saved_pipe.predict(X_test) + 1
                y_proba = saved_pipe.predict_proba(X_test)
                y_true  = y_test + 1
                metrics = compute_metrics(y_true, y_pred, y_proba)

                st.session_state["fct_l_pipe"]     = saved_pipe
                st.session_state["fct_l_y_true"]   = y_true
                st.session_state["fct_l_y_pred"]   = y_pred
                st.session_state["fct_l_y_proba"]  = y_proba
                st.session_state["fct_l_metrics"]  = metrics
                st.session_state["fct_l_model_key"]= saved_key
                saved_lbl = next((l for l, k in available_models().items() if k == saved_key), saved_key)
                st.session_state["fct_l_model_lbl"]= saved_lbl
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Huấn luyện & Đánh giá
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if "fct_l_metrics" not in st.session_state:
        st.info("👈 Nhấn **Huấn luyện mô hình** ở thanh bên để bắt đầu.")
    else:
        metrics  = st.session_state["fct_l_metrics"]
        y_true   = st.session_state["fct_l_y_true"]
        y_pred   = st.session_state["fct_l_y_pred"]
        y_proba  = st.session_state["fct_l_y_proba"]
        lbl      = st.session_state.get("fct_l_model_lbl", "")

        st.success(f"✅ Mô hình: **{lbl}**")

        c1, c2, c3 = st.columns(3)
        c1.metric("Macro F1",    f"{metrics['f1_macro']:.4f}")
        c2.metric("Weighted F1", f"{metrics['f1_weighted']:.4f}")
        c3.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(y_true, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_b:
            st.markdown("**Classification Report**")
            report_df = pd.DataFrame(metrics.get("report", {})).T
            num_cols_r = report_df.select_dtypes(include=float).columns
            st.dataframe(
                report_df.style.format({c: "{:.3f}" for c in num_cols_r}),
                use_container_width=True,
            )

        # Feature importance
        pipe = st.session_state.get("fct_l_pipe")
        if pipe:
            fig_fi = plot_feature_importance(pipe, top_n=16)
            if fig_fi:
                st.markdown("**Feature Importance**")
                st.plotly_chart(fig_fi, use_container_width=True)

        # Xác suất trung bình mỗi nhóm
        st.markdown("#### Xác suất dự báo trung bình theo nhóm thực tế")
        proba_by_group = []
        for g in sorted(np.unique(y_true)):
            mask = y_true == g
            row = {"Nhóm thực tế": f"Nhóm {g}"}
            for i in range(5):
                row[f"P(Nhóm {i+1})"] = y_proba[mask, i].mean().round(4)
            proba_by_group.append(row)
        st.dataframe(pd.DataFrame(proba_by_group), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Chấm điểm tín dụng
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    if "fct_l_y_proba" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để xem kết quả chấm điểm.")
    else:
        y_true  = st.session_state["fct_l_y_true"]
        y_pred  = st.session_state["fct_l_y_pred"]
        y_proba = st.session_state["fct_l_y_proba"]

        df_scores = build_score_df(y_proba, y_pred, y_true)

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số hồ sơ",        f"{len(df_scores):,}")
        c2.metric("Điểm trung bình", f"{df_scores['diem_tin_dung'].mean():.0f}")
        c3.metric("Điểm thấp nhất",  f"{df_scores['diem_tin_dung'].min()}")
        c4.metric("Điểm cao nhất",   f"{df_scores['diem_tin_dung'].max()}")

        fig_dist = plot_score_distribution(df_scores)
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_box = plot_score_by_group(df_scores)
        if fig_box:
            st.plotly_chart(fig_box, use_container_width=True)

        # Bảng thống kê theo nhóm
        st.markdown("#### Điểm trung bình theo nhóm nợ thực tế")
        tbl = (df_scores.groupby("nhom_thuc_te")["diem_tin_dung"]
                        .agg(["mean", "median", "std", "min", "max"])
                        .round(1))
        tbl.index = [f"Nhóm {i}" for i in tbl.index]
        st.dataframe(tbl, use_container_width=True)

        # Phân bổ hạng
        st.markdown("#### Phân bổ hạng tín dụng")
        grade_order = [b[2] for b in SCORE_BANDS]
        vc_g = df_scores["hang"].value_counts().reindex(grade_order, fill_value=0)
        grade_tbl = pd.DataFrame({
            "Hạng":   vc_g.index,
            "Mô tả":  [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc_g.index],
            "Số lượng": vc_g.values,
            "Tỷ lệ (%)": (vc_g.values / len(df_scores) * 100).round(1),
        })
        st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

        # Download
        csv = df_scores.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=csv,
            file_name=f"credit_scores_fct_l_{st.session_state.get('fct_l_model_key','model')}.csv",
            mime="text/csv",
        )
