"""Trang 1 — Phân tích khám phá dữ liệu (EDA)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.config import DATA_RAW, TARGET_COL, NHOMNO_LABELS, GROUP_COLORS
from src.preprocessing import parse_dates, engineer_features
from src.data_loader import get_raw_bytes
from src.iv_analysis import compute_iv_table, plot_iv_bar, plot_woe_bins

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Phân tích khám phá dữ liệu")


@st.cache_data(show_spinner="Đang tải dữ liệu…")
def load_data(raw_bytes: bytes):
    import io
    df = pd.read_excel(io.BytesIO(raw_bytes))
    df2 = parse_dates(df)
    df2 = engineer_features(df2)
    return df, df2


_raw_bytes = get_raw_bytes(DATA_RAW)
df_raw, df = load_data(_raw_bytes)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Tổng quan", "Phân phối biến", "Phân tích nhóm nợ", "Tương quan", "📊 IV Analysis"]
)

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Thông tin dữ liệu")
    r, c = df_raw.shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số hồ sơ",   f"{r:,}")
    m2.metric("Số biến",    c)
    m3.metric("Giá trị thiếu", f"{df_raw.isnull().sum().sum():,}")
    m4.metric("Số nhóm nợ", df_raw[TARGET_COL].nunique())

    st.markdown("#### Mô tả các cột")
    info = pd.DataFrame({
        "Cột":    df_raw.columns,
        "Kiểu":   df_raw.dtypes.astype(str).values,
        "Null":   df_raw.isnull().sum().values,
        "Unique": df_raw.nunique().values,
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

    st.markdown("#### 5 hàng đầu")
    st.dataframe(df_raw.head(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Phân phối nhóm nợ (Target)")
    vc     = df_raw[TARGET_COL].value_counts().sort_index()
    labels = [NHOMNO_LABELS.get(k, str(k)) for k in vc.index]
    colors = [GROUP_COLORS[k] for k in vc.index]

    fig_target = go.Figure(go.Bar(
        x=labels, y=vc.values,
        marker_color=colors,
        text=[f"{v:,}<br>({v/len(df_raw)*100:.1f}%)" for v in vc.values],
        textposition="auto",
    ))
    fig_target.update_layout(title="Phân phối nhóm nợ NHOMNOMOI",
                              xaxis_title="Nhóm nợ", yaxis_title="Số lượng",
                              height=380)
    st.plotly_chart(fig_target, use_container_width=True)

    st.markdown("---")
    st.subheader("Phân phối biến số")

    num_options = ["BASE_BAL", "CURR_BAL", "DUNO_QD", "LAISUAT",
                   "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY", "UTIL_RATE"]
    num_options = [c for c in num_options if c in df.columns]

    col_sel = st.selectbox("Chọn biến", num_options)
    log_scale = st.checkbox("Log scale (cho biến lệch phải)")

    plot_data = np.log1p(df[col_sel].dropna()) if log_scale else df[col_sel].dropna()
    title_suffix = " (log1p)" if log_scale else ""

    fig_hist = px.histogram(plot_data, nbins=50,
                             title=f"Phân phối {col_sel}{title_suffix}",
                             labels={"value": col_sel})
    fig_hist.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")
    st.subheader("Phân phối biến phân loại")
    cat_options = ["SEX", "MJACCTTYPCD", "MJACCTTYPDESC", "DESC_TIME", "LOAIKH"]
    cat_options = [c for c in cat_options if c in df_raw.columns]
    cat_sel = st.selectbox("Chọn biến phân loại", cat_options)

    vc2 = df_raw[cat_sel].value_counts().head(20)
    fig_cat = px.bar(x=vc2.index.astype(str), y=vc2.values,
                     labels={"x": cat_sel, "y": "Số lượng"},
                     title=f"Phân phối {cat_sel} (top 20)")
    fig_cat.update_layout(height=380)
    st.plotly_chart(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Phân tích theo nhóm nợ")

    metric_sel = st.selectbox(
        "Chọn biến phân tích",
        ["LAISUAT", "CURR_BAL", "BASE_BAL", "UTIL_RATE", "LOAN_TENURE_DAYS"],
        key="grp_metric"
    )

    # Boxplot
    fig_box = go.Figure()
    for g, color in GROUP_COLORS.items():
        vals = df.loc[df[TARGET_COL] == g, metric_sel].dropna()
        fig_box.add_trace(go.Box(
            y=vals, name=NHOMNO_LABELS[g],
            marker_color=color, boxpoints="outliers",
        ))
    fig_box.update_layout(title=f"{metric_sel} theo nhóm nợ",
                           yaxis_title=metric_sel, height=420)
    st.plotly_chart(fig_box, use_container_width=True)

    # Bảng thống kê
    st.markdown("#### Thống kê mô tả theo nhóm")
    tbl = (df.groupby(TARGET_COL)[metric_sel]
             .agg(["mean", "median", "std", "min", "max"])
             .round(2))
    tbl.index = [NHOMNO_LABELS.get(i, i) for i in tbl.index]
    st.dataframe(tbl, use_container_width=True)

    st.markdown("---")
    st.subheader("Phân bổ SEX theo nhóm nợ")
    if "SEX" in df_raw.columns:
        # Bỏ NaN trong SEX trước khi crosstab để tránh cột NaN trong Plotly
        df_sex = df_raw.dropna(subset=["SEX"]).copy()
        cross = (pd.crosstab(df_sex[TARGET_COL], df_sex["SEX"], normalize="index") * 100).round(1)
        # reset_index trước, rồi mới đổi nhãn để tên cột TARGET_COL còn nguyên
        cross = cross.reset_index()
        cross[TARGET_COL] = cross[TARGET_COL].map(lambda x: NHOMNO_LABELS.get(x, str(x)))
        sex_cols = [c for c in cross.columns if c != TARGET_COL]
        fig_sex = px.bar(cross, x=TARGET_COL, y=sex_cols,
                         barmode="stack",
                         title="Tỷ lệ giới tính theo nhóm nợ (%)")
        fig_sex.update_layout(height=380)
        st.plotly_chart(fig_sex, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Ma trận tương quan")
    num_cols = ["BASE_BAL", "CURR_BAL", "DUNO_QD", "LAISUAT",
                "LOAIKH", "NHOMNO", TARGET_COL,
                "LOAN_TENURE_DAYS", "UTIL_RATE"]
    num_cols = [c for c in num_cols if c in df.columns]

    corr = df[num_cols].corr().round(2)
    fig_corr = px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
        title="Correlation Heatmap",
    )
    fig_corr.update_layout(height=550)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("#### Tương quan với NHOMNOMOI")
    corr_target = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    fig_ct = px.bar(
        x=corr_target.values, y=corr_target.index,
        orientation="h", color=corr_target.values,
        color_continuous_scale="RdBu_r", range_color=[-1, 1],
        title=f"Tương quan với {TARGET_COL}",
        labels={"x": "Correlation", "y": "Feature"},
    )
    fig_ct.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_ct, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Information Value (IV) — Chuẩn Basel II/III")
    st.markdown("""
    IV đo khả năng phân biệt **good** (nhóm 1) vs **bad** (nhóm 2–5) của từng feature.
    Áp dụng được cho cả biến **categorical** lẫn **numerical** — khắc phục giới hạn của Correlation.

    | IV | Mức độ |
    |----|--------|
    | < 0.02 | Useless — loại khỏi model |
    | 0.02–0.1 | Weak |
    | 0.1–0.3 | Medium |
    | 0.3–0.5 | Strong |
    | 0.5–1.0 | Very strong |
    | > 1.0 | ⚠️ Nghi ngờ leakage |
    """)

    with st.spinner("Đang tính IV…"):
        iv_tbl = compute_iv_table(df_raw, TARGET_COL, good_class=1)

    fig_iv = plot_iv_bar(iv_tbl, threshold=0.02)
    st.plotly_chart(fig_iv, use_container_width=True)

    st.markdown("#### Bảng IV chi tiết")
    st.dataframe(
        iv_tbl.style.background_gradient(subset=["iv"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")
    st.markdown("#### WoE chi tiết theo bin")
    woe_feat = st.selectbox(
        "Chọn feature để xem WoE",
        options=iv_tbl["feature"].tolist(),
        index=0,
    )
    fig_woe = plot_woe_bins(df_raw, woe_feat, TARGET_COL)
    st.plotly_chart(fig_woe, use_container_width=True)
