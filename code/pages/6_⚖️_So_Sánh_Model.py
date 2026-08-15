"""Trang 6 — So sánh 2 mô hình song song với dữ liệu upload riêng"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from config.config import (
    DATA_RAW, MODEL_DIR,
    TARGET_COL, DROP_COLS, SCORE_BANDS, GROUP_COLORS, NHOMNO_LABELS,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_NUMERICAL_COLS, CREDIT_INFO_CATEGORICAL_COLS,
)
from src.preprocessing import parse_dates, engineer_features, clean
from src.credit_info_preprocessing import transform_credit_info, engineer_business_features
from src.models import available_models
from src.scoring import proba_to_score, classify_score
from src.evaluation import compute_metrics, plot_confusion_matrix

st.set_page_config(page_title="So sánh 2 Mô hình", page_icon="⚖️", layout="wide")
st.title("⚖️ So sánh Song Song 2 Mô hình")
st.markdown(
    "Upload dữ liệu riêng cho từng mô hình → chạy dự báo song song → "
    "so sánh điểm tín dụng & hiệu năng."
)

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource
def _load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def _read_file(raw_bytes: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    return pd.read_csv(buf, low_memory=False) if name.endswith(".csv") else pd.read_excel(buf)


def _preprocess_a(df: pd.DataFrame):
    """Tiền xử lý cho Mô hình A (Data_credit_rating_VN format)."""
    has_target = TARGET_COL in df.columns
    df2 = parse_dates(df.copy())
    df2 = engineer_features(df2)
    y_true = df2[TARGET_COL].values.astype(int) if has_target else None
    X = clean(df2, drop_cols=DROP_COLS + ([TARGET_COL] if has_target else []))
    return X, y_true


def _preprocess_b(df: pd.DataFrame):
    """Tiền xử lý cho Mô hình B (dữ liệu Thông tin tín dụng)."""
    has_target = CREDIT_INFO_TARGET_COL in df.columns
    df_eng, eng_cols = engineer_business_features(df)
    y_true = df_eng[CREDIT_INFO_TARGET_COL].dropna().astype(int).values if has_target else None
    X = transform_credit_info(df_eng, CREDIT_INFO_NUMERICAL_COLS + eng_cols, CREDIT_INFO_CATEGORICAL_COLS)
    return X, y_true


def _score_result(pipe, X, y_true=None):
    """Chạy dự báo và tính điểm. Trả về dict kết quả."""
    pred   = pipe.predict(X) + 1          # 1-based
    proba  = pipe.predict_proba(X)
    scores = proba_to_score(proba)
    grades = [classify_score(s) for s in scores]

    result = {
        "pred":   pred,
        "proba":  proba,
        "scores": scores,
        "grades": [g[0] for g in grades],
        "n":      len(pred),
    }
    if y_true is not None and len(y_true) == len(pred):
        result["metrics"] = compute_metrics(y_true, pred, proba)
        result["y_true"]  = y_true
    return result


def _col_header(title: str, color: str, subtitle: str):
    st.markdown(
        f"<div style='background:{color};color:#fff;padding:8px 14px;"
        f"border-radius:8px;margin-bottom:8px'>"
        f"<b style='font-size:1.1rem'>{title}</b><br>"
        f"<span style='font-size:0.82rem;opacity:.9'>{subtitle}</span></div>",
        unsafe_allow_html=True,
    )


# ── Model availability ────────────────────────────────────────────────────────
avail_a = {l: k for l, k in available_models().items()
           if (MODEL_DIR / f"model_{k}.pkl").exists()}
avail_b = {l: k for l, k in available_models().items()
           if (MODEL_DIR / f"model_credit_info_{k}.pkl").exists()}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Cấu hình & Upload song song
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1️⃣  Cấu hình & Upload dữ liệu")
col_a, col_sep, col_b = st.columns([10, 1, 10])

with col_a:
    _col_header("🔵 Mô hình A", "#2980b9", "Data_credit_rating_VN.xlsx — NHOMNOMOI")

    if avail_a:
        a_lbl = st.selectbox("Chọn mô hình A", list(avail_a.keys()), key="sel_a")
        a_key = avail_a[a_lbl]
    else:
        st.warning("Chưa có mô hình A. Huấn luyện ở **Trang 2**.")
        a_lbl = a_key = None

    up_a = st.file_uploader(
        "Upload dữ liệu cho Mô hình A",
        type=["xlsx", "csv"],
        key="up_a",
        help="Cần có cột: MJACCTTYPCD, CURRMIACCTTYPCD, MUCDICHVAY, BASE_BAL, CURR_BAL, "
             "LAISUAT, ORGNBR, PARENTORGNBR, OPEN_DATE, NGAYDENHAN. "
             "Tùy chọn: NHOMNOMOI (để tính độ chính xác).",
    )
    if up_a:
        st.caption(f"✅ {up_a.name} — {up_a.size/1024:.0f} KB")

    # Template download
    if DATA_RAW.exists():
        @st.cache_data
        def _tpl_a():
            df = pd.read_excel(DATA_RAW)
            cols_keep = [c for c in df.columns if c not in ["NHOMNO_TCBS"]]
            buf = io.BytesIO()
            df[cols_keep].head(5).to_excel(buf, index=False)
            return buf.getvalue()
        st.download_button("⬇️ Template Mô hình A", _tpl_a(),
                           "template_model_a.xlsx", key="tpl_a")

with col_sep:
    st.markdown(
        "<div style='display:flex;align-items:center;justify-content:center;"
        "height:200px;font-size:2rem;color:#aaa'>VS</div>",
        unsafe_allow_html=True,
    )

with col_b:
    _col_header("🟠 Mô hình B", "#e67e22", "Thông tin tín dụng — Nhóm nợ tự phân loại")

    if avail_b:
        b_lbl = st.selectbox("Chọn mô hình B", list(avail_b.keys()), key="sel_b")
        b_key = avail_b[b_lbl]
    else:
        st.warning("Chưa có mô hình B. Huấn luyện ở **Trang 5**.")
        b_lbl = b_key = None

    up_b = st.file_uploader(
        "Upload dữ liệu cho Mô hình B",
        type=["xlsx", "csv"],
        key="up_b",
        help="Cần có các cột của file Thông tin tín dụng (vd. Lãi suất, Số dư nợ "
             "theo nguyên tệ, Mã chi nhánh TCTD, Hình thức cấp tín dụng, Phương "
             "thức cho vay, Mã tiền tệ, Mục đích sử dụng tiền vay…, ETL_DATE, "
             "Ngày giải ngân, Ngày kết thúc khế ước). "
             "Tùy chọn: Nhóm nợ tự phân loại (để tính độ chính xác).",
    )
    if up_b:
        st.caption(f"✅ {up_b.name} — {up_b.size/1024:.0f} KB")

    # Template download từ session state
    if st.session_state.get("credit_info_pipe") is not None:
        st.caption("💡 Dùng dữ liệu Thông tin tín dụng làm template cho Mô hình B.")

# ── Nút So sánh ───────────────────────────────────────────────────────────────
st.markdown("---")
ready_a = a_key is not None and up_a is not None
ready_b = b_key is not None and up_b is not None
can_run = ready_a or ready_b

run_btn = st.button(
    "🔄 Chạy So sánh",
    type="primary",
    disabled=not can_run,
    help="Cần chọn ít nhất 1 mô hình và upload file tương ứng.",
)

if not can_run:
    st.info("👆 Upload file cho ít nhất một mô hình và nhấn **Chạy So sánh**.")

# ══════════════════════════════════════════════════════════════════════════════
# Chạy dự báo
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    res_a, res_b = None, None
    err_a, err_b = None, None

    with st.spinner("Đang dự báo…"):
        # Model A
        if ready_a:
            try:
                pipe_a = _load_model(str(MODEL_DIR / f"model_{a_key}.pkl"))
                df_a   = _read_file(up_a.getvalue(), up_a.name)
                X_a, y_a = _preprocess_a(df_a)
                res_a  = _score_result(pipe_a, X_a, y_a)
            except Exception as e:
                err_a = str(e)

        # Model B
        if ready_b:
            try:
                pipe_b = _load_model(str(MODEL_DIR / f"model_credit_info_{b_key}.pkl"))
                df_b   = _read_file(up_b.getvalue(), up_b.name)
                X_b, y_b = _preprocess_b(df_b)
                res_b  = _score_result(pipe_b, X_b, y_b)
            except Exception as e:
                err_b = str(e)

    st.session_state["cmp_res_a"]  = res_a
    st.session_state["cmp_res_b"]  = res_b
    st.session_state["cmp_lbl_a"]  = a_lbl
    st.session_state["cmp_lbl_b"]  = b_lbl
    st.session_state["cmp_err_a"]  = err_a
    st.session_state["cmp_err_b"]  = err_b

# ── Lấy kết quả từ session ───────────────────────────────────────────────────
res_a  = st.session_state.get("cmp_res_a")
res_b  = st.session_state.get("cmp_res_b")
lbl_a  = st.session_state.get("cmp_lbl_a", a_lbl or "Mô hình A")
lbl_b  = st.session_state.get("cmp_lbl_b", b_lbl or "Mô hình B")
err_a  = st.session_state.get("cmp_err_a")
err_b  = st.session_state.get("cmp_err_b")

if err_a:
    st.error(f"🔵 Mô hình A lỗi: {err_a}")
if err_b:
    st.error(f"🟠 Mô hình B lỗi: {err_b}")

if res_a is None and res_b is None:
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Kết quả So sánh
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 2️⃣  Kết quả So sánh")

tab_overview, tab_score, tab_groups, tab_cm, tab_detail = st.tabs([
    "📊 Tổng quan",
    "💳 Điểm tín dụng",
    "📋 Nhóm nợ dự báo",
    "🔢 Confusion Matrix",
    "📥 Tải kết quả",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: Tổng quan (KPIs + metrics nếu có ground truth)
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    col_a, col_b = st.columns(2)

    with col_a:
        _col_header(f"🔵 {lbl_a}", "#2980b9", "Mô hình A")
        if res_a:
            c1, c2, c3 = st.columns(3)
            c1.metric("Số hồ sơ",  f"{res_a['n']:,}")
            c2.metric("Điểm TB",   f"{res_a['scores'].mean():.0f}")
            c3.metric("Điểm Min",  f"{res_a['scores'].min()}")

            if "metrics" in res_a:
                m = res_a["metrics"]
                ca, cb, cc = st.columns(3)
                ca.metric("Macro F1",    f"{m['f1_macro']:.4f}")
                cb.metric("Weighted F1", f"{m['f1_weighted']:.4f}")
                cc.metric("ROC-AUC",     f"{m['roc_auc']:.4f}")
            else:
                st.caption("📌 Không có cột nhãn thực → chỉ hiển thị phân phối dự báo.")
        else:
            st.info("Chưa có kết quả.")

    with col_b:
        _col_header(f"🟠 {lbl_b}", "#e67e22", "Mô hình B")
        if res_b:
            c1, c2, c3 = st.columns(3)
            c1.metric("Số hồ sơ",  f"{res_b['n']:,}")
            c2.metric("Điểm TB",   f"{res_b['scores'].mean():.0f}")
            c3.metric("Điểm Min",  f"{res_b['scores'].min()}")

            if "metrics" in res_b:
                m = res_b["metrics"]
                ca, cb, cc = st.columns(3)
                ca.metric("Macro F1",    f"{m['f1_macro']:.4f}")
                cb.metric("Weighted F1", f"{m['f1_weighted']:.4f}")
                cc.metric("ROC-AUC",     f"{m['roc_auc']:.4f}")
            else:
                st.caption("📌 Không có cột nhãn thực → chỉ hiển thị phân phối dự báo.")
        else:
            st.info("Chưa có kết quả.")

    # Bảng so sánh metrics (nếu cả hai đều có ground truth)
    if res_a and res_b and "metrics" in res_a and "metrics" in res_b:
        st.markdown("---")
        st.markdown("#### So sánh hiệu năng")
        rows = [
            {"Chỉ số": "Macro F1",    f"🔵 {lbl_a}": res_a["metrics"]["f1_macro"],
             f"🟠 {lbl_b}": res_b["metrics"]["f1_macro"]},
            {"Chỉ số": "Weighted F1", f"🔵 {lbl_a}": res_a["metrics"]["f1_weighted"],
             f"🟠 {lbl_b}": res_b["metrics"]["f1_weighted"]},
            {"Chỉ số": "ROC-AUC",     f"🔵 {lbl_a}": res_a["metrics"]["roc_auc"],
             f"🟠 {lbl_b}": res_b["metrics"]["roc_auc"]},
        ]
        df_cmp = pd.DataFrame(rows).set_index("Chỉ số")
        st.dataframe(df_cmp.style.format("{:.4f}").highlight_max(axis=1, color="#d4edda"),
                     use_container_width=True)

        # Bar chart
        fig_bar = go.Figure()
        col_names = [f"🔵 {lbl_a}", f"🟠 {lbl_b}"]
        colors    = ["#2980b9", "#e67e22"]
        for metric in ["Macro F1", "Weighted F1", "ROC-AUC"]:
            fig_bar.add_trace(go.Bar(
                name=metric,
                x=col_names,
                y=[res_a["metrics"][m] for m in ["f1_macro", "f1_weighted", "roc_auc"]]
                  [[("Macro F1", "Weighted F1", "ROC-AUC").index(metric)]],
                text=[f"{v:.4f}" for v in [
                    [res_a["metrics"]["f1_macro"], res_a["metrics"]["f1_weighted"], res_a["metrics"]["roc_auc"]],
                    [res_b["metrics"]["f1_macro"], res_b["metrics"]["f1_weighted"], res_b["metrics"]["roc_auc"]],
                ]],
            ))

        # Simple grouped bar chart
        metrics_list = ["Macro F1", "Weighted F1", "ROC-AUC"]
        vals_a = [res_a["metrics"]["f1_macro"], res_a["metrics"]["f1_weighted"], res_a["metrics"]["roc_auc"]]
        vals_b = [res_b["metrics"]["f1_macro"], res_b["metrics"]["f1_weighted"], res_b["metrics"]["roc_auc"]]
        fig_cmp = go.Figure(data=[
            go.Bar(name=f"🔵 {lbl_a}", x=metrics_list, y=vals_a,
                   marker_color="#2980b9",
                   text=[f"{v:.4f}" for v in vals_a], textposition="auto"),
            go.Bar(name=f"🟠 {lbl_b}", x=metrics_list, y=vals_b,
                   marker_color="#e67e22",
                   text=[f"{v:.4f}" for v in vals_b], textposition="auto"),
        ])
        fig_cmp.update_layout(barmode="group", yaxis=dict(range=[0, 1]),
                              height=380, title="So sánh hiệu năng 2 mô hình")
        st.plotly_chart(fig_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Phân phối điểm tín dụng
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    st.subheader("Phân phối điểm tín dụng [300 – 850]")

    # Histogram overlay
    fig_hist = go.Figure()
    if res_a:
        fig_hist.add_trace(go.Histogram(
            x=res_a["scores"], name=f"🔵 {lbl_a}",
            nbinsx=40, opacity=0.70, marker_color="#2980b9",
        ))
    if res_b:
        fig_hist.add_trace(go.Histogram(
            x=res_b["scores"], name=f"🟠 {lbl_b}",
            nbinsx=40, opacity=0.70, marker_color="#e67e22",
        ))
    for lo, _, grade, _, _ in SCORE_BANDS:
        fig_hist.add_vline(x=lo, line_dash="dash", line_color="#aaa", line_width=1,
                           annotation_text=grade, annotation_position="top right",
                           annotation_font_size=10)
    fig_hist.update_layout(barmode="overlay", height=400,
                           xaxis_title="Điểm tín dụng", yaxis_title="Số hồ sơ",
                           title="Phân phối điểm tín dụng — So sánh 2 mô hình")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Stats & grade table side by side
    col_a, col_b = st.columns(2)
    grade_order = [b[2] for b in SCORE_BANDS]

    with col_a:
        if res_a:
            st.markdown(f"**🔵 {lbl_a} — Thống kê điểm**")
            st.dataframe(pd.DataFrame({
                "": ["Trung bình", "Trung vị", "Thấp nhất", "Cao nhất"],
                "Điểm": [f"{res_a['scores'].mean():.0f}",
                         f"{np.median(res_a['scores']):.0f}",
                         f"{res_a['scores'].min()}",
                         f"{res_a['scores'].max()}"],
            }), use_container_width=True, hide_index=True)

            vc = pd.Series(res_a["grades"]).value_counts().reindex(grade_order, fill_value=0)
            grade_tbl = pd.DataFrame({
                "Hạng": vc.index,
                "Mô tả": [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc.index],
                "SL": vc.values,
                "%": (vc.values / res_a["n"] * 100).round(1),
            })
            st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

    with col_b:
        if res_b:
            st.markdown(f"**🟠 {lbl_b} — Thống kê điểm**")
            st.dataframe(pd.DataFrame({
                "": ["Trung bình", "Trung vị", "Thấp nhất", "Cao nhất"],
                "Điểm": [f"{res_b['scores'].mean():.0f}",
                         f"{np.median(res_b['scores']):.0f}",
                         f"{res_b['scores'].min()}",
                         f"{res_b['scores'].max()}"],
            }), use_container_width=True, hide_index=True)

            vc = pd.Series(res_b["grades"]).value_counts().reindex(grade_order, fill_value=0)
            grade_tbl = pd.DataFrame({
                "Hạng": vc.index,
                "Mô tả": [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc.index],
                "SL": vc.values,
                "%": (vc.values / res_b["n"] * 100).round(1),
            })
            st.dataframe(grade_tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Phân phối nhóm nợ dự báo
# ══════════════════════════════════════════════════════════════════════════════
with tab_groups:
    st.subheader("Phân phối nhóm nợ dự báo")

    col_a, col_b = st.columns(2)

    def _group_bar(res, title, color_base):
        if res is None:
            return
        vc = pd.Series(res["pred"]).value_counts().sort_index()
        colors = [GROUP_COLORS.get(k, color_base) for k in vc.index]
        fig = go.Figure(go.Bar(
            x=[f"Nhóm {k}" for k in vc.index],
            y=vc.values,
            marker_color=colors,
            text=[f"{v:,}<br>({v/res['n']*100:.1f}%)" for v in vc.values],
            textposition="auto",
        ))
        fig.update_layout(title=title, xaxis_title="Nhóm nợ",
                          yaxis_title="Số hồ sơ", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with col_a:
        _group_bar(res_a, f"🔵 {lbl_a}", "#2980b9")
        if res_a and "y_true" in res_a:
            st.markdown("**Thực tế vs Dự báo (Nhóm nợ)**")
            ct = pd.crosstab(
                pd.Series(res_a["y_true"], name="Thực tế"),
                pd.Series(res_a["pred"],   name="Dự báo"),
            )
            st.dataframe(ct, use_container_width=True)

    with col_b:
        _group_bar(res_b, f"🟠 {lbl_b}", "#e67e22")
        if res_b and "y_true" in res_b:
            st.markdown("**Thực tế vs Dự báo (Nhóm nợ)**")
            ct = pd.crosstab(
                pd.Series(res_b["y_true"], name="Thực tế"),
                pd.Series(res_b["pred"],   name="Dự báo"),
            )
            st.dataframe(ct, use_container_width=True)

    # So sánh xác suất trung bình mỗi nhóm
    if res_a and res_b:
        st.markdown("---")
        st.markdown("#### Xác suất dự báo trung bình mỗi nhóm")
        col_a2, col_b2 = st.columns(2)

        with col_a2:
            st.caption(f"🔵 {lbl_a}")
            avg_p = res_a["proba"].mean(axis=0)
            fig_p = px.bar(
                x=[f"Nhóm {i+1}" for i in range(5)], y=avg_p,
                labels={"x": "Nhóm nợ", "y": "Xác suất TB"},
                color=avg_p, color_continuous_scale="Blues",
            )
            fig_p.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)

        with col_b2:
            st.caption(f"🟠 {lbl_b}")
            avg_p = res_b["proba"].mean(axis=0)
            fig_p = px.bar(
                x=[f"Nhóm {i+1}" for i in range(5)], y=avg_p,
                labels={"x": "Nhóm nợ", "y": "Xác suất TB"},
                color=avg_p, color_continuous_scale="Oranges",
            )
            fig_p.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_p, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab_cm:
    st.subheader("Confusion Matrix (chỉ hiển thị khi có nhãn thực tế)")
    col_a, col_b = st.columns(2)

    with col_a:
        if res_a and "y_true" in res_a:
            st.markdown(f"**🔵 {lbl_a}**")
            st.plotly_chart(plot_confusion_matrix(res_a["y_true"], res_a["pred"]),
                            use_container_width=True)
            if "metrics" in res_a:
                report_df = pd.DataFrame(res_a["metrics"].get("report", {})).T
                num_r = report_df.select_dtypes(include=float).columns
                st.dataframe(report_df.style.format({c: "{:.3f}" for c in num_r}),
                             use_container_width=True)
        else:
            st.info("Mô hình A: Upload file có cột `NHOMNOMOI` để xem Confusion Matrix.")

    with col_b:
        if res_b and "y_true" in res_b:
            st.markdown(f"**🟠 {lbl_b}**")
            st.plotly_chart(plot_confusion_matrix(res_b["y_true"], res_b["pred"]),
                            use_container_width=True)
            if "metrics" in res_b:
                report_df = pd.DataFrame(res_b["metrics"].get("report", {})).T
                num_r = report_df.select_dtypes(include=float).columns
                st.dataframe(report_df.style.format({c: "{:.3f}" for c in num_r}),
                             use_container_width=True)
        else:
            st.info("Mô hình B: Upload file có cột `Nhóm nợ tự phân loại` để xem Confusion Matrix.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Tải kết quả
# ══════════════════════════════════════════════════════════════════════════════
with tab_detail:
    st.subheader("Tải kết quả dự báo")
    col_a, col_b = st.columns(2)

    with col_a:
        if res_a:
            df_out_a = pd.DataFrame({
                "nhom_du_bao":   res_a["pred"],
                "diem_tin_dung": res_a["scores"],
                "hang":          res_a["grades"],
                **{f"xac_suat_nhom_{i+1}": res_a["proba"][:, i].round(4) for i in range(5)},
            })
            if "y_true" in res_a:
                df_out_a.insert(0, "nhom_thuc_te", res_a["y_true"])

            st.markdown(f"**🔵 {lbl_a}** — {len(df_out_a):,} hồ sơ")
            st.dataframe(df_out_a.head(20), use_container_width=True, hide_index=True)
            csv_a = df_out_a.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Tải kết quả Mô hình A (CSV)", csv_a,
                               "ket_qua_mo_hinh_A.csv", mime="text/csv")

    with col_b:
        if res_b:
            df_out_b = pd.DataFrame({
                "nhom_du_bao":   res_b["pred"],
                "diem_tin_dung": res_b["scores"],
                "hang":          res_b["grades"],
                **{f"xac_suat_nhom_{i+1}": res_b["proba"][:, i].round(4) for i in range(5)},
            })
            if "y_true" in res_b:
                df_out_b.insert(0, "nhom_thuc_te", res_b["y_true"])

            st.markdown(f"**🟠 {lbl_b}** — {len(df_out_b):,} hồ sơ")
            st.dataframe(df_out_b.head(20), use_container_width=True, hide_index=True)
            csv_b = df_out_b.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button("⬇️ Tải kết quả Mô hình B (CSV)", csv_b,
                               "ket_qua_mo_hinh_B.csv", mime="text/csv")
