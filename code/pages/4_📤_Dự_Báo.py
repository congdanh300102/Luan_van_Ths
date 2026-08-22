"""Trang 4 — Import dữ liệu & Dự báo nhóm nợ"""
import sys, io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.config import (
    DATA_RAW, MODEL_DIR, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    NHOMNO_LABELS, GROUP_COLORS,
)
from src.preprocessing import parse_dates, engineer_features, clean
from src.models import available_models

st.set_page_config(page_title="Dự báo", page_icon="📤", layout="wide")
st.title("📤 Import dữ liệu & Dự báo")
st.markdown(
    "Upload file danh sách khách hàng (Excel/CSV) → hệ thống tự động dự báo "
    "**nhóm nợ** cho toàn bộ danh sách. "
    "Không cần hiểu về mô hình — chỉ cần đúng định dạng file."
)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model(key: str):
    p = MODEL_DIR / f"model_{key}.pkl"
    return pickle.load(open(p, "rb")) if p.exists() else None


avail = {l: k for l, k in available_models().items()
         if (MODEL_DIR / f"model_{k}.pkl").exists()}

if not avail:
    st.warning("Chưa có mô hình. Huấn luyện ở **Trang 2** trước.")
    st.stop()

best_txt = MODEL_DIR / "best_model.txt"
default_key = best_txt.read_text().strip() if best_txt.exists() else list(avail.values())[0]
default_lbl = next((l for l, k in avail.items() if k == default_key), list(avail.keys())[0])

with st.sidebar:
    st.header("📄 Hướng dẫn file import")
    st.markdown("**Định dạng hỗ trợ**")
    st.markdown("- `.xlsx` / `.xls` (Excel)\n- `.csv`")
    st.markdown("**Cột bắt buộc** (tối thiểu):")
    st.code("\n".join([
        "MJACCTTYPCD, CURRMIACCTTYPCD",
        "MIACCTTYPDESC, MJACCTTYPDESC",
        "DESC_TIME, SEX, MUCDICHVAY",
        "LOAIKH, BASE_BAL, CURR_BAL",
        "DUNO_QD, ID_TIME, ORGNBR",
        "PARENTORGNBR, LAISUAT, NHOMNO",
        "OPEN_DATE, NGAYDENHAN",
    ]))
    st.caption("💡 Dùng nút **Template Excel** bên dưới để lấy đúng cột và thứ tự.")

    st.markdown("---")
    with st.expander("⚙️ Tuỳ chọn nâng cao"):
        model_lbl = st.selectbox("Mô hình dự báo", list(avail.keys()),
                                  index=list(avail.keys()).index(default_lbl))

model_key = avail[model_lbl]
pipe = load_model(model_key)

st.success(f"✅ Mô hình đang dùng: **{model_lbl}** (mặc định mô hình tốt nhất — có thể đổi ở mục *Tuỳ chọn nâng cao* trong sidebar)")
st.markdown("---")

# ── Upload section ────────────────────────────────────────────────────────────
st.subheader("1️⃣  Upload file dữ liệu")

col_up, col_tpl = st.columns([3, 1])
with col_up:
    uploaded = st.file_uploader(
        "Chọn file Excel hoặc CSV",
        type=["xlsx", "xls", "csv"],
        help="File phải có đủ các cột mô tả trong sidebar (mục Hướng dẫn file import)",
    )

with col_tpl:
    st.markdown("**Tải file mẫu**")
    # Tạo template từ data gốc (5 hàng đầu, bỏ cột target)
    @st.cache_data
    def make_template() -> bytes:
        df_src = pd.read_excel(DATA_RAW)
        sample = df_src.drop(columns=[TARGET_COL, "NHOMNO_TCBS", "CURRENCYCD"],
                             errors="ignore").head(5)
        buf = io.BytesIO()
        sample.to_excel(buf, index=False)
        return buf.getvalue()

    st.download_button(
        "⬇️ Template Excel",
        data=make_template(),
        file_name="template_credit.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

if uploaded is None:
    st.info("👆 Upload file để bắt đầu dự báo.")
    st.stop()

# ── Đọc file ─────────────────────────────────────────────────────────────────
st.subheader("2️⃣  Xem trước dữ liệu")

@st.cache_data(show_spinner="Đang đọc file…")
def read_uploaded(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    buf = io.BytesIO(file_bytes)
    if file_name.endswith(".csv"):
        return pd.read_csv(buf)
    return pd.read_excel(buf)

try:
    df_input = read_uploaded(uploaded.read(), uploaded.name)
except Exception as e:
    st.error(f"Không đọc được file: {e}")
    st.stop()

st.markdown(f"**{len(df_input):,} hàng × {df_input.shape[1]} cột**")
st.dataframe(df_input.head(10), use_container_width=True)

# ── Tiền xử lý & Dự báo ──────────────────────────────────────────────────────
st.subheader("3️⃣  Dự báo nhóm nợ")

run_btn = st.button("🚀 Chạy dự báo", type="primary")

if run_btn:
    with st.spinner("Đang xử lý…"):
        try:
            df_proc = parse_dates(df_input.copy())
            df_proc = engineer_features(df_proc)

            # Xác định có cột target không (để tính độ chính xác nếu có)
            has_target = TARGET_COL in df_proc.columns
            y_true = df_proc[TARGET_COL].values.copy() if has_target else None

            df_feat = clean(df_proc, drop_cols=DROP_COLS + ([TARGET_COL] if has_target else []))

            proba   = pipe.predict_proba(df_feat)
            pred    = pipe.predict(df_feat) + 1   # 1-based

            df_result = df_input.copy()
            df_result["nhom_du_bao"]   = pred
            df_result["ten_nhom"]      = [NHOMNO_LABELS.get(p, str(p)) for p in pred]
            for i in range(5):
                df_result[f"xac_suat_nhom_{i+1}"] = proba[:, i].round(4)

            st.session_state["df_result"] = df_result
            st.session_state["y_true"]    = y_true
            st.session_state["proba"]     = proba
            st.session_state["pred"]      = pred

        except Exception as e:
            st.error(f"Lỗi khi dự báo: {e}")
            st.stop()

# ── Hiển thị kết quả ─────────────────────────────────────────────────────────
if "df_result" not in st.session_state:
    st.stop()

df_result = st.session_state["df_result"]
y_true    = st.session_state["y_true"]
proba     = st.session_state["proba"]
pred      = st.session_state["pred"]

st.success(f"✅ Đã dự báo xong **{len(df_result):,}** hồ sơ!")

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tổng hồ sơ", f"{len(df_result):,}")
for i, (col, g) in enumerate([(c2, 1), (c3, 2), (c4, 4), (c5, 5)]):
    cnt = (pred == g).sum()
    col.metric(f"Nhóm {g}", f"{cnt:,}", f"{cnt/len(pred)*100:.1f}%")

tab_dist, tab_table, tab_acc = st.tabs(
    ["📊 Phân phối nhóm", "📋 Bảng kết quả", "🎯 Độ chính xác"]
)

# ── Tab 1: Phân phối nhóm dự báo ─────────────────────────────────────────────
with tab_dist:
    vc = pd.Series(pred).value_counts().sort_index()
    colors = [GROUP_COLORS.get(k, "#999") for k in vc.index]
    fig_pred = go.Figure(go.Bar(
        x=[NHOMNO_LABELS.get(k, str(k)) for k in vc.index],
        y=vc.values,
        marker_color=colors,
        text=[f"{v:,}<br>({v/len(pred)*100:.1f}%)" for v in vc.values],
        textposition="auto",
    ))
    fig_pred.update_layout(title="Phân phối nhóm nợ dự báo",
                           xaxis_title="Nhóm nợ", yaxis_title="Số lượng", height=400)
    st.plotly_chart(fig_pred, use_container_width=True)

    # Xác suất trung bình mỗi nhóm
    proba_df = pd.DataFrame(proba, columns=[f"Nhóm {i+1}" for i in range(5)])
    fig_proba = px.box(proba_df.melt(var_name="Nhóm", value_name="Xác suất"),
                       x="Nhóm", y="Xác suất", color="Nhóm",
                       title="Phân phối xác suất dự báo theo nhóm")
    fig_proba.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_proba, use_container_width=True)

# ── Tab 2: Bảng kết quả đầy đủ ────────────────────────────────────────────────
with tab_table:
    result_cols = (
        [TARGET_COL] if TARGET_COL in df_result.columns else []
    ) + ["nhom_du_bao", "ten_nhom",
         "xac_suat_nhom_1", "xac_suat_nhom_2", "xac_suat_nhom_3",
         "xac_suat_nhom_4", "xac_suat_nhom_5"]

    result_cols = [c for c in result_cols if c in df_result.columns]

    st.dataframe(
        df_result[result_cols],
        use_container_width=True, height=420,
    )

    # Download
    csv_out = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Tải kết quả đầy đủ (CSV)",
        data=csv_out,
        file_name="ket_qua_du_bao.csv",
        mime="text/csv",
    )

    excel_buf = io.BytesIO()
    df_result.to_excel(excel_buf, index=False)
    st.download_button(
        "⬇️ Tải kết quả đầy đủ (Excel)",
        data=excel_buf.getvalue(),
        file_name="ket_qua_du_bao.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ── Tab 3: Độ chính xác (nếu có nhãn thực tế) ────────────────────────────────
with tab_acc:
    if y_true is None:
        st.info("File không có cột `NHOMNOMOI` (nhãn thực tế) nên không thể tính độ chính xác.")
    else:
        from src.evaluation import compute_metrics, plot_confusion_matrix
        metrics = compute_metrics(y_true, pred, proba)

        ca, cb, cc = st.columns(3)
        ca.metric("Macro F1",    f"{metrics['f1_macro']:.4f}")
        cb.metric("Weighted F1", f"{metrics['f1_weighted']:.4f}")
        cc.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")

        fig_cm = plot_confusion_matrix(y_true, pred)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(metrics["report"]).T
        num_r = report_df.select_dtypes(include=float).columns
        st.dataframe(report_df.style.format({c: "{:.3f}" for c in num_r}),
                     use_container_width=True)
