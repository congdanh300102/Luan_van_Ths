"""Trang 3 — Chấm điểm tín dụng"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW, MODEL_DIR, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS,
    RANDOM_STATE, TEST_SIZE, SCORE_BANDS,
)
from src.models import available_models
from src.preprocessing import prepare
from src.scoring import (
    build_score_df, plot_score_distribution,
    plot_score_by_group, score_gauge,
    proba_to_score, classify_score,
)
from src.data_loader import get_raw_bytes

st.set_page_config(page_title="Chấm điểm", page_icon="💳", layout="wide")
st.title("💳 Chấm điểm Tín dụng")


# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dữ liệu…")
def load_test_set(raw_bytes: bytes):
    df = pd.read_excel(io.BytesIO(raw_bytes))
    X, y = prepare(df, DROP_COLS, TARGET_COL)
    y_0  = y - 1
    _, X_test, _, y_test = train_test_split(
        X, y_0, test_size=TEST_SIZE, stratify=y_0, random_state=RANDOM_STATE
    )
    return X_test, y_test


@st.cache_resource(show_spinner="Đang tải mô hình…")
def load_model(model_key: str):
    path = MODEL_DIR / f"model_{model_key}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Sidebar: chọn mô hình ────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")

    available = {label: key for label, key in available_models().items()
                 if (MODEL_DIR / f"model_{key}.pkl").exists()}

    best_txt = MODEL_DIR / "best_model.txt"
    default_key = best_txt.read_text().strip() if best_txt.exists() else None
    default_label = next((l for l, k in available_models().items() if k == default_key), None)

    if not available:
        st.warning("Chưa có mô hình nào. Hãy huấn luyện ở Trang 2 trước.")
        st.stop()

    model_label = st.selectbox(
        "Chọn mô hình",
        options=list(available.keys()),
        index=list(available.keys()).index(default_label) if default_label in available else 0,
    )
    model_key = available[model_label]

    st.markdown("---")
    st.markdown("**Thang điểm tín dụng**")
    for lo, hi, grade, desc, color in SCORE_BANDS:
        st.markdown(
            f"<span style='background:{color};color:#fff;padding:2px 8px;"
            f"border-radius:4px'>{grade}</span>  [{lo}–{hi}] {desc}",
            unsafe_allow_html=True,
        )

# ── Load model ───────────────────────────────────────────────────────────────
pipe = load_model(model_key)
if pipe is None:
    st.error(f"Không tìm thấy model_{model_key}.pkl. Vui lòng huấn luyện ở Trang 2.")
    st.stop()

st.success(f"✅ Đang dùng mô hình: **{model_label}**")

# ── Load dữ liệu ─────────────────────────────────────────────────────────────
_raw_bytes = get_raw_bytes(DATA_RAW)
_df_raw = pd.read_excel(io.BytesIO(_raw_bytes))

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Đánh giá tập Test", "🔎 Nhập tay / Upload"])

# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Kết quả chấm điểm trên tập kiểm tra")

    X_test, y_test = load_test_set(_raw_bytes)
    y_pred  = pipe.predict(X_test) + 1
    y_proba = pipe.predict_proba(X_test)
    y_true  = y_test + 1

    df_scores = build_score_df(y_proba, y_pred, y_true)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số hồ sơ",        f"{len(df_scores):,}")
    c2.metric("Điểm trung bình", f"{df_scores['diem_tin_dung'].mean():.0f}")
    c3.metric("Điểm thấp nhất",  f"{df_scores['diem_tin_dung'].min()}")
    c4.metric("Điểm cao nhất",   f"{df_scores['diem_tin_dung'].max()}")

    # Biểu đồ phân phối
    fig_dist = plot_score_distribution(df_scores)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Boxplot theo nhóm nợ
    fig_box = plot_score_by_group(df_scores)
    if fig_box:
        st.plotly_chart(fig_box, use_container_width=True)

    # Bảng thống kê
    st.markdown("#### Điểm trung bình theo nhóm nợ thực tế")
    tbl = (df_scores.groupby("nhom_thuc_te")["diem_tin_dung"]
                    .agg(["mean", "median", "std", "min", "max"])
                    .round(1))
    tbl.index = [f"Nhóm {i}" for i in tbl.index]
    st.dataframe(tbl, use_container_width=True)

    # Phân bổ hạng
    st.markdown("#### Phân bổ hạng tín dụng")
    grade_order = [b[2] for b in SCORE_BANDS]
    grade_colors = {b[2]: b[4] for b in SCORE_BANDS}
    vc = df_scores["hang"].value_counts().reindex(grade_order, fill_value=0)
    grade_tbl = pd.DataFrame({
        "Hạng": vc.index,
        "Mô tả": [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc.index],
        "Số lượng": vc.values,
        "Tỷ lệ (%)": (vc.values / len(df_scores) * 100).round(1),
    })
    st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

    # Download
    csv = df_scores.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Tải kết quả CSV",
        data=csv,
        file_name=f"credit_scores_{model_key}.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    input_mode = st.radio("Chế độ nhập liệu", ["Nhập tay 1 khách hàng", "Upload file CSV/Excel"])

    # ── Nhập tay ─────────────────────────────────────────────────────────────
    if input_mode == "Nhập tay 1 khách hàng":
        st.subheader("Thông tin khách hàng")

        df_raw_all = _df_raw

        col1, col2, col3 = st.columns(3)
        with col1:
            loaikh   = st.selectbox("Loại khách hàng", [1, 2])
            sex      = st.selectbox("Giới tính", ["ONG", "BA", "MR", "MRS", "MS"])
            mjtype   = st.selectbox("MJACCTTYPCD", df_raw_all["MJACCTTYPCD"].unique().tolist())
        with col2:
            base_bal  = st.number_input("Dư nợ gốc (BASE_BAL)", min_value=0, value=50_000_000, step=1_000_000)
            curr_bal  = st.number_input("Dư nợ hiện tại (CURR_BAL)", min_value=0, value=45_000_000, step=1_000_000)
            laisuat   = st.number_input("Lãi suất", min_value=0.0, max_value=1.0, value=0.12, step=0.01, format="%.4f")
        with col3:
            nhomno    = st.selectbox("Nhóm nợ cũ (NHOMNO)", [1, 2, 3, 4, 5])
            open_date = st.date_input("Ngày mở khoản vay", value=pd.Timestamp("2018-01-01"))
            due_date  = st.date_input("Ngày đáo hạn", value=pd.Timestamp("2025-12-31"))

        if st.button("💳 Chấm điểm", type="primary"):
            import datetime
            ref = datetime.datetime(2021, 12, 31)
            open_dt = pd.Timestamp(open_date)
            due_dt  = pd.Timestamp(due_date)
            tenure  = max((due_dt - open_dt).days, 0)
            days_mat = (due_dt - pd.Timestamp(ref)).days
            util    = min(curr_bal / base_bal, 10.0) if base_bal > 0 else 0.0

            # Tạo dataframe 1 hàng khớp với features đã train
            X_sample = pd.DataFrame([{
                "MJACCTTYPCD":     mjtype,
                "CURRMIACCTTYPCD": df_raw_all["CURRMIACCTTYPCD"].mode()[0],
                "MIACCTTYPDESC":   df_raw_all["MIACCTTYPDESC"].mode()[0],
                "MJACCTTYPDESC":   df_raw_all["MJACCTTYPDESC"].mode()[0],
                "DESC_TIME":       df_raw_all["DESC_TIME"].mode()[0],
                "SEX":             sex,
                "MUCDICHVAY":      df_raw_all["MUCDICHVAY"].mode()[0],
                "LOAIKH":          loaikh,
                "BASE_BAL":        base_bal,
                "CURR_BAL":        curr_bal,
                "DUNO_QD":         curr_bal,
                "ID_TIME":         1,
                "ORGNBR":          df_raw_all["ORGNBR"].median(),
                "PARENTORGNBR":    df_raw_all["PARENTORGNBR"].median(),
                "LAISUAT":         laisuat,
                "NHOMNO":          nhomno,
                "LOAN_TENURE_DAYS": tenure,
                "DAYS_TO_MATURITY": days_mat,
                "UTIL_RATE":        util,
            }])

            proba   = pipe.predict_proba(X_sample)
            pred    = pipe.predict(X_sample)[0] + 1
            score   = proba_to_score(proba)[0]
            grade, desc, color = classify_score(score)

            c_res1, c_res2 = st.columns([1, 2])
            with c_res1:
                st.plotly_chart(score_gauge(score), use_container_width=True)
                st.markdown(
                    f"<h3 style='text-align:center;color:{color}'>Hạng {grade} — {desc}</h3>",
                    unsafe_allow_html=True,
                )
            with c_res2:
                st.markdown(f"**Dự báo nhóm nợ**: Nhóm **{pred}**")
                st.markdown("**Xác suất dự báo theo nhóm:**")
                proba_df = pd.DataFrame({
                    "Nhóm nợ": [f"Nhóm {i+1}" for i in range(5)],
                    "Xác suất": proba[0].round(4),
                })
                import plotly.express as px
                fig_pb = px.bar(proba_df, x="Nhóm nợ", y="Xác suất",
                                color="Xác suất", color_continuous_scale="RdYlGn_r",
                                range_y=[0, 1])
                fig_pb.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_pb, use_container_width=True)

    # ── Upload file ───────────────────────────────────────────────────────────
    else:
        st.subheader("Upload file dữ liệu khách hàng")
        st.markdown("""
        File cần có các cột: `MJACCTTYPCD`, `CURRMIACCTTYPCD`, `MIACCTTYPDESC`,
        `MJACCTTYPDESC`, `DESC_TIME`, `SEX`, `MUCDICHVAY`, `LOAIKH`,
        `BASE_BAL`, `CURR_BAL`, `DUNO_QD`, `ID_TIME`, `ORGNBR`, `PARENTORGNBR`,
        `LAISUAT`, `NHOMNO`, `OPEN_DATE`, `NGAYDENHAN`
        """)

        uploaded = st.file_uploader("Chọn file", type=["csv", "xlsx"])
        if uploaded:
            try:
                if uploaded.name.endswith(".csv"):
                    df_up = pd.read_csv(uploaded)
                else:
                    df_up = pd.read_excel(uploaded)

                from src.preprocessing import parse_dates, engineer_features, clean
                df_up = parse_dates(df_up)
                df_up = engineer_features(df_up)
                df_up = clean(df_up, drop_cols=DROP_COLS + [TARGET_COL])

                with st.spinner("Đang dự báo…"):
                    proba_up = pipe.predict_proba(df_up)
                    pred_up  = pipe.predict(df_up) + 1

                df_result = build_score_df(proba_up, pred_up)
                st.success(f"✅ Đã chấm điểm {len(df_result):,} hồ sơ")

                fig_d = plot_score_distribution(df_result)
                st.plotly_chart(fig_d, use_container_width=True)

                st.dataframe(df_result.head(100), use_container_width=True)

                csv_up = df_result.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                st.download_button("⬇️ Tải kết quả CSV", data=csv_up,
                                   file_name="credit_scores_upload.csv",
                                   mime="text/csv")
            except Exception as e:
                st.error(f"Lỗi xử lý file: {e}")
