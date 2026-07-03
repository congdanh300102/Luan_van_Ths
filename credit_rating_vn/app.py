"""Trang chủ — Credit Rating Vietnam"""
import streamlit as st

st.set_page_config(
    page_title="Credit Rating VN",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Hệ thống Dự báo Nhóm Nợ & Chấm điểm Tín dụng")
st.markdown("---")

# ── Dataset 1: Data_credit_rating_VN.xlsx ────────────────────────────────────
st.markdown("## 📁 Bộ dữ liệu 1 — Data_credit_rating_VN.xlsx")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("### 📊 Trang 1 — EDA\nKhám phá phân phối nhóm nợ, thống kê mô tả, IV Analysis.")

with col2:
    st.success("### 🤖 Trang 2 — Huấn luyện\nChọn mô hình (Logistic, RF, XGBoost, LightGBM), huấn luyện và đánh giá với Confusion Matrix & ROC-AUC.")

with col3:
    st.warning("### 💳 Trang 3 — Chấm điểm\nChấm điểm tín dụng [300–850] từ xác suất dự báo nhóm nợ. Xếp hạng A+ → E.")

with col4:
    st.info("### 📤 Trang 4 — Dự báo\nUpload file mới → dự báo nhóm nợ + chấm điểm hàng loạt.")

st.markdown("---")

# ── Dataset 2: fct_l.xlsx ─────────────────────────────────────────────────────
st.markdown("## 📁 Bộ dữ liệu 2 — fct_l.xlsx (Mới)")
col5, col6 = st.columns(2)

with col5:
    st.success("### 📊 Trang 5 — Mô hình FCT_L\nXây dựng & huấn luyện mô hình dự báo nhóm nợ trên dữ liệu `fct_l.xlsx` (178 biến). Chấm điểm tín dụng từ kết quả dự báo.")

with col6:
    st.warning("### ⚖️ Trang 6 — So sánh Model\nSo sánh hiệu năng giữa mô hình cũ (Data_credit_rating_VN) và mô hình FCT_L: metrics, phân phối điểm, confusion matrix.")

st.markdown("---")
st.markdown("""
### 📌 Thông tin bộ dữ liệu

| | Data_credit_rating_VN.xlsx | fct_l.xlsx |
|---|---|---|
| **Số hồ sơ** | ~27,001 | 5,400 (30,000 sau augmentation) |
| **Số biến** | 21 | 178 |
| **Target** | `NHOMNOMOI` | `CLASSIFICATION` |
| **Imbalance ratio** | ~34.7× | ~228× |

**Nhóm nợ (theo quy định SBV):**
| Nhóm | Tên | Đặc điểm |
|------|-----|-----------|
| 1 | Nợ đủ tiêu chuẩn | Không quá hạn |
| 2 | Nợ cần chú ý | Quá hạn < 10 ngày |
| 3 | Nợ dưới tiêu chuẩn | Quá hạn 10–90 ngày |
| 4 | Nợ nghi ngờ | Quá hạn 91–180 ngày |
| 5 | Nợ có khả năng mất vốn | Quá hạn > 180 ngày |

👈 **Dùng thanh điều hướng bên trái để bắt đầu.**
""")
