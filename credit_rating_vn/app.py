"""Trang chủ — Credit Rating Vietnam"""
import streamlit as st

st.set_page_config(
    page_title="Credit Rating VN",
    page_icon="💳",
    layout="wide",
)

st.title("💳 Hệ thống Dự báo Nhóm Nợ & Chấm điểm Tín dụng")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("### 📊 Bước 1 — EDA\nKhám phá và phân tích dữ liệu tín dụng. Xem phân phối nhóm nợ, thống kê mô tả, biểu đồ tương quan.")

with col2:
    st.success("### 🤖 Bước 2 — Huấn luyện\nChọn mô hình (Logistic, Random Forest, XGBoost, LightGBM), huấn luyện và đánh giá với Confusion Matrix & ROC-AUC.")

with col3:
    st.warning("### 💯 Bước 3 — Chấm điểm\nChấm điểm tín dụng [300–850] từ xác suất dự báo nhóm nợ. Xếp hạng A+ → E. Nhập tay hoặc tải kết quả.")

st.markdown("---")
st.markdown("""
**Dữ liệu**: `Data_credit_rating_VN.xlsx` — 27,001 hồ sơ vay | 21 biến
**Nhóm nợ target** (`NHOMNOMOI`):
| Nhóm | Tên | Tỷ lệ |
|------|-----|--------|
| 1 | Nợ đủ tiêu chuẩn | ~77.6% |
| 2 | Nợ cần chú ý     | ~4.8%  |
| 3 | Nợ dưới tiêu chuẩn | ~2.2% |
| 4 | Nợ nghi ngờ       | ~2.9%  |
| 5 | Nợ có khả năng mất vốn | ~12.5% |

👈 **Dùng thanh điều hướng bên trái để bắt đầu.**
""")
