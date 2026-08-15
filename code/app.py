"""Trang chủ — Credit Rating Vietnam"""
import streamlit as st

st.set_page_config(
    page_title="Credit Rating VN",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Hệ thống Dự báo Nhóm Nợ")
st.markdown(
    "Ứng dụng trí tuệ nhân tạo (Machine Learning) giúp ngân hàng "
    "**tự động dự báo khả năng chuyển nhóm nợ** của từng khách hàng vay — "
    "hỗ trợ cán bộ tín dụng ra quyết định nhanh, nhất quán và kiểm soát "
    "rủi ro danh mục cho vay tốt hơn."
)

st.markdown("---")

# ── Giới thiệu tổng quan ──────────────────────────────────────────────────────
st.markdown("## 🎯 Giới thiệu chung")
i1, i2, i3 = st.columns(3)

with i1:
    st.markdown("#### 🧭 Mục đích")
    st.markdown(
        "Phát hiện sớm khách hàng có nguy cơ chuyển sang nhóm nợ xấu hơn, "
        "từ đó **sàng lọc, giám sát và ra quyết định tín dụng** dựa trên "
        "dữ liệu thay vì cảm tính."
    )

with i2:
    st.markdown("#### 👥 Dành cho ai?")
    st.markdown(
        "- **Cán bộ nghiệp vụ / nhập liệu**: chỉ cần import file khách "
        "hàng và nhận kết quả, không cần biết về mô hình.\n"
        "- **Chuyên viên phân tích / QLRR**: khám phá dữ liệu, huấn luyện "
        "và so sánh các mô hình dự báo."
    )

with i3:
    st.markdown("#### 🔄 Quy trình 3 bước")
    st.markdown(
        "1. Tải **file mẫu** đúng định dạng.\n"
        "2. **Upload** dữ liệu khách hàng ở trang *Dự báo*.\n"
        "3. Nhận **kết quả** dự báo nhóm nợ, tải về Excel/CSV."
    )

st.markdown("---")

# ── CTA — dành cho người nhập liệu / cán bộ nghiệp vụ ────────────────────────
st.markdown("## 📥 Import dữ liệu & dự báo ngay")

with st.container(border=True):
    st.markdown("### 📤 Import hàng loạt (khuyến nghị)")
    st.markdown(
        "Upload 1 file Excel/CSV chứa **nhiều khách hàng** → hệ thống "
        "tự động dự báo nhóm nợ cho toàn bộ danh sách, kèm file mẫu để "
        "nhập đúng định dạng."
    )
    st.page_link("pages/4_📤_Dự_Báo.py", label="➡️  Đi tới trang Dự báo", icon="📤")

st.markdown("---")

# ── Tham chiếu nhanh: ý nghĩa nhóm nợ ─────────────────────────────────────────
st.markdown("## 📌 Ý nghĩa kết quả trả về")

st.markdown("**Nhóm nợ (theo quy định SBV):**")
st.markdown("""
| Nhóm | Tên | Đặc điểm |
|------|-----|-----------|
| 1 | Nợ đủ tiêu chuẩn | Không quá hạn |
| 2 | Nợ cần chú ý | Quá hạn < 10 ngày |
| 3 | Nợ dưới tiêu chuẩn | Quá hạn 10–90 ngày |
| 4 | Nợ nghi ngờ | Quá hạn 91–180 ngày |
| 5 | Nợ có khả năng mất vốn | Quá hạn > 180 ngày |
""")

st.markdown("---")

# ── Dành cho chuyên viên phân tích / Data Scientist ──────────────────────────
with st.expander("🔬 Dành cho chuyên viên phân tích / Data Scientist (EDA · Huấn luyện · So sánh mô hình)"):
    st.markdown("#### 📁 Bộ dữ liệu 1 — Data_credit_rating_VN.xlsx")
    col1, col2 = st.columns(2)
    with col1:
        st.page_link("pages/1_📊_EDA.py", label="Trang 1 — EDA: phân phối nhóm nợ, thống kê mô tả, IV Analysis", icon="📊")
    with col2:
        st.page_link("pages/2_🤖_Huấn_Luyện.py", label="Trang 2 — Huấn luyện mô hình (Logistic, RF, XGBoost, LightGBM)", icon="🤖")

    st.markdown("#### 📁 Bộ dữ liệu 2 — Thông tin tín dụng (train 20260430 / test 20260507)")
    col3, col4 = st.columns(2)
    with col3:
        st.page_link("pages/5_📊_Mô_Hình_Tín_Dụng.py", label="Trang 5 — Mô hình Tín dụng: xây dựng & huấn luyện", icon="📊")
    with col4:
        st.page_link("pages/6_⚖️_So_Sánh_Model.py", label="Trang 6 — So sánh Model: metrics, confusion matrix", icon="⚖️")

    st.page_link("pages/7_🧬_Phân_Tích_Đặc_Trưng.py",
                 label="Trang 7 — Phân tích & Lựa chọn Đặc trưng: nhóm nào quan trọng nhất, giảm còn bao nhiêu biến, nên đầu tư thu thập gì",
                 icon="🧬")

    st.markdown("---")
    st.markdown("""
    **So sánh hai bộ dữ liệu:**

    | | Data_credit_rating_VN.xlsx | Thông tin tín dụng |
    |---|---|---|
    | **Số hồ sơ** | ~27,001 | 100.617 (train 20260430) + 100.274 (test 20260507) |
    | **Số biến** | 21 | 41 (train) / 33 (test) — 33 cột chung dùng làm đặc trưng |
    | **Target** | `NHOMNOMOI` | `Nhóm nợ tự phân loại` |
    | **Imbalance ratio** | ~34.7× | ~89× |
    """)

st.markdown("👈 Hoặc dùng thanh điều hướng bên trái để đi tới bất kỳ trang nào.")
