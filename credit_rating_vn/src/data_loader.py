"""
Tải dữ liệu: ưu tiên file local, fallback sang st.file_uploader khi deploy.
Trả về raw bytes để dùng làm cache key ổn định cho @st.cache_data.
"""
import streamlit as st
from pathlib import Path


def get_raw_bytes(data_path: Path) -> bytes:
    """
    Trả về bytes của file Excel.
    - Nếu file tồn tại trên disk → đọc trực tiếp
    - Nếu không → hiện file uploader (Streamlit Cloud không có file)
    """
    if data_path.exists():
        return data_path.read_bytes()

    st.warning(
        "⚠️ Không tìm thấy file dữ liệu tại:\n"
        f"`{data_path}`\n\n"
        "Vui lòng upload file **Data_credit_rating_VN.xlsx**:"
    )
    uploaded = st.file_uploader(
        "Upload file dữ liệu", type=["xlsx"],
        key="global_data_upload",
    )
    if uploaded is None:
        st.stop()

    return uploaded.read()
