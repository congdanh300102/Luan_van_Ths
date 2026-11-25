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

    # Giữ file đã upload trong phiên làm việc → chỉ cần upload một lần cho mọi trang.
    cache_key = f"_uploaded_bytes::{data_path.name}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    st.info(
        "Bản online không kèm dữ liệu (dữ liệu ngân hàng không được đưa lên git). "
        f"Vui lòng upload file **{data_path.name}**; file chỉ được giữ trong phiên làm việc này."
    )
    uploaded = st.file_uploader(
        "Upload file dữ liệu", type=["xlsx"],
        key="global_data_upload",
    )
    if uploaded is None:
        st.stop()

    st.session_state[cache_key] = uploaded.getvalue()
    return st.session_state[cache_key]
