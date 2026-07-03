"""Preprocessing cho tập dữ liệu fct_l.xlsx (178 cột, target = CLASSIFICATION)."""
import io
import numpy as np
import pandas as pd


def load_fct_l(path_or_bytes) -> pd.DataFrame:
    """Đọc fct_l từ đường dẫn file hoặc bytes object."""
    if isinstance(path_or_bytes, (str, type(None))):
        return pd.read_excel(path_or_bytes)
    if isinstance(path_or_bytes, bytes):
        return pd.read_excel(io.BytesIO(path_or_bytes))
    return pd.read_excel(path_or_bytes)


def load_fct_l_processed(path) -> pd.DataFrame:
    """Đọc fct_l_30k.csv (dữ liệu đã tăng cường)."""
    return pd.read_csv(path)


def prepare_fct_l(df: pd.DataFrame,
                  target_col: str,
                  numerical_cols: list,
                  categorical_cols: list):
    """
    Trích xuất features + target từ fct_l DataFrame.

    Returns
    -------
    X : pd.DataFrame  — chứa các cột feature
    y : np.ndarray    — 0-based label (0=N1 … 4=N5)
    """
    df = df.dropna(subset=[target_col]).copy()
    y = df[target_col].astype(int).values - 1  # 1-5 → 0-4

    feature_cols = [c for c in numerical_cols + categorical_cols if c in df.columns]
    X = df[feature_cols].copy()
    return X, y


def fct_l_feature_info(numerical_cols: list, categorical_cols: list) -> dict:
    """Mô tả ngắn từng feature để hiển thị trên UI."""
    descriptions = {
        "INTEREST_RATE":        "Lãi suất hiện hành (%)",
        "INTEREST_SPREAD":      "Biên độ lãi suất (spread)",
        "BALANCE":              "Dư nợ hiện tại (VND)",
        "BALANCE_PE":           "Dư nợ gốc (principal balance)",
        "BALANCE_PS":           "Dư nợ lãi (interest balance)",
        "AGG_DISBURSEMENT_AMT": "Tổng giải ngân tích lũy (VND)",
        "CONTRACT_CHANGE_CNT":  "Số lần thay đổi hợp đồng",
        "NUM_GRACE_PERIOD":     "Số kỳ ân hạn",
        "MIS_DAO":              "MIS DAO",
        "CURR_MIS_DAO":         "MIS DAO hiện tại",
        "FIXED_RATE":           "Lãi suất cố định (%)",
        "LN_APPR_AMT":          "Hạn mức phê duyệt (VND)",
        "CATEGORY":             "Danh mục khoản vay (20 nhóm)",
        "SEAB_PRODUCTS":        "Sản phẩm SEAB (36 nhóm)",
        "TERM_SBV":             "Kỳ hạn theo SBV (M01, M12, H00…)",
        "DATASOURCE":           "Nguồn dữ liệu (PD / LD)",
    }
    return {c: descriptions.get(c, c) for c in numerical_cols + categorical_cols}
