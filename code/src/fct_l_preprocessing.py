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
    Trích xuất features + target từ fct_l DataFrame (dùng cho bộ RÚT GỌN 16 biến).

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


def transform_fct_l(df: pd.DataFrame,
                    numerical_cols: list,
                    categorical_cols: list) -> pd.DataFrame:
    """Trích xuất feature columns để dự báo (không cần cột target)."""
    feature_cols = [c for c in numerical_cols + categorical_cols if c in df.columns]
    return df[feature_cols].copy()


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
        "TERM_SBV":              "Kỳ hạn theo SBV (M01, M12, H00…)",
        "DATASOURCE":           "Nguồn dữ liệu (PD / LD)",
    }
    return {c: descriptions.get(c, c) for c in numerical_cols + categorical_cols}


# ══════════════════════════════════════════════════════════════════════════════
# Bộ ĐẦY ĐỦ — khai thác toàn bộ 178 cột thô của fct_l.xlsx một cách có quy tắc
# ══════════════════════════════════════════════════════════════════════════════
#
# Mục tiêu: trả lời 4 câu hỏi nghiên cứu về số lượng/nhóm đặc trưng, cần một bộ
# đặc trưng "đầy đủ" lớn hơn nhiều bộ rút gọn 16 biến ở trên, nhưng phải loại
# trừ ID/khóa thay thế (không mang tín hiệu rủi ro) và các cột gây rò rỉ dữ
# liệu (leakage) trước khi đưa vào mô hình.

TARGET_COL = "CLASSIFICATION"

# Cột định danh / khóa thay thế (surrogate key) / văn bản tự do — không phải
# đặc điểm rủi ro, chỉ dùng để join dữ liệu nội bộ.
ID_LEAKAGE_COLS = [
    "CONTRACT", "DAYID", "CUSTOMER_CODE", "LIMIT_REF", "CAMPAIGN_ID",
    "PD_CONTRACT", "LD_LEGACY_ID", "SEAB_LOS_ID", "SEAB_INSUR_ID",
    "SEAB_SALES_ID", "SEAB_BROKERS_ID", "CONTRACT_DES",
    "LAST_EXTEND_USER_CHANGE", "LAST_EXTEND_DATE_CHANGE", "NEW_INT_KEY",
    # Leakage thật sự: CLASS_CUR/CLASSIFICATION_SK là bản sao/khóa của chính
    # target; NO_OVD_DAYS, NO_DAYS_OVERDUE là số ngày quá hạn — theo quy định
    # nhóm nợ được xác định trực tiếp từ số ngày quá hạn nên đưa 2 cột này
    # vào làm feature gần như lộ đáp án, không còn là "dự báo".
    "CLASS_CUR", "CLASSIFICATION_SK", "NO_OVD_DAYS", "NO_DAYS_OVERDUE",
    # Mọi khóa thay thế (*_SK) — trùng thông tin với cột categorical gốc.
    "SUB_CATEGORY_SK", "LIMIT_SK", "CONTRACT_SK", "SUB_PRODUCT_SK",
    "SEAB_PRODUCTS_SK", "CATEGORY_SK", "PRODUCT_SK", "COMPANY_SK",
    "CUSTOMER_SK", "SEAB_PRODUCTS_DE_SK", "CHANNEL_SK", "TERM_SK",
    "TERM_SBV_SK", "SEAB_INSUR_SK", "SEAB_PUR_DE_SK", "SEAB_PUR_FCY_SK",
    "SEAB_REPAY_SRC_SK", "SEAB_BIZ_IMEX_SK", "SUB_AGENCY_CODE_SK",
]

# Cột ngày thô — không đưa thẳng vào model, chỉ dùng để sinh đặc trưng
# (thời lượng, độ mới) trong engineer_business_features().
DATE_COLS = [
    "FIRST_DISBURSEMENT_DATE", "LAST_DISBURSEMENT_DATE", "EXTENDED_DATE",
    "CONTRACT_DATE_CLASS", "PR_NEXT_REPAY_DATE", "START_PERIOD_INT",
    "END_PERIOD_INT", "PERIOD_START_DATE", "PERIOD_END_DATE",
    "RECORD_START_DATE", "INPUT_EXTENDED_DATE", "INT_RATE_V_DATE",
    "SPREAD_V_DATE", "FIRST_MATURITY_DATE", "EXTEND_SCH_DATE",
    "PREF_EXPIRY_DATE", "REMOVED_INSUR_DATE", "PROMO_END_DATE",
    "LAST_EXTEND_SCH_DATE",
]

# Taxonomy nghiệp vụ: mỗi nhóm gồm 2 danh sách (numerical, categorical).
# Đây là bể đặc trưng "đầy đủ" — cột hằng số/gần hằng số trong dữ liệu thực tế
# sẽ bị build_raw_feature_pool() loại tự động (không phải lựa chọn tay).
FEATURE_GROUPS: dict[str, dict[str, list]] = {
    "Dư nợ & Giải ngân": {
        "numerical": [
            "FIRST_DISBURSEMENT_AMT", "LAST_DISBURSEMENT_AMT",
            "DISBURSEMENT_AMT_YTD", "DISBURSEMENT_AMT_WTD",
            "DISBURSEMENT_AMT_MTD", "DISBURSEMENT_AMT", "AGG_DISBURSEMENT_AMT",
            "DISBURSEMENT_NUM_YTD", "DISBURSEMENT_NUM_MTD", "DISBURSEMENT_NUM_WTD",
            "BALANCE", "BALANCE_PE", "BALANCE_PS", "BALANCE_IN", "PD_BALANCE",
            "LN_APPR_AMT", "ACC_BAL_RATE", "AGG_BAL", "AGG_BAL_MTD",
            "AGG_PD_BAL_MTD", "BAL_DLM", "ACC_DISBURSEMENT_RATE",
            "ACC_PD_BAL_RATE", "AGG_PD_BAL", "ACC_BAL_RATE_WTD",
            "ACC_BAL_RATE_MTD", "ACC_BAL_RATE_QTD", "AGG_BAL_WTD",
            "AGG_PD_BAL_WTD", "AGG_BAL_QTD", "AGG_PD_BAL_QTD", "BALANCE_MAT",
            "AGG_BAL_YTD", "AGG_PD_BAL_YTD", "AGG_BAL_MTD_LCY",
            "AGG_BAL_YTD_LCY", "AGG_PD_BAL_MTD_LCY", "AGG_PD_BAL_YTD_LCY",
            "DISBURSEMENT_AMT_MTD_LCY", "DISBURSEMENT_AMT_YTD_LCY",
            "FIRST_DISBURSEMENT_AMT_LCY", "BALANCE_PE_TCKH", "BALANCE_PS_TCKH",
        ],
        "categorical": ["CCY"],
    },
    "Lãi suất": {
        "numerical": [
            "FIXED_RATE", "INTEREST_RATE", "MIN_CONTRACT_RATE", "INTEREST_SPREAD",
            "NEW_INT_RATE_1", "NEW_INT_RATE_2", "NEW_SPREAD", "FIRST_RATE",
            "REVAL_RATE", "DISCOUNT_RATE", "DEF_INT_CAP",
        ],
        "categorical": ["BASIC_RATE_CODE"],
    },
    "Trả nợ & Thu hồi": {
        "numerical": [
            "PRINCIPAL_RCVD", "RECEIVABLE_INTEREST", "RECEIVED_INTEREST",
            "RECEIVED_PE", "RECEIVABLE_PE", "RECEIVED_PS", "RECEIVABLE_PS",
            "ACC_RECEIVABLE_INTEREST", "ACC_RECEIVED_INTEREST", "ACC_RECEIVED_PS",
            "ACC_RECEIVABLE_PS", "ACC_RECEIVED_PE", "ACC_RECEIVABLE_PE",
            "OTS_INTEREST", "OTS_SUSP_INT", "COMMITTED_INT", "ACC_RECEIVED_PR",
            "PR_NEXT_REPAY_AMT", "OTS_INTEREST_TCKH", "AMT_REC",
            "REPAYMENT_AMOUNT", "REPAYMENT_DATE",
        ],
        "categorical": [],
    },
    "Kỳ hạn & Gia hạn": {
        "numerical": [
            "NUM_GRACE_PERIOD", "MIS_DAO", "CURR_MIS_DAO", "EXTEND_SCH_NUM",
            "IS_EXTENDED", "BILLING_CLOSE", "LST_BILLING_CLOSE", "EXTEND_MAT_NUM",
        ],
        "categorical": ["TERM", "TERM_SBV", "FREQUENCY_RATE", "EXTEND_SCH"],
    },
    "Sản phẩm, Mục đích & Kênh phân phối": {
        "numerical": [],
        "categorical": [
            "CATEGORY", "SUB_CATEGORY", "SUB_PRODUCT", "SEAB_PRODUCTS",
            "PRODUCT", "SEAB_PRODUCTS_DE", "CHANNEL_NAME", "SEAB_PUR_DE",
            "SEAB_PUR_FCY", "SEAB_REPAY_SRC", "SEAB_BIZ_IMEX", "SEAB_POLICY",
            "SOURCE_OF_FUND", "SEAB_RESTR_TYPE", "SEAB_OTHER_TYPE",
        ],
    },
    "Khách hàng & Đối tác": {
        "numerical": [],
        "categorical": [
            "CO_CODE", "CUSTOMER_CLASS", "CUST_STATUS", "SEAB_SALES_TYPE",
            "SEAB_BROKERS_TYPE", "SEAB_PARTNER", "INDUSTRY_L1", "SUB_AGENCY_CODE",
        ],
    },
    "Thay đổi & Trạng thái hợp đồng": {
        "numerical": ["CONTRACT_CHANGE_CNT", "CONTRACT_MAT_CNT"],
        "categorical": ["CONTRACT_CLASS", "CONTRACT_CHANGE_DATE", "CONTRACT_TYPE", "CHANGED"],
    },
    "Bối cảnh & Nguồn dữ liệu": {
        "numerical": [],
        "categorical": ["DATASOURCE"],
    },
    # Nhóm này được engineer_business_features() bổ sung động — khai báo rỗng
    # ở đây để feature_to_group() luôn trả về nhóm hợp lệ cho đặc trưng mới.
    "Đặc trưng kỹ thuật (engineered)": {
        "numerical": [],
        "categorical": [],
    },
}


def _all_group_columns() -> set:
    cols = set()
    for grp in FEATURE_GROUPS.values():
        cols.update(grp["numerical"])
        cols.update(grp["categorical"])
    return cols


def build_raw_feature_pool(df: pd.DataFrame) -> tuple[list, list]:
    """
    Xây bộ đặc trưng ĐẦY ĐỦ từ 178 cột thô, theo quy tắc tái lập được:
      1. Loại target, ID_LEAKAGE_COLS, DATE_COLS.
      2. Loại cột hằng số/gần hằng số (nunique <= 1 trên dữ liệu quan sát được) —
         không mang tín hiệu, bất kể câu hỏi "nhiều đặc trưng có tốt hơn".
      3. Phần còn lại lấy theo taxonomy nghiệp vụ FEATURE_GROUPS.

    Returns
    -------
    numerical_cols, categorical_cols : list[str]
    """
    candidates = _all_group_columns() & set(df.columns)

    numerical_cols, categorical_cols = [], []
    for grp in FEATURE_GROUPS.values():
        for c in grp["numerical"]:
            if c in candidates and df[c].nunique(dropna=True) > 1:
                numerical_cols.append(c)
        for c in grp["categorical"]:
            if c in candidates and df[c].nunique(dropna=True) > 1:
                categorical_cols.append(c)

    return numerical_cols, categorical_cols


def feature_to_group(feature_name: str) -> str:
    """Trả về tên nhóm nghiệp vụ của một đặc trưng (dùng cho group-importance)."""
    for grp_name, grp in FEATURE_GROUPS.items():
        if feature_name in grp["numerical"] or feature_name in grp["categorical"]:
            return grp_name
    return "Khác"


_EPS = 1e-6


def engineer_business_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Sinh đặc trưng theo nghiệp vụ ngân hàng từ cột thô + cột ngày.
    Mỗi đặc trưng được gắn vào nhóm "Đặc trưng kỹ thuật (engineered)".

    Returns
    -------
    df_new : pd.DataFrame — df gốc + các cột engineered
    engineered_cols : list[str] — tên các cột vừa sinh (đều numerical)
    """
    df = df.copy()
    new_cols = []

    def _add(name, series):
        df[name] = series
        new_cols.append(name)

    # Dư nợ & Hạn mức: mức độ sử dụng hạn mức được phê duyệt.
    if {"BALANCE", "LN_APPR_AMT"} <= set(df.columns):
        _add("ENG_UTIL_RATE",
             np.where(df["LN_APPR_AMT"] > 0, df["BALANCE"] / df["LN_APPR_AMT"], np.nan))

    # Thu hồi nợ: tỷ lệ đã thu hồi trên tổng phải thu + đã thu (lãi).
    if {"ACC_RECEIVED_INTEREST", "ACC_RECEIVABLE_INTEREST"} <= set(df.columns):
        denom = df["ACC_RECEIVABLE_INTEREST"] + df["ACC_RECEIVED_INTEREST"] + _EPS
        _add("ENG_COLLECTION_RATIO", df["ACC_RECEIVED_INTEREST"] / denom)

    # Lãi suất: chênh lệch lãi suất hiện hành so với biên độ đăng ký.
    if {"INTEREST_RATE", "INTEREST_SPREAD"} <= set(df.columns):
        _add("ENG_RATE_SPREAD_DELTA", df["INTEREST_RATE"] - df["INTEREST_SPREAD"])

    ref = pd.Timestamp.now().normalize()

    # Kỳ hạn & Gia hạn: độ mới của lần giải ngân, thời gian tới đáo hạn, gia hạn.
    if "FIRST_DISBURSEMENT_DATE" in df.columns:
        _add("ENG_DAYS_SINCE_FIRST_DISBURSEMENT",
             (ref - df["FIRST_DISBURSEMENT_DATE"]).dt.days)
    if "LAST_DISBURSEMENT_DATE" in df.columns:
        _add("ENG_DAYS_SINCE_LAST_DISBURSEMENT",
             (ref - df["LAST_DISBURSEMENT_DATE"]).dt.days)
    if "FIRST_MATURITY_DATE" in df.columns:
        _add("ENG_DAYS_TO_MATURITY", (df["FIRST_MATURITY_DATE"] - ref).dt.days)
    if "EXTENDED_DATE" in df.columns:
        _add("ENG_HAS_EXTENSION", df["EXTENDED_DATE"].notna().astype(int))
    if "EXTEND_SCH_NUM" in df.columns:
        _add("ENG_EXTEND_COUNT", df["EXTEND_SCH_NUM"].fillna(0))

    # Hợp đồng: đã từng thay đổi hợp đồng hay chưa.
    if "CONTRACT_CHANGE_CNT" in df.columns:
        _add("ENG_HAS_CONTRACT_CHANGE", (df["CONTRACT_CHANGE_CNT"].fillna(0) > 0).astype(int))

    eng_group = FEATURE_GROUPS["Đặc trưng kỹ thuật (engineered)"]
    eng_group["numerical"] = list(dict.fromkeys(eng_group["numerical"] + new_cols))

    return df, new_cols
