"""Tiền xử lý cho tập dữ liệu "Thông tin tín dụng" (train 20260430 / test 20260507).

Target = "Nhóm nợ tự phân loại" (1-5). Tập đặc trưng chỉ được xây từ 33 cột
chung giữa 2 file train/test — 20260507 (test) thiếu 8 cột so với 20260430
(train), nên mọi cột chỉ có ở train đều bị loại để mô hình huấn luyện vẫn
chấm điểm được trên tập test độc lập.
"""
import io
import numpy as np
import pandas as pd


def load_credit_info(path_or_bytes) -> pd.DataFrame:
    """Đọc file Thông tin tín dụng từ đường dẫn hoặc bytes object, parse cột ngày."""
    if isinstance(path_or_bytes, bytes):
        df = pd.read_excel(io.BytesIO(path_or_bytes))
    else:
        df = pd.read_excel(path_or_bytes)
    return parse_dates(df)


TARGET_COL = "Nhóm nợ tự phân loại"

# Cột định danh — không mang tín hiệu rủi ro, chỉ dùng để join/tra cứu nội bộ.
ID_LEAKAGE_COLS = [
    "Mã khách hàng do TCTD cấp", "Tên khách hàng",
    "Số hợp đồng tín dụng", "Số khế ước",
    # Leakage thật sự: nhóm nợ đối chiếu CIC và dự phòng cụ thể được tính trực
    # tiếp từ nhóm nợ theo quy định NHNN (tỉ lệ trích lập cố định 0/5/20/50/100%
    # theo từng nhóm) — đưa vào làm feature gần như lộ đáp án. Tương tự, dư nợ/
    # lãi chậm trả thực tế và ngày chậm trả là căn cứ trực tiếp để xếp nhóm nợ.
    "Nhóm nợ phân loại sau khi tham chiếu CIC",
    "Dư nợ gốc chậm trả thực tế", "Ngày chậm trả nợ gốc",
    "Số tiền lãi chậm trả thực tế", "Ngày chậm trả nợ lãi",
    "Dự phòng cụ thể phải trích nội bảng", "Dự phòng cụ thể đã trích nội bảng",
]

# Cột ngày thô (lưu dạng số nguyên YYYYMMDD trong file gốc) — không đưa thẳng
# vào model, chỉ dùng để sinh đặc trưng trong engineer_business_features().
# "Thời điểm truy đòi" trùng khớp 100% với "Ngày kết thúc khế ước" trong dữ
# liệu quan sát được (kiểm chứng thực nghiệm) — bỏ hẳn khỏi DATE_COLS để
# tránh sinh đặc trưng trùng lặp hoàn toàn (đa cộng tuyến vô ích).
DATE_COLS = ["ETL_DATE", "Ngày giải ngân", "Ngày kết thúc khế ước"]

RAW_NUMERICAL_COLS = [
    "Mã chi nhánh TCTD",
    "Lãi suất",
    "Số dư nợ theo nguyên tệ",
    "Số lần cơ cấu lại thời hạn trả nợ",
    "Số tiền nợ gốc cơ cấu",
    "Số tiền nợ lãi cơ cấu",
    "Lãi phải thu hạch toán nội bảng",
    "Lãi chưa thu hạch toán ngoại bảng",
    "Số tiền đã thanh toán",
]

RAW_CATEGORICAL_COLS = [
    "Hoạt động cấp tín dụng bằng phương tiện điện tử",
    "Mã thời hạn cấp tín dụng",
    "Hình thức cấp tín dụng",
    "Phương thức cho vay",
    "Mã tiền tệ",
    "Mục đích sử dụng tiền vay phân theo ngành kinh tế",
    "Mô tả mục đích sử dụng tiền vay",
    "Nguồn cấp tín dụng",
]


def _parse_yyyymmdd(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = _parse_yyyymmdd(df[col])
    return df


def prepare_credit_info(df: pd.DataFrame,
                        target_col: str,
                        numerical_cols: list,
                        categorical_cols: list):
    """
    Trích xuất features + target từ DataFrame Thông tin tín dụng.

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


def transform_credit_info(df: pd.DataFrame,
                          numerical_cols: list,
                          categorical_cols: list) -> pd.DataFrame:
    """Trích xuất feature columns để dự báo (không cần cột target)."""
    feature_cols = [c for c in numerical_cols + categorical_cols if c in df.columns]
    return df[feature_cols].copy()


def credit_info_feature_info(numerical_cols: list, categorical_cols: list) -> dict:
    """Mô tả ngắn từng feature để hiển thị trên UI."""
    descriptions = {
        "Mã chi nhánh TCTD":                              "Mã chi nhánh cấp tín dụng",
        "Lãi suất":                                        "Lãi suất hiện hành (%)",
        "Số dư nợ theo nguyên tệ":                         "Dư nợ hiện tại theo nguyên tệ",
        "Số lần cơ cấu lại thời hạn trả nợ":                "Số lần cơ cấu lại thời hạn trả nợ",
        "Số tiền nợ gốc cơ cấu":                            "Số tiền nợ gốc đã cơ cấu",
        "Số tiền nợ lãi cơ cấu":                            "Số tiền nợ lãi đã cơ cấu",
        "Lãi phải thu hạch toán nội bảng":                  "Lãi phải thu hạch toán nội bảng",
        "Lãi chưa thu hạch toán ngoại bảng":                "Lãi chưa thu hạch toán ngoại bảng",
        "Số tiền đã thanh toán":                            "Số tiền đã thanh toán",
        "Hoạt động cấp tín dụng bằng phương tiện điện tử":  "Cấp tín dụng qua kênh điện tử (0/1)",
        "Mã thời hạn cấp tín dụng":                         "Mã thời hạn cấp tín dụng (ngắn/trung/dài hạn)",
        "Hình thức cấp tín dụng":                           "Hình thức cấp tín dụng",
        "Phương thức cho vay":                              "Phương thức cho vay",
        "Mã tiền tệ":                                       "Loại tiền tệ (VND/USD/…)",
        "Mục đích sử dụng tiền vay phân theo ngành kinh tế": "Mục đích vay theo ngành kinh tế",
        "Mô tả mục đích sử dụng tiền vay":                  "Mô tả mục đích vay",
        "Nguồn cấp tín dụng":                               "Nguồn cấp tín dụng",
        "ENG_LOAN_AGE_DAYS":       "Số ngày kể từ khi giải ngân đến ngày chốt dữ liệu (ETL_DATE)",
        "ENG_DAYS_TO_MATURITY":    "Số ngày còn lại đến khi kết thúc khế ước (tính từ ETL_DATE)",
        "ENG_TENOR_ACTUAL_DAYS":   "Kỳ hạn thực tế của khế ước (ngày kết thúc − ngày giải ngân)",
        "ENG_HAS_RESTRUCTURE":     "Đã từng cơ cấu lại thời hạn trả nợ hay chưa (0/1)",
    }
    return {c: descriptions.get(c, c) for c in numerical_cols + categorical_cols}


# ══════════════════════════════════════════════════════════════════════════════
# Taxonomy nghiệp vụ — 33 cột chung giữa 2 file train/test.
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_GROUPS: dict[str, dict[str, list]] = {
    "Dư nợ & Thanh toán": {
        "numerical": [
            "Số dư nợ theo nguyên tệ", "Số tiền đã thanh toán",
            "Lãi phải thu hạch toán nội bảng", "Lãi chưa thu hạch toán ngoại bảng",
        ],
        "categorical": [],
    },
    "Lãi suất": {
        "numerical": ["Lãi suất"],
        "categorical": [],
    },
    "Kỳ hạn & Cơ cấu lại": {
        "numerical": ["Số lần cơ cấu lại thời hạn trả nợ", "Số tiền nợ gốc cơ cấu", "Số tiền nợ lãi cơ cấu"],
        "categorical": ["Mã thời hạn cấp tín dụng"],
    },
    "Hình thức & Mục đích vay": {
        "numerical": [],
        "categorical": [
            "Hình thức cấp tín dụng", "Phương thức cho vay", "Mã tiền tệ",
            "Mục đích sử dụng tiền vay phân theo ngành kinh tế",
            "Mô tả mục đích sử dụng tiền vay", "Nguồn cấp tín dụng",
            "Hoạt động cấp tín dụng bằng phương tiện điện tử",
        ],
    },
    "Chi nhánh": {
        "numerical": ["Mã chi nhánh TCTD"],
        "categorical": [],
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
    Xây bộ đặc trưng hợp lệ từ 33 cột thô chung, theo quy tắc tái lập được:
      1. Loại target, ID_LEAKAGE_COLS, DATE_COLS.
      2. Loại cột hằng số/gần hằng số (nunique <= 1 trên dữ liệu quan sát được).
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


def engineer_business_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Sinh đặc trưng thời gian từ cột ngày, dùng ETL_DATE của TỪNG DÒNG làm mốc
    cố định (thay cho pd.Timestamp.now() của bản fct_l trước đây — mốc "ngày
    chạy chương trình" khiến giá trị biến đổi theo ngày thực thi, một lỗi tái
    lập kết quả đã được ghi nhận và cần khắc phục ở bộ dữ liệu này).

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

    ref = df["ETL_DATE"] if "ETL_DATE" in df.columns else None

    if ref is not None and "Ngày giải ngân" in df.columns:
        _add("ENG_LOAN_AGE_DAYS", (ref - df["Ngày giải ngân"]).dt.days)
    if ref is not None and "Ngày kết thúc khế ước" in df.columns:
        _add("ENG_DAYS_TO_MATURITY", (df["Ngày kết thúc khế ước"] - ref).dt.days)
    if {"Ngày kết thúc khế ước", "Ngày giải ngân"} <= set(df.columns):
        _add("ENG_TENOR_ACTUAL_DAYS", (df["Ngày kết thúc khế ước"] - df["Ngày giải ngân"]).dt.days)
    if "Số lần cơ cấu lại thời hạn trả nợ" in df.columns:
        _add("ENG_HAS_RESTRUCTURE", (df["Số lần cơ cấu lại thời hạn trả nợ"].fillna(0) > 0).astype(int))

    eng_group = FEATURE_GROUPS["Đặc trưng kỹ thuật (engineered)"]
    eng_group["numerical"] = list(dict.fromkeys(eng_group["numerical"] + new_cols))

    return df, new_cols
