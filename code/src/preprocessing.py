import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

REFERENCE_DATE = datetime(2021, 12, 31)


def _parse_date(s):
    """Ngày tháng trong file Excel gốc lưu ở hai dạng lẫn lộn trong cùng một
    cột: chuỗi "dd/mm/YYYY" (khi ô định dạng Text) và số serial ngày kiểu
    Excel (khi ô định dạng Date — pandas đọc về int/float thô, không tự động
    parse). Bỏ sót nhánh số serial khiến _parse_date trả về NaT cho MỌI ô
    dạng này — batch kiểm tra trên Data_credit_rating_VN.xlsx cho thấy đây là
    36.6% số dòng ở NGAYDENHAN và 36.8% ở OPEN_DATE, không phải một số ít
    ngoại lệ có thể bỏ qua."""
    if pd.isnull(s):
        return pd.NaT
    if isinstance(s, (int, float, np.integer, np.floating)):
        # Excel serial date, epoch 1899-12-30 (bù trừ sẵn lỗi năm nhuận 1900
        # huyền thoại của Lotus 1-2-3 mà Excel kế thừa).
        return pd.to_datetime(float(s), unit="D", origin="1899-12-30")
    if isinstance(s, (pd.Timestamp, datetime)):
        return pd.Timestamp(s)
    s = str(s).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["OPEN_DATE", "NGAYDENHAN"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_date)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ref = pd.Timestamp(REFERENCE_DATE)

    df["LOAN_TENURE_DAYS"] = (df["NGAYDENHAN"] - df["OPEN_DATE"]).dt.days.clip(lower=0)
    df["DAYS_TO_MATURITY"] = (df["NGAYDENHAN"] - ref).dt.days
    df["UTIL_RATE"] = np.where(df["BASE_BAL"] > 0,
                               df["CURR_BAL"] / df["BASE_BAL"], 0.0).clip(0, 10)
    return df


def clean(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    df = df.copy()
    remove = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=remove)
    df = df.drop(columns=["OPEN_DATE", "NGAYDENHAN"], errors="ignore")
    return df


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_cols: list, numerical_cols: list):
        self.categorical_cols = categorical_cols
        self.numerical_cols   = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_present_ = [c for c in self.categorical_cols if c in X.columns]
        self.num_present_ = [c for c in self.numerical_cols   if c in X.columns]

        self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
        # keep_empty_features=True: nếu một cột số toàn NaN trên phần train của
        # một fold (dễ gặp với sự kiện cực hiếm như bài toán chuyển nhóm nợ),
        # SimpleImputer mặc định ÂM THẦM LOẠI BỎ cột đó khỏi transform() —
        # np.hstack() ở transform() bên dưới không kiểm tra số cột nên không
        # báo lỗi, chỉ lặng lẽ huấn luyện thiếu 1 đặc trưng ở đúng fold đó.
        # keep_empty_features giữ nguyên số cột (điền 0) để hành vi nhất quán
        # giữa mọi fold, đồng thời khớp với CatBoostPreprocessor bên dưới.
        self.num_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler_      = StandardScaler()
        self.label_encoders_ = {}

        if self.cat_present_:
            self.cat_imputer_.fit(X[self.cat_present_].astype(str))

        if self.num_present_:
            self.num_imputer_.fit(X[self.num_present_])
            X_num_imp = self.num_imputer_.transform(X[self.num_present_])
            self.scaler_.fit(X_num_imp)

        for col in self.cat_present_:
            le = LabelEncoder()
            le.fit(X[col].fillna("UNKNOWN").astype(str))
            self.label_encoders_[col] = le

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        for col in self.cat_present_:
            vals = X[col].fillna("UNKNOWN").astype(str)
            known = vals.isin(self.label_encoders_[col].classes_)
            vals[~known] = self.label_encoders_[col].classes_[0]
            X[col] = self.label_encoders_[col].transform(vals)

        X_num = self.num_imputer_.transform(X[self.num_present_])
        X_num = self.scaler_.transform(X_num)
        X_cat = X[self.cat_present_].values.astype(float)
        return np.hstack([X_num, X_cat])

    def get_feature_names(self) -> list:
        return self.num_present_ + self.cat_present_


class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    """
    Tiền xử lý riêng cho CatBoost: chỉ điền khuyết, KHÔNG label-encode
    categorical — CatBoost xử lý categorical trực tiếp từ chuỗi gốc qua
    tham số cat_features, tận dụng đúng thế mạnh của thuật toán thay vì đi
    qua encoding chung như các model khác.
    """

    def __init__(self, categorical_cols: list, numerical_cols: list):
        self.categorical_cols = categorical_cols
        self.numerical_cols   = numerical_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_present_ = [c for c in self.categorical_cols if c in X.columns]
        self.num_present_ = [c for c in self.numerical_cols   if c in X.columns]
        # keep_empty_features=True: xem giải thích ở CreditPreprocessor.fit()
        # phía trên — ở đây việc cột bị loại còn nghiêm trọng hơn, vì
        # transform() gán thẳng vào X[self.num_present_] theo TÊN cột nên số
        # cột lệch sẽ ValueError ngay ("Columns must be same length as key"),
        # thay vì âm thầm sai lệch như ở CreditPreprocessor.
        self.num_imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        if self.num_present_:
            self.num_imputer_.fit(X[self.num_present_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cat_present_:
            X[c] = X[c].fillna("UNKNOWN").astype(str)
        if self.num_present_:
            X[self.num_present_] = self.num_imputer_.transform(X[self.num_present_])
        return X[self.num_present_ + self.cat_present_]

    def get_feature_names(self) -> list:
        return self.num_present_ + self.cat_present_

    def cat_feature_indices(self) -> list:
        """Chỉ số (0-based) của các cột categorical trong output — dùng cho CatBoostClassifier(cat_features=...)."""
        n = len(self.num_present_)
        return list(range(n, n + len(self.cat_present_)))


@staticmethod
def load_raw(filepath: str) -> pd.DataFrame:
    return pd.read_excel(filepath)


def prepare(df: pd.DataFrame, drop_cols: list, target_col: str):
    """Raw DataFrame → (X_features_df, y_1based)"""
    df = parse_dates(df)
    df = engineer_features(df)
    y  = df[target_col].values.copy()
    df = clean(df, drop_cols=drop_cols + [target_col])
    return df, y


# ══════════════════════════════════════════════════════════════════════════════
# Taxonomy nghiệp vụ cho Dataset 1 (Data_credit_rating_VN.xlsx) — dùng cho phân
# tích chọn đặc trưng (trang 7), đối chiếu song song với dữ liệu Thông tin tín dụng.
# Dataset 1 chỉ có 21 cột thô; sau khi loại leakage/trùng lặp/hằng số còn 11
# đặc trưng đã chọn qua IV (dùng để huấn luyện chính ở trang 2) + 2 đặc trưng
# từng bị loại vì IV quá thấp (SEX, LOAIKH) — giữ lại 2 cột này làm "pool đầy
# đủ" 13 biến để kiểm chứng câu hỏi "thêm đặc trưng yếu có giúp ích không".
# ══════════════════════════════════════════════════════════════════════════════
DATASET1_FEATURE_GROUPS: dict[str, list] = {
    "Dư nợ & Hạn mức sử dụng": ["BASE_BAL", "CURR_BAL", "UTIL_RATE"],
    "Lãi suất":                ["LAISUAT"],
    "Kỳ hạn khoản vay":        ["LOAN_TENURE_DAYS", "DAYS_TO_MATURITY"],
    "Đơn vị / Chi nhánh":      ["ORGNBR", "PARENTORGNBR"],
    "Sản phẩm & Mục đích vay": ["MJACCTTYPCD", "CURRMIACCTTYPCD", "MUCDICHVAY"],
    "Nhân khẩu học (IV thấp)": ["SEX", "LOAIKH"],
}


def dataset1_feature_to_group(feature_name: str) -> str:
    for gname, cols in DATASET1_FEATURE_GROUPS.items():
        if feature_name in cols:
            return gname
    return "Khác"
