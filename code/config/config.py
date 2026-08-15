from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT_DIR / "data" / "raw" / "Data_credit_rating_VN.xlsx"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset 1: Data_credit_rating_VN.xlsx ─────────────────────────────────────
TARGET_COL = "NHOMNOMOI"

NHOMNO_LABELS = {
    1: "Nhóm 1 – Đủ tiêu chuẩn",
    2: "Nhóm 2 – Cần chú ý",
    3: "Nhóm 3 – Dưới tiêu chuẩn",
    4: "Nhóm 4 – Nghi ngờ",
    5: "Nhóm 5 – Có khả năng mất vốn",
}

GROUP_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}

DROP_COLS = [
    "NHOMNO_TCBS",   # text version của target — leakage
    "CURRENCYCD",    # 1 giá trị duy nhất (VND) — IV=0.00
    "NHOMNO",        # leakage: corr=0.98, IV=1.06 (suspicious)
    "DUNO_QD",       # 100% giống CURR_BAL — redundant
    "MIACCTTYPDESC", # text duplicate của CURRMIACCTTYPCD
    "MJACCTTYPDESC", # text duplicate của MJACCTTYPCD
    "ID_TIME",       # period ID, 3 giá trị — IV=0.006 (useless)
    "DESC_TIME",     # text version của ID_TIME
    "SEX",           # IV=0.008 — gần như không có predictive power
    "LOAIKH",        # IV=0.000 — hoàn toàn không dự báo được
]

CATEGORICAL_COLS = [
    "MJACCTTYPCD",       # IV=0.97 — loại sản phẩm vay chính (3 nhóm)
    "CURRMIACCTTYPCD",   # IV=1.99 — loại sản phẩm vay chi tiết (31 nhóm)
    "MUCDICHVAY",        # IV=1.41 — mục đích vay (79 nhóm)
]

NUMERICAL_COLS = [
    "BASE_BAL",          # hạn mức/dư nợ gốc
    "CURR_BAL",          # dư nợ hiện tại
    "LAISUAT",           # lãi suất
    "ORGNBR",            # mã chi nhánh
    "PARENTORGNBR",      # mã chi nhánh cấp trên
    "LOAN_TENURE_DAYS",  # thời hạn khoản vay (ngày)
    "DAYS_TO_MATURITY",  # số ngày đến đáo hạn
    "UTIL_RATE",         # tỷ lệ sử dụng hạn mức = CURR_BAL / BASE_BAL
]

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

SCORE_MIN = 300
SCORE_MAX = 850
RISK_WEIGHTS = [0.0, 0.25, 0.50, 0.75, 1.0]

SCORE_BANDS = [
    (750, 850, "A+", "Xuất sắc",            "#1a9850"),
    (700, 749, "A",  "Tốt",                  "#66bd63"),
    (650, 699, "B+", "Khá",                  "#a6d96a"),
    (600, 649, "B",  "Trung bình khá",        "#fee08b"),
    (550, 599, "C+", "Trung bình",            "#fdae61"),
    (500, 549, "C",  "Trung bình yếu",        "#f46d43"),
    (450, 499, "D",  "Yếu",                  "#d73027"),
    (300, 449, "E",  "Rất yếu / Từ chối",    "#a50026"),
]

MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
    "LightGBM":            "lightgbm",
}

# ── Dataset 2: Thông tin tín dụng (train 20260430 / test 20260507) ───────────
DATA_CREDIT_INFO_TRAIN = ROOT_DIR / "data" / "raw" / "Thông tin tín dụng 20260430.xlsx"
DATA_CREDIT_INFO_TEST  = ROOT_DIR / "data" / "raw" / "Thông tin tín dụng 20260507.xlsx"

CREDIT_INFO_TARGET_COL = "Nhóm nợ tự phân loại"

# 20260507 (test) chỉ có 33/41 cột của 20260430 (train) — tập đặc trưng chỉ
# được xây từ 33 cột chung để mô hình huấn luyện trên train vẫn chấm điểm
# được trên tập test độc lập. Danh sách đầy đủ + lý do loại từng cột nằm ở
# src/credit_info_preprocessing.py (ID_LEAKAGE_COLS, DATE_COLS, FEATURE_GROUPS).
CREDIT_INFO_NUMERICAL_COLS = [
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

CREDIT_INFO_CATEGORICAL_COLS = [
    "Hoạt động cấp tín dụng bằng phương tiện điện tử",
    "Mã thời hạn cấp tín dụng",
    "Hình thức cấp tín dụng",
    "Phương thức cho vay",
    "Mã tiền tệ",
    "Mục đích sử dụng tiền vay phân theo ngành kinh tế",
    "Mô tả mục đích sử dụng tiền vay",
    "Nguồn cấp tín dụng",
]

# SMOTE cho tập train (80% của 20260430, ~80.5k dòng) — 0-based class index.
# Phân phối gốc trên 100% file 20260430: N1=88.875, N2=6.321, N3=998, N4=1.446,
# N5=2.977 → phân phối trên phần train (80%) xấp xỉ N1=71.100, N2=5.057,
# N3=798, N4=1.157, N5=2.382. Nâng minority lên mức vừa phải (không full
# balance), theo đúng triết lý smote_moderate/custom đã dùng cho 2 bộ dữ liệu
# trước — tỉ lệ synthetic vẫn < 50% mỗi lớp thiểu số.
CREDIT_INFO_SMOTE_STRATEGY = {
    1: 8000,   # N2: ~5.057 → 8.000  (~11% của N1)
    2: 4000,   # N3: ~798   → 4.000  (~5.6% của N1)
    3: 4500,   # N4: ~1.157 → 4.500  (~6.3% của N1)
    4: 6000,   # N5: ~2.382 → 6.000  (~8.4% của N1)
}

CREDIT_INFO_NHOMNO_LABELS = NHOMNO_LABELS
