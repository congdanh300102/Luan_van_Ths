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

# ── Dataset 2: fct_l.xlsx ─────────────────────────────────────────────────────
DATA_FCT_L           = ROOT_DIR / "data" / "raw" / "fct_l.xlsx"
DATA_FCT_L_PROCESSED = ROOT_DIR / "data" / "processed" / "fct_l_30k.csv"

FCT_L_TARGET_COL = "CLASSIFICATION"

FCT_L_NUMERICAL_COLS = [
    "INTEREST_RATE",       # lãi suất
    "INTEREST_SPREAD",     # biên độ lãi suất
    "BALANCE",             # dư nợ hiện tại
    "BALANCE_PE",          # dư nợ gốc
    "BALANCE_PS",          # dư nợ lãi
    "AGG_DISBURSEMENT_AMT",# tổng giải ngân tích lũy
    "CONTRACT_CHANGE_CNT", # số lần thay đổi hợp đồng
    "NUM_GRACE_PERIOD",    # số kỳ ân hạn
    "MIS_DAO",             # MIS DAO
    "CURR_MIS_DAO",        # MIS DAO hiện tại
    "FIXED_RATE",          # lãi suất cố định
    "LN_APPR_AMT",         # hạn mức được phê duyệt
]

FCT_L_CATEGORICAL_COLS = [
    "CATEGORY",       # danh mục khoản vay (20 nhóm)
    "SEAB_PRODUCTS",  # sản phẩm SEAB (36 nhóm)
    "TERM_SBV",       # kỳ hạn theo SBV (M01, M12, H00, ...)
    "DATASOURCE",     # nguồn dữ liệu (PD / LD)
]

# SMOTE cho fct_l gốc (5,400 rows) — 0-based class index
FCT_L_SMOTE_ORIGINAL = {
    1: 300,   # N2: 87  → 300
    2: 100,   # N3: 11  → 100
    3: 100,   # N4: 13  → 100
    4: 200,   # N5: 23  → 200
}

# SMOTE cho fct_l_30k (30,000 rows) — 0-based class index
FCT_L_SMOTE_30K = {
    1: 2000,  # N2: ~483  → 2000
    2: 1000,  # N3: ~60   → 1000
    3: 1000,  # N4: ~72   → 1000
    4: 1500,  # N5: ~129  → 1500
}

FCT_L_NHOMNO_LABELS = {
    1: "Nhóm 1 – Đủ tiêu chuẩn",
    2: "Nhóm 2 – Cần chú ý",
    3: "Nhóm 3 – Dưới tiêu chuẩn",
    4: "Nhóm 4 – Nghi ngờ",
    5: "Nhóm 5 – Có khả năng mất vốn",
}
