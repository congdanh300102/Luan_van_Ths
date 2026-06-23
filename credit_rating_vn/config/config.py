from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent.parent
DATA_RAW  = ROOT_DIR / "data" / "raw" / "Data_credit_rating_VN.xlsx"
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "NHOMNOMOI"

NHOMNO_LABELS = {
    1: "Nhóm 1 – Đủ tiêu chuẩn",
    2: "Nhóm 2 – Cần chú ý",
    3: "Nhóm 3 – Dưới tiêu chuẩn",
    4: "Nhóm 4 – Nghi ngờ",
    5: "Nhóm 5 – Có khả năng mất vốn",
}

GROUP_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}

DROP_COLS = ["NHOMNO_TCBS", "CURRENCYCD"]

CATEGORICAL_COLS = [
    "MJACCTTYPCD", "CURRMIACCTTYPCD", "MIACCTTYPDESC",
    "MJACCTTYPDESC", "DESC_TIME", "SEX", "MUCDICHVAY",
]

NUMERICAL_COLS = [
    "LOAIKH", "BASE_BAL", "CURR_BAL", "DUNO_QD",
    "ID_TIME", "ORGNBR", "PARENTORGNBR", "LAISUAT", "NHOMNO",
    "LOAN_TENURE_DAYS", "DAYS_TO_MATURITY", "UTIL_RATE",
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
