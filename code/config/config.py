import unicodedata
from pathlib import Path

ROOT_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_raw(filename: str) -> Path:
    """
    Trả về đường dẫn tới file trong data/raw, chịu được lệch chuẩn hoá Unicode
    giữa tên file trên đĩa và chuỗi literal trong code. Một số file được sao
    chép từ máy macOS lưu tên ở dạng NFD (tổ hợp ký tự, VD "ô" = "o" +
    U+0302), trong khi chuỗi Python thông thường ở dạng NFC (ký tự dựng sẵn)
    — so sánh trực tiếp hai dạng này luôn sai lệch dù nhìn "giống hệt nhau",
    khiến pandas.read_excel báo FileNotFoundError dù `ls`/File Explorer vẫn
    thấy file. Ưu tiên khớp trực tiếp (đường thông thường); nếu không thấy,
    quét thư mục và so khớp theo dạng NFC đã chuẩn hoá.
    """
    raw_dir = ROOT_DIR / "data" / "raw"
    direct = raw_dir / filename
    if direct.exists():
        return direct
    target_nfc = unicodedata.normalize("NFC", filename)
    for p in raw_dir.iterdir():
        if unicodedata.normalize("NFC", p.name) == target_nfc:
            return p
    return direct  # giữ nguyên hành vi cũ (báo lỗi rõ ràng khi thực sự không có file)


DATA_RAW  = _resolve_raw("Data_credit_rating_VN.xlsx")

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
    # Mã đơn vị là nominal identifiers, không phải đại lượng số.
    # Giữ chúng ở categorical tránh tạo thứ tự/khoảng cách giả khi
    # chuẩn hoá cho Logistic Regression hoặc chia ngưỡng trong tree models.
    "ORGNBR",
    "PARENTORGNBR",
]

NUMERICAL_COLS = [
    "BASE_BAL",          # hạn mức/dư nợ gốc
    "CURR_BAL",          # dư nợ hiện tại
    "LAISUAT",           # lãi suất
    "LOAN_TENURE_DAYS",  # thời hạn khoản vay (ngày)
    "DAYS_TO_MATURITY",  # số ngày đến đáo hạn
    "UTIL_RATE",         # tỷ lệ sử dụng hạn mức = CURR_BAL / BASE_BAL
]

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

MODEL_OPTIONS = {
    "Logistic Regression": "logistic",
    "Random Forest":       "random_forest",
    "XGBoost":             "xgboost",
    "LightGBM":            "lightgbm",
}

# ── Dataset 2: Thông tin tín dụng (train 20260430 / test 20260531) ───────────
DATA_CREDIT_INFO_TRAIN = _resolve_raw("Thông tin tín dụng 20260430.xlsx")
DATA_CREDIT_INFO_TEST  = _resolve_raw("Thông tin tín dụng 20260531.xlsx")

# Kỳ báo cáo thứ ba, THẬT — ngân hàng cung cấp bổ sung ngày 2026-09-12, cho
# phép xây cặp chuyển nhóm nợ thứ hai (20260531 -> 20260630) bằng dữ liệu
# thật, thay vì kỳ mô phỏng DATA_CREDIT_INFO_SIM ở dưới. Dùng trong
# src/generate_transition_report_pair2.py.
DATA_CREDIT_INFO_TEST2 = _resolve_raw("Thông tin tín dụng 20260630.xlsx")

CREDIT_INFO_TARGET_COL = "Nhóm nợ tự phân loại"

# 20260531 (test) chỉ có 33/41 cột của 20260430 (train) — tập đặc trưng chỉ
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

# ── Bài toán bổ sung: dự báo chuyển nhóm nợ trong 31 ngày (bộ B) ──────────────
# Ghép 20260430 (T) với 20260531 (T+31 ngày) qua "Số khế ước" — 98.968/100.617
# khoản vay khớp được (98,4%); trong đó chỉ 265 khoản vay (0,27%) chuyển sang
# nhóm nợ cao hơn — sự kiện hiếm, không dùng lại CREDIT_INFO_SMOTE_STRATEGY
# (thiết kế cho bài toán 5 lớp với tỉ lệ mất cân bằng ~1-3%, không phải 0,27%).
TRANSITION_ID_COL = "Số khế ước"

# 0-based, minority class = 1 (TRANSITION_WORSENED=1). Áp dụng trên phần train
# (80% của ~98.968 dòng, ~212 sự kiện dương) khi imbalance_strategy="custom".
# Mục tiêu nâng vừa phải, tránh SMOTE áp đảo mẫu dương gốc quá thưa.
TRANSITION_SMOTE_STRATEGY = {1: 1200}

TRANSITION_CV_SPLITS = 5
TRANSITION_CV_REPEATS = 10  # đồng bộ 5x10 với repeated_stratified_recall() của bài toán 5 lớp

# Chỉ dùng cho Phụ lục minh họa mở rộng đánh giá đa kỳ — KHÔNG phải dữ liệu
# thật, KHÔNG dùng trong pipeline chính (generate_credit_info_report.py,
# generate_transition_report.py, Streamlit app). Sinh bởi
# src/generate_synthetic_period.py từ cặp kỳ thật 20260430/20260531.
DATA_CREDIT_INFO_SIM = _resolve_raw("Thông tin tín dụng 20260630 (mô phỏng).xlsx")
