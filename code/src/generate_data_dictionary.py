"""Sinh file dictionary (mô tả cột) cho 2 bộ dữ liệu Thông tin tín dụng.

Chạy: python code/src/generate_data_dictionary.py
Đầu ra: code/results/data_dictionary_20260430.xlsx
        code/results/data_dictionary_20260507.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.credit_info_preprocessing import (
    TARGET_COL, ID_LEAKAGE_COLS, DATE_COLS,
    RAW_NUMERICAL_COLS, RAW_CATEGORICAL_COLS,
    credit_info_feature_info, feature_to_group,
)

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

FILES = {
    "20260430": RAW_DIR / "Thông tin tín dụng 20260430.xlsx",
    "20260507": RAW_DIR / "Thông tin tín dụng 20260507.xlsx",
}

FEATURE_DESC = credit_info_feature_info(RAW_NUMERICAL_COLS, RAW_CATEGORICAL_COLS)

MANUAL_DESC = {
    TARGET_COL: "Biến mục tiêu — nhóm nợ do TCTD tự phân loại (1=Đủ tiêu chuẩn … 5=Có khả năng mất vốn)",
    "Mã khách hàng do TCTD cấp": "Mã định danh khách hàng nội bộ (ID)",
    "Tên khách hàng": "Tên khách hàng (định danh trực tiếp)",
    "Số hợp đồng tín dụng": "Số hợp đồng tín dụng (ID)",
    "Số khế ước": "Số khế ước nhận nợ (ID)",
    "Nhóm nợ phân loại sau khi tham chiếu CIC": "Nhóm nợ sau khi đối chiếu CIC — loại: rò rỉ nhãn (suy ra gần như trực tiếp từ target)",
    "Dư nợ gốc chậm trả thực tế": "Dư nợ gốc chậm trả — loại: căn cứ trực tiếp để xếp nhóm nợ (rò rỉ nhãn)",
    "Ngày chậm trả nợ gốc": "Ngày chậm trả nợ gốc — loại: rò rỉ nhãn",
    "Số tiền lãi chậm trả thực tế": "Số tiền lãi chậm trả — loại: rò rỉ nhãn",
    "Ngày chậm trả nợ lãi": "Ngày chậm trả nợ lãi — loại: rò rỉ nhãn",
    "Dự phòng cụ thể phải trích nội bảng": "Dự phòng cụ thể phải trích — tỉ lệ cố định theo nhóm nợ (rò rỉ nhãn)",
    "Dự phòng cụ thể đã trích nội bảng": "Dự phòng cụ thể đã trích — rò rỉ nhãn",
    "ETL_DATE": "Ngày chốt dữ liệu (mốc trích xuất) — dùng làm mốc cố định để sinh đặc trưng thời gian, không đưa thẳng vào model",
    "Ngày giải ngân": "Ngày giải ngân khoản vay — dùng để sinh đặc trưng kỳ hạn/tuổi khoản vay",
    "Ngày kết thúc khế ước": "Ngày đáo hạn khế ước — dùng để sinh đặc trưng kỳ hạn còn lại",
    "Thời điểm truy đòi": "Trùng khớp 100% với Ngày kết thúc khế ước trong dữ liệu quan sát được — không dùng để tránh trùng lặp đặc trưng",
    "Tên chi nhánh TCTD": "Tên chi nhánh cấp tín dụng — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Ngày hiệu lực hợp đồng": "Ngày hiệu lực hợp đồng — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Ngày kết thúc hợp đồng": "Ngày kết thúc hợp đồng — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Thời hạn cấp tín dụng (ngày)": "Thời hạn cấp tín dụng tính theo ngày — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Trạng thái TSBĐ": "Trạng thái tài sản bảo đảm — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Hạn mức tín dụng trên hợp đồng": "Hạn mức tín dụng theo hợp đồng — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Mã tiền tệ.1": "Cột trùng tên với Mã tiền tệ (bản sao/khác nguồn) — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
    "Mục đích sử dụng tiền vay theo lĩnh vực": "Mục đích vay theo lĩnh vực — chỉ có ở 20260430, không dùng làm feature (test thiếu cột)",
}


def classify(col: str) -> str:
    if col == TARGET_COL:
        return "Target"
    if col in ID_LEAKAGE_COLS:
        if col in ("Mã khách hàng do TCTD cấp", "Tên khách hàng", "Số hợp đồng tín dụng", "Số khế ước"):
            return "Loại - ID/khóa"
        return "Loại - rò rỉ nhãn (leakage)"
    if col in DATE_COLS or col == "Thời điểm truy đòi":
        return "Ngày (dùng sinh đặc trưng)"
    if col in RAW_NUMERICAL_COLS:
        return "Feature - numerical"
    if col in RAW_CATEGORICAL_COLS:
        return "Feature - categorical"
    return "Loại - chỉ có ở train, không dùng (test thiếu cột)"


def describe(col: str) -> str:
    return MANUAL_DESC.get(col) or FEATURE_DESC.get(col) or ""


def build_dictionary(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    n = len(df)
    rows = []
    for col in df.columns:
        s = df[col]
        missing = s.isna().sum()
        rows.append({
            "Tên cột": col,
            "Nhóm sử dụng": classify(col),
            "Kiểu dữ liệu (pandas)": str(s.dtype),
            "Số dòng không thiếu": n - missing,
            "Số dòng thiếu": missing,
            "Tỉ lệ thiếu (%)": round(100 * missing / n, 2) if n else 0,
            "Số giá trị duy nhất": s.nunique(dropna=True),
            "Ví dụ giá trị": ", ".join(
                str(v) for v in s.dropna().unique()[:3]
            ) if s.notna().any() else "",
            "Mô tả": describe(col),
        })
    return pd.DataFrame(rows)


def main():
    for tag, path in FILES.items():
        if not path.exists():
            print(f"BỎ QUA (không tìm thấy file): {path}")
            continue
        dic = build_dictionary(path)
        out_path = OUT_DIR / f"data_dictionary_{tag}.xlsx"
        dic.to_excel(out_path, index=False)
        print(f"Đã ghi {out_path} ({len(dic)} cột)")


if __name__ == "__main__":
    main()
