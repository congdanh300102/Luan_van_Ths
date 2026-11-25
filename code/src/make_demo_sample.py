"""Tạo bộ dữ liệu MẪU ẨN DANH cho bản Streamlit online (code/data/sample/).

Chỉ dùng để demo giao diện. KHÔNG dùng cho số liệu của luận văn hay bài báo:
mẫu đã bị lấy mẫu lại và làm nhiễu nên các tỷ lệ, hệ số không còn khớp dữ liệu gốc.

Cách ẩn danh:
  * Bộ A (Data_credit_rating_VN.xlsx): lấy mẫu phân tầng theo NHOMNOMOI; mã chi
    nhánh (ORGNBR, PARENTORGNBR) đổi sang số thứ tự giả; số dư nhân nhiễu
    ngẫu nhiên trong [0,9; 1,1].
  * Bộ B (Thông tin tín dụng 20260430 / 20260531): lấy mẫu theo KHÁCH HÀNG
    (giữ trọn các hợp đồng của khách hàng được chọn ở cả hai kỳ); tên khách hàng,
    mã khách hàng, số hợp đồng, số khế ước, mã chi nhánh đổi sang mã giả nhất
    quán giữa hai kỳ (vẫn ghép được kỳ 4 -> kỳ 5); các trường số tiền nhân cùng
    một hệ số nhiễu [0,9; 1,1] cho mỗi khế ước ở cả hai kỳ. Giữ toàn bộ khách
    hàng có khế ước chuyển nhóm xấu đi để trang dự báo chuyển nhóm có đủ sự kiện,
    nên tỷ lệ chuyển nhóm trong mẫu CAO hơn thực tế.

Chạy:  cd code && python3 -m src.make_demo_sample
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "sample"
rng = np.random.default_rng(20261126)


def noise(n):
    return rng.uniform(0.9, 1.1, n)


def recode(values, prefix=None, width=6):
    """Ánh xạ giá trị gốc -> mã giả (thứ tự ngẫu nhiên, không suy ngược được)."""
    uniq = pd.Series(pd.unique(pd.Series(values).dropna()))
    codes = rng.permutation(len(uniq)) + 1
    return {v: (f"{prefix}{c:0{width}d}" if prefix else int(c)) for v, c in zip(uniq, codes)}


# ── Bộ A ─────────────────────────────────────────────────────────────────────
def sample_a(n_per_group=1200):
    a = pd.read_excel(RAW / "Data_credit_rating_VN.xlsx")
    a = (a.groupby("NHOMNOMOI", group_keys=False)
           .apply(lambda g: g.sample(min(len(g), n_per_group), random_state=1)))
    a = a.sample(frac=1, random_state=2).reset_index(drop=True)
    for c in ["ORGNBR", "PARENTORGNBR"]:
        a[c] = a[c].map(recode(a[c]))
    f = noise(len(a))
    a["BASE_BAL"] = (a["BASE_BAL"] * f).round(-3).astype("int64")
    a["CURR_BAL"] = (a["CURR_BAL"] * f).round(-3)
    a["DUNO_QD"] = a["CURR_BAL"]
    a.to_excel(OUT / "Data_credit_rating_VN.xlsx", index=False)
    print("A:", a.shape, a["NHOMNOMOI"].value_counts().sort_index().to_dict())


# ── Bộ B ─────────────────────────────────────────────────────────────────────
CUST, NAME, CONTRACT, KU, BR, TGT = ("Mã khách hàng do TCTD cấp", "Tên khách hàng",
                                     "Số hợp đồng tín dụng", "Số khế ước",
                                     "Mã chi nhánh TCTD", "Nhóm nợ tự phân loại")
MONEY = ["Số dư nợ theo nguyên tệ", "Dư nợ gốc chậm trả thực tế", "Số tiền lãi chậm trả thực tế",
         "Số tiền nợ gốc cơ cấu", "Số tiền nợ lãi cơ cấu", "Lãi phải thu hạch toán nội bảng",
         "Lãi chưa thu hạch toán ngoại bảng", "Dự phòng cụ thể phải trích nội bảng",
         "Dự phòng cụ thể đã trích nội bảng"]


def sample_b(frac_other=0.08):
    a = pd.read_excel(RAW / "Thông tin tín dụng 20260430.xlsx")
    b = pd.read_excel(RAW / "Thông tin tín dụng 20260531.xlsx")
    for d in (a, b):
        d[KU] = d[KU].astype(str)

    m = a[[KU, CUST, TGT]].merge(b[[KU, TGT]], on=KU, suffixes=("", "_5"))
    event_cust = set(m.loc[m[TGT + "_5"] > m[TGT], CUST])
    worst = pd.concat([a[[CUST, TGT]], b[[CUST, TGT]]]).groupby(CUST)[TGT].max()
    others = worst[~worst.index.isin(event_cust)]
    picked = (others.groupby(others, group_keys=False)
                    .apply(lambda s: s.sample(max(1, int(round(len(s) * frac_other))), random_state=3)))
    keep = event_cust | set(picked.index)
    a, b = a[a[CUST].isin(keep)].copy(), b[b[CUST].isin(keep)].copy()

    both = pd.concat([a, b])
    # mã khách hàng giả bắt đầu từ 900.000.000 để không trùng ngẫu nhiên với mã thật
    maps = {CUST: {k: 900_000_000 + v for k, v in recode(both[CUST]).items()}, NAME: recode(both[CUST], "KH_"),
            CONTRACT: recode(both[CONTRACT], "HD_", 7), KU: recode(both[KU], "KU_", 7),
            BR: recode(both[BR])}
    fac = dict(zip(pd.unique(both[KU]), noise(both[KU].nunique())))
    for d in (a, b):
        d[NAME] = d[CUST].map(maps[NAME])
        for c in (CUST, CONTRACT, BR):
            d[c] = d[c].map(maps[c])
        f = d[KU].map(fac).values
        for c in MONEY:
            d[c] = (pd.to_numeric(d[c], errors="coerce") * f).round().astype(d[c].dtype, errors="ignore")
        d[KU] = d[KU].map(maps[KU])
    a = a.sample(frac=1, random_state=4)
    b = b.sample(frac=1, random_state=5)
    a.to_excel(OUT / "Thông tin tín dụng 20260430.xlsx", index=False)
    b.to_excel(OUT / "Thông tin tín dụng 20260531.xlsx", index=False)
    ev = a[[KU, TGT]].merge(b[[KU, TGT]], on=KU, suffixes=("", "_5"))
    print("B 30/4:", a.shape, a[TGT].value_counts().sort_index().to_dict())
    print("B 31/5:", b.shape, b[TGT].value_counts().sort_index().to_dict())
    print("B khớp 4->5:", len(ev), "sự kiện chuyển nhóm:", int((ev[TGT + "_5"] > ev[TGT]).sum()))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sample_a()
    sample_b()
