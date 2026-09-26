"""Sinh kỳ dữ liệu MÔ PHỎNG 20260630 cho Phụ lục minh họa mở rộng đánh giá đa
kỳ. Đây KHÔNG phải dữ liệu ngân hàng thật và KHÔNG được dùng trong pipeline
chính (generate_credit_info_report.py, generate_transition_report.py,
Streamlit app) — chỉ phục vụ generate_transition_multiperiod_demo.py.

Phương pháp: ước lượng ma trận xác suất chuyển nhóm nợ P(nhóm tại T+1=j |
nhóm tại T=i) và tỷ lệ loại (tất toán/xóa nợ) theo từng nhóm từ cặp kỳ THẬT
duy nhất hiện có (20260430 → 20260531), rồi áp lên quần thể 20260531 để sinh
nhãn nhóm nợ giả lập cho 20260630. Toàn bộ cột đặc trưng khác giữ nguyên giá
trị của 20260531 — đơn giản hóa có chủ đích: nhãn mô phỏng chỉ phụ thuộc
nhóm nợ hiện tại, không phụ thuộc các đặc trưng còn lại (khác cơ chế sinh
nhãn thật, vốn do hành vi khách hàng quyết định) — cần nêu rõ giới hạn này
khi diễn giải kết quả ở Phụ lục.

Chạy: cd code && py -m src.generate_synthetic_period
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.config import (
    RANDOM_STATE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, TRANSITION_ID_COL,
)
from src.credit_info_preprocessing import build_transition_dataset

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = RAW_DIR / "Thông tin tín dụng 20260630 (mô phỏng).xlsx"
GROUPS = [1, 2, 3, 4, 5]


def log(msg):
    print(msg, flush=True)


def main():
    log("Đang tải cặp kỳ THẬT 20260430 / 20260531 (chưa parse ngày, giữ nguyên "
        "định dạng gốc để ghi lại đúng schema) để ước lượng tham số mô phỏng…")
    df_t = pd.read_excel(DATA_CREDIT_INFO_TRAIN)
    df_t1 = pd.read_excel(DATA_CREDIT_INFO_TEST)

    df_merged = build_transition_dataset(
        df_t, df_t1, id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL,
    )
    matched_ids = set(df_merged[TRANSITION_ID_COL])
    excluded = df_t[~df_t[TRANSITION_ID_COL].isin(matched_ids)]

    # ── Tỷ lệ loại theo từng nhóm nợ tại T (tất toán/xóa nợ giữa 2 kỳ) ──────
    n_all = df_t[CREDIT_INFO_TARGET_COL].astype(int).value_counts()
    n_excl = excluded[CREDIT_INFO_TARGET_COL].dropna().astype(int).value_counts()
    excl_rate = {g: float(n_excl.get(g, 0) / n_all.get(g, 1)) for g in GROUPS}
    log(f"Tỷ lệ loại theo nhóm (ước lượng từ cặp thật): {excl_rate}")

    # ── Ma trận chuyển nhóm P(nhóm tại T+1=j | nhóm tại T=i) ────────────────
    ctab = pd.crosstab(
        df_merged[CREDIT_INFO_TARGET_COL].astype(int),
        df_merged["TRANSITION_GRP_T1"].astype(int),
    ).reindex(index=GROUPS, columns=GROUPS, fill_value=0)
    trans_probs = ctab.div(ctab.sum(axis=1), axis=0).fillna(0.0)
    for g in GROUPS:  # nhóm không có dữ liệu chuyển (hàng toàn 0) → giữ nguyên nhóm cũ
        if trans_probs.loc[g].sum() == 0:
            trans_probs.loc[g, g] = 1.0
    log(f"Ma trận chuyển nhóm ước lượng từ cặp thật:\n{trans_probs}")

    # ── Ghi tham số mô phỏng (dạng gọn, dùng cho Phụ lục) ───────────────────
    rows = []
    for i in GROUPS:
        for j in GROUPS:
            rows.append({"kind": "transition_prob", "group_from": i, "group_to": j,
                        "value": float(trans_probs.loc[i, j])})
    for g in GROUPS:
        rows.append({"kind": "exclusion_rate", "group_from": g, "group_to": None,
                    "value": excl_rate[g]})
    rows.append({"kind": "param", "group_from": "random_state", "group_to": None,
                "value": RANDOM_STATE})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "synthetic_20260630_method.csv",
                              index=False, encoding="utf-8-sig")

    # ── Áp dụng lên quần thể 20260531 để sinh 20260630 mô phỏng ─────────────
    log("Đang sinh kỳ mô phỏng 20260630 từ quần thể 20260531…")
    df_t1 = df_t1.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()
    rng = np.random.default_rng(RANDOM_STATE)
    cur_group = df_t1[CREDIT_INFO_TARGET_COL].astype(int).values

    drop_prob = np.array([excl_rate.get(g, 0.0) for g in cur_group])
    keep_mask = rng.random(len(df_t1)) >= drop_prob

    new_group = np.empty(len(df_t1), dtype=int)
    for g in GROUPS:
        idx = np.where(cur_group == g)[0]
        if len(idx) == 0:
            continue
        probs = trans_probs.loc[g, GROUPS].values.astype(float)
        probs = probs / probs.sum()
        new_group[idx] = rng.choice(GROUPS, size=len(idx), p=probs)

    df_sim = df_t1.loc[keep_mask].copy()
    df_sim[CREDIT_INFO_TARGET_COL] = new_group[keep_mask]
    df_sim["ETL_DATE"] = 20260630

    n_dropped = len(df_t1) - len(df_sim)
    log(f"Quần thể 20260531 (đã loại NaN target): {len(df_t1):,} dòng → "
        f"mô phỏng 20260630: {len(df_sim):,} dòng "
        f"(loại {n_dropped:,}, {n_dropped / len(df_t1) * 100:.2f}%)")
    log(f"Phân bố nhóm nợ mô phỏng 20260630:\n"
        f"{df_sim[CREDIT_INFO_TARGET_COL].value_counts().sort_index()}")

    df_sim.to_excel(OUT_PATH, index=False)
    log(f"Đã ghi: {OUT_PATH}")
    log("LƯU Ý: đây là dữ liệu MÔ PHỎNG, chỉ dùng cho Phụ lục minh họa — "
        "không đưa vào pipeline chính hay xem như dữ liệu ngân hàng thật.")


if __name__ == "__main__":
    main()
