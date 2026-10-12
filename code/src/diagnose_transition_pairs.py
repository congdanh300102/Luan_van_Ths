"""Chẩn đoán cặp kỳ 2: loại trừ khả năng lỗi xử lý trước khi kết luận
"tín hiệu dự báo chuyển nhóm không ổn định theo thời gian".

Bảy kiểm tra (Mục IV.3 của kế hoạch chỉnh sửa luận văn):

  D1  Tính toàn vẹn dữ liệu ba kỳ: số dòng, khóa trùng lặp, danh mục cột,
      ETL_DATE, tỷ lệ khớp khóa của từng cặp.
  D2  Ma trận chuyển nhóm và phân bố nhãn của hai cặp, đặt cạnh nhau.
  D3  Tỷ lệ thiếu của 20 đặc trưng ở từng kỳ.
  D4  PSI (Population Stability Index) của từng đặc trưng giữa kỳ T của
      cặp 1 (20260430) và kỳ T của cặp 2 (20260531).
  D5  AUC đơn biến của từng đặc trưng số so với nhãn chuyển nhóm, trên
      từng cặp — cho biết tín hiệu của cặp 1 nằm ở biến nào và có còn ở
      cặp 2 hay không.
  D6  Chuyển giao hai chiều: huấn luyện cặp 1 chấm cặp 2, VÀ huấn luyện
      cặp 2 chấm cặp 1. Nếu chiều ngược lại vẫn tốt thì đặc trưng của cặp 2
      bình thường, vấn đề nằm ở nhãn của cặp 2.
  D7  AP theo loại chuyển nhóm (1->2 so với các loại còn lại) trên từng cặp.

Chạy:  cd code && py -m src.diagnose_transition_pairs
Output: code/results/diagnose_*.csv
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST, DATA_CREDIT_INFO_TEST2,
    CREDIT_INFO_TARGET_COL, TRANSITION_ID_COL, RANDOM_STATE,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, build_transition_dataset,
    TRANSITION_LABEL_COL, TRANSITION_GRP_T1_COL,
)
from src.models import build_pipeline

RES = Path(__file__).parent.parent / "results"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """PSI với các mốc chia lấy từ phân vị của `expected`.

    Giá trị thiếu được coi là một nhóm riêng, vì thay đổi tỷ lệ thiếu cũng là
    một dạng dịch chuyển phân phối cần phát hiện.
    """
    e, a = pd.Series(expected), pd.Series(actual)
    e_na, a_na = e.isna().mean(), a.isna().mean()
    e_v, a_v = e.dropna(), a.dropna()
    if len(e_v) == 0 or len(a_v) == 0:
        return np.nan
    numeric = pd.api.types.is_numeric_dtype(e_v)
    edges = (np.unique(np.quantile(e_v, np.linspace(0, 1, n_bins + 1)))
             if numeric else np.array([0.0]))
    if not numeric or len(edges) < 3:
        # biến phân loại hoặc gần như hằng số: so sánh trực tiếp tỷ lệ mỗi giá trị
        vals = sorted(set(e_v.unique()) | set(a_v.unique()), key=str)
        e_p = np.array([(e_v == v).mean() for v in vals]) * (1 - e_na)
        a_p = np.array([(a_v == v).mean() for v in vals]) * (1 - a_na)
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        e_p = np.histogram(e_v, bins=edges)[0] / len(e_v) * (1 - e_na)
        a_p = np.histogram(a_v, bins=edges)[0] / len(a_v) * (1 - a_na)
    e_p = np.append(e_p, e_na)
    a_p = np.append(a_p, a_na)
    eps = 1e-6
    e_p, a_p = np.clip(e_p, eps, None), np.clip(a_p, eps, None)
    return float(((a_p - e_p) * np.log(a_p / e_p)).sum())


def build_pair(df_t_raw, df_t1_raw, name):
    dm = build_transition_dataset(df_t_raw, df_t1_raw,
                                  id_col=TRANSITION_ID_COL, target_col=CREDIT_INFO_TARGET_COL)
    num_raw, cat_raw = build_raw_feature_pool(dm)
    de, eng = engineer_business_features(dm)
    num_p, cat_p = num_raw + eng, cat_raw
    X, y = prepare_credit_info(de, TRANSITION_LABEL_COL, num_p, cat_p, label_offset=0)
    log(f"{name}: {len(dm):,} khoản vay khớp, {int(y.sum())} sự kiện "
        f"({y.mean()*100:.3f}%), {X.shape[1]} đặc trưng")
    return dm, de, X, y, num_p, cat_p


def main():
    log("Đang tải ba kỳ báo cáo…")
    raws = {
        "20260430": load_credit_info(DATA_CREDIT_INFO_TRAIN),
        "20260531": load_credit_info(DATA_CREDIT_INFO_TEST),
        "20260630": load_credit_info(DATA_CREDIT_INFO_TEST2),
    }

    # ── D1. Toàn vẹn dữ liệu ba kỳ ──────────────────────────────────────────
    rows = []
    for k, df in raws.items():
        etl = df["ETL_DATE"].dropna().unique() if "ETL_DATE" in df.columns else []
        rows.append({
            "ky": k, "so_dong": len(df), "so_cot": df.shape[1],
            "khoa_duy_nhat": df[TRANSITION_ID_COL].nunique(),
            "khoa_trung_lap": int(df[TRANSITION_ID_COL].duplicated().sum()),
            "etl_date": "; ".join(str(pd.Timestamp(v).date()) for v in etl[:3]),
            "so_etl_date": len(etl),
            "target_thieu": int(df[CREDIT_INFO_TARGET_COL].isna().sum()),
        })
    d1 = pd.DataFrame(rows)
    cols_same = len(set(tuple(sorted(df.columns)) for df in raws.values())) == 1
    d1["danh_muc_cot_trung_khop_3_ky"] = cols_same
    d1.to_csv(RES / "diagnose_d1_integrity.csv", index=False, encoding="utf-8-sig")
    log(f"D1 toàn vẹn dữ liệu (danh mục cột trùng khớp: {cols_same}):\n{d1}")

    pairs = {}
    for name, (a, b) in {"cap1": ("20260430", "20260531"),
                         "cap2": ("20260531", "20260630")}.items():
        pairs[name] = build_pair(raws[a], raws[b], f"{name} ({a}->{b})")

    # ── D2. Ma trận chuyển nhóm và phân bố nhãn ─────────────────────────────
    frames = []
    for name, (dm, *_rest) in pairs.items():
        m = pd.crosstab(dm[CREDIT_INFO_TARGET_COL].astype(int),
                        dm[TRANSITION_GRP_T1_COL].astype(int))
        m.index.name, m.columns.name = "nhom_T", "nhom_T1"
        frames.append(m.assign(cap=name))
        worse = int((dm[TRANSITION_GRP_T1_COL] > dm[CREDIT_INFO_TARGET_COL]).sum())
        better = int((dm[TRANSITION_GRP_T1_COL] < dm[CREDIT_INFO_TARGET_COL]).sum())
        log(f"D2 {name}: xấu đi {worse}, cải thiện {better}, giữ nguyên "
            f"{len(dm) - worse - better}")
    pd.concat(frames).to_csv(RES / "diagnose_d2_matrices.csv", encoding="utf-8-sig")

    # ── D3/D4. Tỷ lệ thiếu từng kỳ và PSI giữa hai kỳ T ─────────────────────
    X1, X2 = pairs["cap1"][2], pairs["cap2"][2]
    rows = []
    for c in X1.columns:
        rows.append({
            "dac_trung": c,
            "thieu_cap1_pct": X1[c].isna().mean() * 100,
            "thieu_cap2_pct": X2[c].isna().mean() * 100,
            "psi": psi(X1[c], X2[c]),
            "kieu": "số" if pd.api.types.is_numeric_dtype(X1[c]) else "phân loại",
        })
    d34 = pd.DataFrame(rows).sort_values("psi", ascending=False)
    d34.to_csv(RES / "diagnose_d34_missing_psi.csv", index=False, encoding="utf-8-sig")
    log(f"D3/D4 tỷ lệ thiếu và PSI (kỳ T cặp 1 so với kỳ T cặp 2):\n{d34.round(4)}")

    # ── D5. AUC đơn biến của từng đặc trưng số ──────────────────────────────
    rows = []
    for c in X1.columns:
        if not pd.api.types.is_numeric_dtype(X1[c]):
            continue
        r = {"dac_trung": c}
        for name, (_dm, _de, X, y, *_r) in pairs.items():
            v = X[c].fillna(X[c].median())
            r[f"auc_{name}"] = roc_auc_score(y, v) if v.nunique() > 1 else np.nan
        r["chenh_lech"] = abs(r["auc_cap1"] - 0.5) - abs(r["auc_cap2"] - 0.5)
        rows.append(r)
    d5 = pd.DataFrame(rows).sort_values("chenh_lech", ascending=False)
    d5.to_csv(RES / "diagnose_d5_univariate_auc.csv", index=False, encoding="utf-8-sig")
    log(f"D5 AUC đơn biến theo cặp:\n{d5.round(4)}")

    # ── D6. Chuyển giao hai chiều ───────────────────────────────────────────
    def fit_score(train, test):
        _dm, _de, Xtr, ytr, num_p, cat_p = pairs[train]
        _dm2, _de2, Xte, yte, *_r = pairs[test]
        p = build_pipeline("xgboost", cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy="none", eval_metric="logloss")
        p.fit(Xtr, ytr)
        s = p.predict_proba(Xte[Xtr.columns])[:, 1]
        return {"huan_luyen": train, "cham_diem": test, "n_train": len(ytr),
                "n_test": len(yte), "n_pos_test": int(yte.sum()),
                "ap": average_precision_score(yte, s),
                "ap_ngau_nhien": float(yte.mean()),
                "roc_auc": roc_auc_score(yte, s)}

    rows = []
    for tr, te in [("cap1", "cap2"), ("cap2", "cap1"), ("cap1", "cap1"), ("cap2", "cap2")]:
        rows.append(fit_score(tr, te))
        log(f"D6 {tr}->{te}: AP={rows[-1]['ap']:.4f} "
            f"(ngẫu nhiên {rows[-1]['ap_ngau_nhien']:.4f}), ROC-AUC={rows[-1]['roc_auc']:.4f}")
    d6 = pd.DataFrame(rows)
    d6["ghi_chu"] = np.where(d6.huan_luyen == d6.cham_diem,
                             "trùng tập huấn luyện (chỉ để tham chiếu)", "chuyển giao thật")
    d6.to_csv(RES / "diagnose_d6_transfer.csv", index=False, encoding="utf-8-sig")

    # ── D7. AP theo loại chuyển nhóm ────────────────────────────────────────
    rows = []
    for name, (dm, _de, X, y, num_p, cat_p) in pairs.items():
        Xtr, Xte, ytr, yte, _itr, ite = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
        p = build_pipeline("xgboost", cat_p, num_p, random_state=RANDOM_STATE,
                           imbalance_strategy="none", eval_metric="logloss")
        p.fit(Xtr, ytr)
        s = p.predict_proba(Xte)[:, 1]
        g_from = dm[CREDIT_INFO_TARGET_COL].astype(int).values[ite]
        is_1to2 = (g_from == 1) & (yte == 1)
        for lab, keep in [("tất cả sự kiện", np.ones(len(yte), bool)),
                          ("chỉ 1->2", (yte == 0) | is_1to2),
                          ("chỉ 2->3, 3->4, 4->5", (yte == 0) | ((yte == 1) & ~is_1to2))]:
            if yte[keep].sum() == 0:
                continue
            rows.append({"cap": name, "nhom_su_kien": lab, "n": int(keep.sum()),
                         "n_pos": int(yte[keep].sum()),
                         "ap": average_precision_score(yte[keep], s[keep]),
                         "ap_ngau_nhien": float(yte[keep].mean()),
                         "roc_auc": roc_auc_score(yte[keep], s[keep])})
    d7 = pd.DataFrame(rows)
    d7.to_csv(RES / "diagnose_d7_by_event_type.csv", index=False, encoding="utf-8-sig")
    log(f"D7 AP theo loại chuyển nhóm:\n{d7.round(4)}")

    log("HOÀN TẤT. Kết quả ở code/results/diagnose_*.csv")


if __name__ == "__main__":
    main()
