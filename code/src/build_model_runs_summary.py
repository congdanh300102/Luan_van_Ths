"""Dựng lại `code/results/model_runs_summary.xlsx`.

Lý do tồn tại: bản workbook cũ có phần **Bộ B** lấy từ bộ dữ liệu fct_l.xlsx
(đã bị loại khỏi luận văn). Script này giữ nguyên toàn bộ số liệu **Bộ A** đọc
lại từ workbook hiện có, rồi **thay thế mọi số liệu Bộ B** bằng số liệu hiện
hành sinh từ `src/generate_credit_info_report.py`
(train 20260430 / kiểm định 20% / kiểm tra kỳ sau 20260531):

    cd code && python3 -m src.build_model_runs_summary

Script chỉ đọc/ghi file trong `code/results/`, không huấn luyện lại mô hình —
nên các con số hoàn toàn khớp với các file `credit_info_*.csv` đang được trích
dẫn trong luận văn.
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
XLSX = RESULTS_DIR / "model_runs_summary.xlsx"

# Cấu hình mặc định của từng mô hình trong `src/models.py::build_pipeline`
# (dùng để điền cột "Cấu hình mặc định" cho sheet GridSearch_MacDinh_vs_ToiUu).
DEFAULT_PARAMS = {
    "decision_tree": "{'max_depth': 10, 'min_samples_leaf': 10}",
    "random_forest": "{'n_estimators': 300, 'max_depth': 15, 'min_samples_leaf': 5}",
    "xgboost": "{'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 400}",
    "lightgbm": "{'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 400}",
    "catboost": "{'depth': 6, 'learning_rate': 0.05, 'iterations': 400}",
    "logistic": "{'C': 1.0}",
}

# Thứ tự trình bày 6 mô hình, dùng chung cho mọi sheet so sánh.
MODEL_ORDER = ["Logistic Regression", "Decision Tree", "Random Forest",
               "XGBoost", "LightGBM", "CatBoost"]

GRID_COLS = ["Bộ dữ liệu", "Mô hình", "model_key", "config_id", "params",
             "Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC macro",
             "Trạng thái", "Lỗi"]

# Sheet Bộ A được giữ nguyên; tên cũ → tên mới (nếu có đổi) để script chạy lại
# được trên cả workbook cũ và workbook đã dựng lại.
A_SHEETS = ["A_SoSanhMoHinh", "A_ChiTietTheoNhom", "A_TopK_XGBoost",
            "A_GroupImportance", "IV_BoA_13bien"]
GRID_DETAIL_ALIASES = ["GridSearch_ChiTiet125", "GridSearch_ChiTiet130"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _csv(name):
    return pd.read_csv(RESULTS_DIR / f"credit_info_{name}.csv")


def _by_model_order(df, col="Mô hình"):
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=MODEL_ORDER, ordered=True)
    return df.sort_values(col).reset_index(drop=True)


def _pick_sheet(book, names):
    for n in names:
        if n in book:
            return book[n]
    raise KeyError(f"Không tìm thấy sheet nào trong {names} ở {XLSX.name}")


# ── Bộ B: các sheet so sánh mô hình ─────────────────────────────────────────
def build_b_sosanh():
    """6 mô hình × 2 tầng đánh giá (kiểm định 20% kỳ 20260430 / kỳ sau 20260531)."""
    val = _csv("model_comparison_validation").rename(columns={"model": "Mô hình"})
    hol = _csv("model_comparison_holdout").rename(columns={"model": "Mô hình"})
    ren = {"accuracy": "Accuracy", "f1_macro": "Macro-F1",
           "f1_weighted": "Weighted-F1", "roc_auc": "ROC-AUC macro"}
    val = val.rename(columns={k: f"{v} (kiểm định)" for k, v in ren.items()})
    hol = hol.rename(columns={k: f"{v} (kỳ sau 20260531)" for k, v in ren.items()})
    out = val.merge(hol.drop(columns=["model_key"]), on="Mô hình")
    out["Trạng thái"] = "OK"
    cols = (["Mô hình", "model_key", "Trạng thái"]
            + [f"{v} (kiểm định)" for v in ren.values()]
            + [f"{v} (kỳ sau 20260531)" for v in ren.values()])
    return _by_model_order(out[cols])


def build_b_chitiet():
    """Precision/Recall/F1 theo nhóm nợ cho cả 6 mô hình, trên 2 tầng đánh giá.

    Nguồn là credit_info_by_class_all_models_*.csv (sinh bởi
    src/generate_credit_info_by_class.py). Nếu chưa có file đó thì rơi về
    credit_info_best_model_by_class_*.csv — chỉ có mô hình tốt nhất.
    """
    tags = [("Kiểm định (20% kỳ 20260430)", "validation"),
            ("Kiểm tra kỳ sau (20260531)", "holdout")]
    cols = ["Tầng đánh giá", "Mô hình", "Nhóm", "Support", "Precision", "Recall", "F1"]

    if (RESULTS_DIR / "credit_info_by_class_all_models_validation.csv").exists():
        frames = []
        for tag, suffix in tags:
            df = _csv(f"by_class_all_models_{suffix}")
            df = df[df["nhom"].str.startswith("Nhóm")].copy()
            df.insert(0, "Tầng đánh giá", tag)
            df = df.rename(columns={"model": "Mô hình", "nhom": "Nhóm", "support": "Support",
                                    "precision": "Precision", "recall": "Recall", "f1": "F1"})
            frames.append(_by_model_order(df))
        return pd.concat(frames, ignore_index=True)[cols]

    frames = []
    for tag, suffix in tags:
        df = _csv(f"best_model_by_class_{suffix}").rename(columns={"Unnamed: 0": "Nhóm"})
        df = df[df["Nhóm"].str.startswith("Nhóm")].copy()
        df.insert(0, "Tầng đánh giá", tag)
        df.insert(1, "Mô hình", "LightGBM")
        df["support"] = df["support"].astype(int)
        frames.append(df.rename(columns={"support": "Support", "precision": "Precision",
                                         "recall": "Recall", "f1-score": "F1"}))
    return pd.concat(frames, ignore_index=True)[cols]


def build_b_topk():
    """Đường cong hiệu năng theo số đặc trưng top-k (mô hình tốt nhất: LightGBM)."""
    return _csv("topk").rename(columns={
        "accuracy": "Accuracy", "f1_macro": "Macro-F1",
        "f1_weighted": "Weighted-F1", "roc_auc": "ROC-AUC macro"})


def build_b_group_importance():
    return _csv("group_importance").rename(columns={
        "group": "Nhóm nghiệp vụ", "importance": "Importance", "pct": "Tỷ trọng (%)"})


def build_b_priority():
    return _csv("priority")


def build_b_iv():
    return _csv("iv_table")


# ── Bộ B: các sheet grid search ─────────────────────────────────────────────
def build_grid_detail(a_rows):
    b = _csv("grid_search_results").rename(columns={
        "model": "Mô hình", "accuracy": "Accuracy", "f1_macro": "Macro-F1",
        "f1_weighted": "Weighted-F1", "roc_auc": "ROC-AUC macro", "status": "Trạng thái"})
    b.insert(0, "Bộ dữ liệu", "B")
    b["Lỗi"] = b["Trạng thái"].where(b["Trạng thái"] != "OK")
    b.loc[b["Trạng thái"] != "OK", "Trạng thái"] = "ERROR"
    return pd.concat([a_rows[GRID_COLS], b[GRID_COLS]], ignore_index=True)


def build_grid_best(a_rows):
    b = _csv("grid_search_best").rename(columns={
        "model": "Mô hình", "accuracy": "Accuracy", "f1_macro": "Macro-F1",
        "f1_weighted": "Weighted-F1", "roc_auc": "ROC-AUC macro", "status": "Trạng thái"})
    b.insert(0, "Bộ dữ liệu", "B")
    b["Lỗi"] = pd.NA
    b = _by_model_order(b)
    return pd.concat([a_rows[GRID_COLS], b[GRID_COLS]], ignore_index=True)


def build_grid_sensitivity(a_rows):
    sens = _csv("grid_search_sensitivity").rename(columns={"model": "Mô hình"})
    n_cfg = (_csv("grid_search_results").groupby("model").size()
             .rename("Số cấu hình đã thử").reset_index()
             .rename(columns={"model": "Mô hình"}))
    b = sens.merge(n_cfg, on="Mô hình")
    b.insert(0, "Bộ dữ liệu", "B")
    b = b.rename(columns={
        "min": "Macro-F1 nhỏ nhất", "max": "Macro-F1 lớn nhất",
        "mean": "Macro-F1 trung bình", "std": "Độ lệch chuẩn",
        "range": "range (max-min)"})
    b = b[a_rows.columns].sort_values("range (max-min)", ascending=False)
    return pd.concat([a_rows, b], ignore_index=True)


def build_grid_default_vs_best(a_rows):
    dvb = _csv("grid_search_default_vs_best").rename(columns={"model": "Mô hình"})
    best = _csv("grid_search_best")[["model", "params"]].rename(
        columns={"model": "Mô hình", "params": "Cấu hình tốt nhất (Grid Search)"})
    b = dvb.merge(best, on="Mô hình")
    b.insert(0, "Bộ dữ liệu", "B")
    b["Cấu hình mặc định"] = b["model_key"].map(DEFAULT_PARAMS)
    b["Macro-F1 (mặc định)"] = b["f1_macro_default"].round(4)
    b["Macro-F1 (tốt nhất)"] = b["f1_macro_best"].round(4)
    b["Cải thiện (tuyệt đối)"] = (b["f1_macro_best"] - b["f1_macro_default"]).round(4)
    b["Cải thiện (%)"] = b["improve_pct"]
    b = _by_model_order(b)[a_rows.columns]
    return pd.concat([a_rows, b], ignore_index=True)


# ── Sheet thuyết minh nguồn số liệu ─────────────────────────────────────────
def build_source_sheet(sheet_rows):
    return pd.DataFrame(sheet_rows, columns=["Sheet", "Bộ dữ liệu", "Nguồn số liệu", "Ghi chú"])


def main():
    if not XLSX.exists():
        raise SystemExit(f"Không tìm thấy {XLSX} — cần workbook cũ để lấy lại số liệu Bộ A.")

    log(f"Đọc workbook hiện có: {XLSX.name}")
    book = pd.read_excel(XLSX, sheet_name=None)
    log(f"  {len(book)} sheet: {list(book)}")

    # Số liệu Bộ A — giữ nguyên 100%.
    a_keep = {name: book[name] for name in A_SHEETS}
    grid_detail_old = _pick_sheet(book, GRID_DETAIL_ALIASES)
    a_detail = grid_detail_old[grid_detail_old["Bộ dữ liệu"] == "A"].copy()
    a_best = book["GridSearch_TotNhat"].query("`Bộ dữ liệu` == 'A'").copy()
    a_sens = book["GridSearch_DoNhay"].query("`Bộ dữ liệu` == 'A'").copy()
    a_dvb = book["GridSearch_MacDinh_vs_ToiUu"].query("`Bộ dữ liệu` == 'A'").copy()
    log(f"Giữ lại Bộ A: {len(a_detail)} cấu hình grid, {len(a_best)} best-per-model.")

    # Số liệu Bộ B — dựng lại từ credit_info_*.csv.
    log("Dựng lại số liệu Bộ B từ credit_info_*.csv…")
    b_sosanh = build_b_sosanh()
    b_chitiet = build_b_chitiet()
    b_topk = build_b_topk()
    b_group = build_b_group_importance()
    b_priority = build_b_priority()
    b_iv = build_b_iv()
    grid_detail = build_grid_detail(a_detail)
    grid_best = build_grid_best(a_best)
    grid_sens = build_grid_sensitivity(a_sens)
    grid_dvb = build_grid_default_vs_best(a_dvb)

    n_b_cfg = int((grid_detail["Bộ dữ liệu"] == "B").sum())
    n_a_cfg = int((grid_detail["Bộ dữ liệu"] == "A").sum())
    detail_name = f"GridSearch_ChiTiet{n_a_cfg + n_b_cfg}"
    iv_name = f"IV_BoB_{len(b_iv)}bien"
    log(f"  Grid search: {n_a_cfg} cấu hình Bộ A + {n_b_cfg} cấu hình Bộ B → {detail_name}")
    log(f"  IV Bộ B: {len(b_iv)} biến → {iv_name}")

    src = [
        ("A_SoSanhMoHinh", "A", "Trang 2 — app Streamlit", "Giữ nguyên từ workbook cũ"),
        ("A_ChiTietTheoNhom", "A", "Trang 2 — app Streamlit", "Giữ nguyên từ workbook cũ"),
        ("A_TopK_XGBoost", "A", "Trang 7 — app Streamlit", "Giữ nguyên từ workbook cũ"),
        ("A_GroupImportance", "A", "Trang 7 — app Streamlit", "Giữ nguyên từ workbook cũ"),
        ("B_SoSanhMoHinh", "B", "credit_info_model_comparison_validation.csv + _holdout.csv",
         "6 mô hình × 2 tầng đánh giá (Accuracy / Macro-F1 / Weighted-F1 / ROC-AUC macro)"),
        ("B_ChiTietTheoNhom", "B",
         "credit_info_by_class_all_models_validation.csv + _holdout.csv",
         "Đủ 6 mô hình × 5 nhóm × 2 tầng đánh giá (sinh bởi generate_credit_info_by_class.py)"),
        ("B_TopK_LightGBM", "B", "credit_info_topk.csv",
         "Mô hình tốt nhất là LightGBM (Bộ A dùng XGBoost); k = 3…20"),
        ("B_GroupImportance", "B", "credit_info_group_importance.csv",
         "6 nhóm nghiệp vụ (bộ B cũ có 9 nhóm)"),
        ("B_UuTienDauTu", "B", "credit_info_priority.csv", "Importance × % dữ liệu thiếu"),
        ("IV_BoA_13bien", "A", "iv_table_A.csv", "Giữ nguyên từ workbook cũ"),
        (iv_name, "B", "credit_info_iv_table.csv",
         f"{len(b_iv)} biến sau khi lọc leakage/trùng lặp (bộ B cũ có 98 biến)"),
        (detail_name, "A + B", "GridSearch cũ (Bộ A) + credit_info_grid_search_results.csv (Bộ B)",
         f"{n_a_cfg} cấu hình Bộ A + {n_b_cfg} cấu hình Bộ B; cả 2 bộ đều có Accuracy"),
        ("GridSearch_TotNhat", "A + B", "… + credit_info_grid_search_best.csv", "Best Macro-F1 mỗi mô hình"),
        ("GridSearch_DoNhay", "A + B", "… + credit_info_grid_search_sensitivity.csv", "min/max/mean/std Macro-F1"),
        ("GridSearch_MacDinh_vs_ToiUu", "A + B", "… + credit_info_grid_search_default_vs_best.csv",
         "Cấu hình mặc định lấy từ src/models.py::build_pipeline"),
    ]

    sheets = {
        "Nguon_DuLieu": build_source_sheet(src),
        "A_SoSanhMoHinh": a_keep["A_SoSanhMoHinh"],
        "A_ChiTietTheoNhom": a_keep["A_ChiTietTheoNhom"],
        "A_TopK_XGBoost": a_keep["A_TopK_XGBoost"],
        "A_GroupImportance": a_keep["A_GroupImportance"],
        "B_SoSanhMoHinh": b_sosanh,
        "B_ChiTietTheoNhom": b_chitiet,
        "B_TopK_LightGBM": b_topk,
        "B_GroupImportance": b_group,
        "B_UuTienDauTu": b_priority,
        "IV_BoA_13bien": a_keep["IV_BoA_13bien"],
        iv_name: b_iv,
        detail_name: grid_detail,
        "GridSearch_TotNhat": grid_best,
        "GridSearch_DoNhay": grid_sens,
        "GridSearch_MacDinh_vs_ToiUu": grid_dvb,
    }

    backup = XLSX.with_suffix(".xlsx.bak")
    shutil.copy2(XLSX, backup)
    log(f"Đã sao lưu workbook cũ → {backup.name}")

    with pd.ExcelWriter(XLSX, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)
            log(f"  ghi sheet {name} ({df.shape[0]} dòng × {df.shape[1]} cột)")

    log(f"HOÀN TẤT — {XLSX} có {len(sheets)} sheet, toàn bộ số liệu Bộ B đã là bộ hiện hành.")


if __name__ == "__main__":
    main()
