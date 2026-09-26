"""Script tái lập kết quả cho luận văn — Bộ dữ liệu Thông tin tín dụng
(train 20260430 / test 20260531).

Không phải trang Streamlit — chạy độc lập để sinh toàn bộ số liệu/bảng/hình
dùng trong Chương 3-4 (thay thế các bảng/hình cũ dựa trên fct_l.xlsx):

    cd code && DYLD_FALLBACK_LIBRARY_PATH=.../sklearn/.dylibs \
        python3 -m src.generate_credit_info_report

Output:
  code/results/credit_info_*.csv        — các bảng số liệu
  luanvan_latex/figures/eda_credit_info_*.png — các hình EDA
"""
import sys
import time
import warnings
from pathlib import Path
from itertools import product

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from config.config import (
    RANDOM_STATE, TEST_SIZE, DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_SMOTE_STRATEGY, CREDIT_INFO_NHOMNO_LABELS,
)
from src.credit_info_preprocessing import (
    load_credit_info, build_raw_feature_pool, engineer_business_features,
    prepare_credit_info, transform_credit_info, feature_to_group, FEATURE_GROUPS,
    ID_LEAKAGE_COLS, DATE_COLS,
)
from src.iv_analysis import compute_iv_table
from src.models import build_pipeline, available_models
from src.evaluation import compute_metrics
from src.feature_selection import (
    importance_from_pipeline, group_importance_table, cumulative_importance_curve,
    recommend_k_for_target, evaluate_performance_vs_k, data_investment_priority,
)
from src.explainability import compute_shap_values

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIG_DIR = Path(__file__).parent.parent.parent / "luanvan_latex" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _pct(x):
    return round(float(x) * 100, 2)


def _acc(metrics):
    """Accuracy lấy từ classification_report — compute_metrics() không trả
    riêng chỉ số này nhưng report dạng dict luôn có khoá 'accuracy'."""
    return float(metrics["report"]["accuracy"])


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # ── 1. Load dữ liệu ──────────────────────────────────────────────────────
    log("Đang tải dữ liệu train (20260430) và test (20260531)…")
    df_train_raw = load_credit_info(DATA_CREDIT_INFO_TRAIN)
    df_test_raw = load_credit_info(DATA_CREDIT_INFO_TEST)
    log(f"Train: {df_train_raw.shape}, Test: {df_test_raw.shape}")

    df_valid = df_train_raw.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()
    df_valid[CREDIT_INFO_TARGET_COL] = df_valid[CREDIT_INFO_TARGET_COL].astype(int)
    vc = df_valid[CREDIT_INFO_TARGET_COL].value_counts().sort_index()
    imbalance_ratio = vc.iloc[0] / vc.min()
    log(f"Phân phối nhóm nợ (train, {len(df_valid):,} hồ sơ hợp lệ):\n{vc}")
    log(f"Imbalance ratio: {imbalance_ratio:.1f}x")

    # ── 2. Tập đặc trưng ─────────────────────────────────────────────────────
    num_raw, cat_raw = build_raw_feature_pool(df_train_raw)
    df_train_eng, eng_cols = engineer_business_features(df_train_raw)
    num_p, cat_p = num_raw + eng_cols, cat_raw
    log(f"Tập đặc trưng: {len(num_p)} numerical (gồm {len(eng_cols)} engineered) "
        f"+ {len(cat_p)} categorical = {len(num_p) + len(cat_p)} tổng.")
    log(f"  numerical: {num_p}")
    log(f"  categorical: {cat_p}")

    # ── 3. IV table ──────────────────────────────────────────────────────────
    log("Đang tính Information Value…")
    df_iv_input = df_valid.copy()
    df_iv_input, _ = engineer_business_features(df_iv_input)
    iv_cols = num_p + cat_p + [CREDIT_INFO_TARGET_COL]
    iv_table = compute_iv_table(df_iv_input[iv_cols], CREDIT_INFO_TARGET_COL, good_class=1)
    iv_table.to_csv(RESULTS_DIR / "credit_info_iv_table.csv", index=False, encoding="utf-8-sig")
    log(f"IV table đã lưu — phân bố mức: \n{iv_table['level'].value_counts()}")
    log(f"Top 5 IV cao nhất:\n{iv_table.head(5)}")

    # ── 4. Missing / outlier / correlation (EDA) ────────────────────────────
    log("Đang tính thống kê missing/outlier/correlation…")
    missing_pct = (df_train_eng[num_p + cat_p].isnull().mean() * 100).sort_values(ascending=False)
    missing_pct.to_csv(RESULTS_DIR / "credit_info_missing_pct.csv", header=["missing_pct"], encoding="utf-8-sig")

    desc = df_train_eng[num_p].describe().T[["mean", "std", "50%"]].rename(columns={"50%": "median"})
    desc["missing_pct"] = missing_pct.reindex(num_p)
    desc.to_csv(RESULTS_DIR / "credit_info_describe_numerical.csv", encoding="utf-8-sig")

    outlier_rows = []
    for c in num_p:
        s = df_train_eng[c].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((s < lo) | (s > hi)).sum()
        outlier_rows.append({"feature": c, "n_outlier": int(n_out), "pct_outlier": _pct(n_out / len(s))})
    outlier_df = pd.DataFrame(outlier_rows).sort_values("pct_outlier", ascending=False)
    outlier_df.to_csv(RESULTS_DIR / "credit_info_outliers.csv", index=False, encoding="utf-8-sig")

    corr = df_train_eng[num_p].corr(numeric_only=True)
    corr.to_csv(RESULTS_DIR / "credit_info_correlation.csv", encoding="utf-8-sig")
    corr_pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                      .stack().rename("corr").reset_index())
    corr_pairs.columns = ["feature_1", "feature_2", "corr"]
    corr_pairs = corr_pairs.reindex(corr_pairs["corr"].abs().sort_values(ascending=False).index)
    corr_pairs.head(15).to_csv(RESULTS_DIR / "credit_info_correlation_top_pairs.csv", index=False, encoding="utf-8-sig")
    log(f"Top 5 cặp tương quan mạnh nhất:\n{corr_pairs.head(5)}")

    # ── EDA figures ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    top_missing = missing_pct.head(15)[::-1]
    ax.barh(top_missing.index, top_missing.values, color="#e67e22")
    ax.set_xlabel("% giá trị thiếu")
    ax.set_title("Top 15 biến có tỷ lệ thiếu cao nhất — Thông tin tín dụng")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_credit_info_missing.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Ma trận tương quan — biến số học (Thông tin tín dụng)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_credit_info_correlation.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = sorted(df_valid[CREDIT_INFO_TARGET_COL].unique())
    data_box = [df_valid.loc[df_valid[CREDIT_INFO_TARGET_COL] == g, "Lãi suất"].dropna() for g in groups]
    ax.boxplot(data_box, showfliers=False)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([f"Nhóm {g}" for g in groups])
    ax.set_ylabel("Lãi suất (%)")
    ax.set_title("Phân phối Lãi suất theo nhóm nợ — Thông tin tín dụng")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "eda_credit_info_boxplot_laisuat.png", dpi=150)
    plt.close(fig)
    log("Đã lưu 3 hình EDA vào luanvan_latex/figures/")

    # ── 5. Train/validation split (80/20) + test độc lập 20260531 ──────────
    X_all, y_all = prepare_credit_info(df_train_eng, CREDIT_INFO_TARGET_COL, num_p, cat_p)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE,
    )
    log(f"Train: {X_train.shape}, Validation: {X_val.shape}")

    df_test_eng, _ = engineer_business_features(df_test_raw)
    df_test_valid = df_test_eng.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()
    X_holdout = transform_credit_info(df_test_valid, num_p, cat_p)
    y_holdout = df_test_valid[CREDIT_INFO_TARGET_COL].astype(int).values - 1
    log(f"Test độc lập (20260531): {X_holdout.shape}")

    # ── 6. So sánh 6 mô hình (cấu hình mặc định) ────────────────────────────
    log("Đang huấn luyện & so sánh các mô hình (cấu hình mặc định)…")
    models = available_models()
    log(f"Mô hình khả dụng trong môi trường hiện tại: {list(models.keys())}")

    val_rows, holdout_rows, fitted_pipes = [], [], {}
    for label, key in models.items():
        t0 = time.time()
        pipe = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                              imbalance_strategy="custom",
                              custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        pipe.fit(X_train, y_train)
        fitted_pipes[key] = pipe

        y_pred_v = pipe.predict(X_val) + 1
        y_proba_v = pipe.predict_proba(X_val)
        m_v = compute_metrics(y_val + 1, y_pred_v, y_proba_v)
        val_rows.append({"model": label, "model_key": key, "accuracy": _acc(m_v),
                         "f1_macro": m_v["f1_macro"], "f1_weighted": m_v["f1_weighted"],
                         "roc_auc": m_v["roc_auc"]})

        y_pred_h = pipe.predict(X_holdout) + 1
        y_proba_h = pipe.predict_proba(X_holdout)
        m_h = compute_metrics(y_holdout + 1, y_pred_h, y_proba_h)
        holdout_rows.append({"model": label, "model_key": key, "accuracy": _acc(m_h),
                             "f1_macro": m_h["f1_macro"], "f1_weighted": m_h["f1_weighted"],
                             "roc_auc": m_h["roc_auc"]})
        log(f"  {label}: val Macro-F1={m_v['f1_macro']:.4f} | "
            f"holdout Macro-F1={m_h['f1_macro']:.4f} ({time.time()-t0:.1f}s)")

    val_df = pd.DataFrame(val_rows).sort_values("f1_macro", ascending=False)
    holdout_df = pd.DataFrame(holdout_rows).sort_values("f1_macro", ascending=False)
    val_df.to_csv(RESULTS_DIR / "credit_info_model_comparison_validation.csv", index=False, encoding="utf-8-sig")
    holdout_df.to_csv(RESULTS_DIR / "credit_info_model_comparison_holdout.csv", index=False, encoding="utf-8-sig")

    best_label = val_df.iloc[0]["model"]
    best_key = val_df.iloc[0]["model_key"]
    best_pipe = fitted_pipes[best_key]
    log(f"Mô hình tốt nhất (theo validation Macro-F1): {best_label}")

    # ── 7. Chi tiết theo nhóm nợ — mô hình tốt nhất ─────────────────────────
    y_pred_v = best_pipe.predict(X_val) + 1
    y_proba_v = best_pipe.predict_proba(X_val)
    m_v = compute_metrics(y_val + 1, y_pred_v, y_proba_v)
    report_v = pd.DataFrame(m_v["report"]).T
    report_v.to_csv(RESULTS_DIR / "credit_info_best_model_by_class_validation.csv", encoding="utf-8-sig")

    y_pred_h = best_pipe.predict(X_holdout) + 1
    y_proba_h = best_pipe.predict_proba(X_holdout)
    m_h = compute_metrics(y_holdout + 1, y_pred_h, y_proba_h)
    report_h = pd.DataFrame(m_h["report"]).T
    report_h.to_csv(RESULTS_DIR / "credit_info_best_model_by_class_holdout.csv", encoding="utf-8-sig")
    log(f"Chi tiết theo nhóm nợ (validation):\n{report_v}")
    log(f"Chi tiết theo nhóm nợ (holdout 20260531):\n{report_h}")

    # ── 8. Top-k theo importance ─────────────────────────────────────────────
    log("Đang đánh giá hiệu năng theo số lượng đặc trưng (top-k)…")
    importance = importance_from_pipeline(best_pipe)
    ordered_features = importance.index.tolist()
    n_feat = len(ordered_features)
    ks = sorted(set([k for k in [3, 5, 8, 11, 14, 17, n_feat] if k <= n_feat]))

    def _build_and_eval(subset):
        cat_sub = [c for c in cat_p if c in subset]
        num_sub = [c for c in num_p if c in subset]
        p = build_pipeline(best_key, cat_sub, num_sub, random_state=RANDOM_STATE,
                           imbalance_strategy="custom", custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(X_train[subset], y_train)
        y_pred = p.predict(X_val[subset]) + 1
        y_proba = p.predict_proba(X_val[subset])
        m = compute_metrics(y_val + 1, y_pred, y_proba)
        return {"accuracy": _acc(m), "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}

    topk_df = evaluate_performance_vs_k(_build_and_eval, ordered_features, ks)
    topk_df.to_csv(RESULTS_DIR / "credit_info_topk.csv", index=False, encoding="utf-8-sig")
    log(f"Top-k:\n{topk_df}")

    curve_df = cumulative_importance_curve(importance)
    k95 = recommend_k_for_target(curve_df, 95.0)
    k99 = recommend_k_for_target(curve_df, 99.0)
    log(f"Số đặc trưng để giữ 95% importance: {k95}/{n_feat} | 99%: {k99}/{n_feat}")

    # ── 9. Group importance (Gain) + priority ───────────────────────────────
    grp_df = group_importance_table(importance, feature_to_group)
    grp_df.to_csv(RESULTS_DIR / "credit_info_group_importance.csv", index=False, encoding="utf-8-sig")
    log(f"Group importance (Gain):\n{grp_df}")

    used_cols = set(num_p) | set(cat_p)
    group_missing = {}
    for gname, g in FEATURE_GROUPS.items():
        cols = [c for c in (g["numerical"] + g["categorical"]) if c in used_cols and c in df_train_eng.columns]
        if cols:
            group_missing[gname] = float(df_train_eng[cols].isnull().mean().mean() * 100)
    priority_df = data_investment_priority(grp_df, group_missing)
    priority_df.to_csv(RESULTS_DIR / "credit_info_priority.csv", index=False, encoding="utf-8-sig")
    log(f"Priority đầu tư dữ liệu:\n{priority_df}")

    # ── 10. SHAP top-15 + group SHAP ─────────────────────────────────────────
    log("Đang tính SHAP values cho mô hình tốt nhất…")
    try:
        shap_values, feat_names, X_disp = compute_shap_values(best_pipe, X_val, best_key, max_samples=500)
        shap_imp = np.abs(shap_values).mean(axis=(0, 2))
        shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": shap_imp}).sort_values(
            "mean_abs_shap", ascending=False)
        shap_df.to_csv(RESULTS_DIR / "credit_info_shap_top.csv", index=False, encoding="utf-8-sig")
        log(f"SHAP top 15:\n{shap_df.head(15)}")

        shap_grp = pd.DataFrame({"feature": feat_names, "importance": shap_imp})
        shap_grp["group"] = shap_grp["feature"].map(feature_to_group)
        shap_grp_agg = (shap_grp.groupby("group")["importance"].sum()
                        .sort_values(ascending=False).reset_index())
        shap_grp_agg["pct"] = (shap_grp_agg["importance"] / shap_grp_agg["importance"].sum() * 100).round(1)
        shap_grp_agg.to_csv(RESULTS_DIR / "credit_info_shap_group.csv", index=False, encoding="utf-8-sig")
        log(f"SHAP group importance:\n{shap_grp_agg}")
    except Exception as e:
        log(f"Bỏ qua SHAP (lỗi: {e})")

    # ── 11. Grid search hyperparameter (nhỏ gọn, per model) ─────────────────
    log("Đang chạy grid search hyperparameter…")
    grids = {
        "logistic": [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]],
        "decision_tree": [{"max_depth": d, "min_samples_leaf": l}
                          for d in [5, 8, 12, 20] for l in [5, 20]],
        "random_forest": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": l}
                          for n in [200, 400] for d in [10, 20, None] for l in [5, 15]],
        "xgboost": [{"max_depth": d, "learning_rate": lr, "n_estimators": n}
                   for d in [3, 6, 9] for lr in [0.03, 0.1] for n in [200, 400]],
        "lightgbm": [{"max_depth": d, "learning_rate": lr, "n_estimators": n}
                    for d in [3, 6, 9] for lr in [0.03, 0.1] for n in [200, 400]],
        "catboost": [{"depth": d, "learning_rate": lr, "iterations": it}
                    for d in [4, 6, 8] for lr in [0.03, 0.1] for it in [200, 400]],
    }

    grid_rows = []
    for label, key in models.items():
        for i, params in enumerate(grids.get(key, [])):
            try:
                p = build_pipeline(key, cat_p, num_p, random_state=RANDOM_STATE,
                                   imbalance_strategy="custom",
                                   custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY,
                                   model_params=params)
                p.fit(X_train, y_train)
                y_pred = p.predict(X_val) + 1
                y_proba = p.predict_proba(X_val)
                m = compute_metrics(y_val + 1, y_pred, y_proba)
                grid_rows.append({
                    "model": label, "model_key": key, "config_id": i, "params": str(params),
                    "accuracy": _acc(m),
                    "f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"],
                    "status": "OK",
                })
            except Exception as e:
                grid_rows.append({
                    "model": label, "model_key": key, "config_id": i, "params": str(params),
                    "accuracy": np.nan,
                    "f1_macro": np.nan, "f1_weighted": np.nan, "roc_auc": np.nan,
                    "status": f"ERROR: {e}",
                })
        log(f"  Grid search {label} hoàn tất ({len(grids.get(key, []))} cấu hình).")

    grid_df = pd.DataFrame(grid_rows)
    grid_df.to_csv(RESULTS_DIR / "credit_info_grid_search_results.csv", index=False, encoding="utf-8-sig")

    ok = grid_df[grid_df["status"] == "OK"]
    grid_best = ok.loc[ok.groupby("model")["f1_macro"].idxmax()].sort_values("f1_macro", ascending=False)
    grid_best.to_csv(RESULTS_DIR / "credit_info_grid_search_best.csv", index=False, encoding="utf-8-sig")
    log(f"Best per model (grid search):\n{grid_best[['model', 'params', 'f1_macro']]}")

    sens = ok.groupby("model")["f1_macro"].agg(["min", "max", "mean", "std"]).reset_index()
    sens["range"] = sens["max"] - sens["min"]
    sens.to_csv(RESULTS_DIR / "credit_info_grid_search_sensitivity.csv", index=False, encoding="utf-8-sig")
    log(f"Độ nhạy hyperparameter:\n{sens}")

    default_vs_best = val_df.merge(grid_best[["model", "accuracy", "f1_macro"]], on="model",
                                   suffixes=("_default", "_best"))
    default_vs_best["improve_pct"] = (
        (default_vs_best["f1_macro_best"] - default_vs_best["f1_macro_default"])
        / default_vs_best["f1_macro_default"] * 100
    ).round(2)
    default_vs_best.to_csv(RESULTS_DIR / "credit_info_grid_search_default_vs_best.csv", index=False, encoding="utf-8-sig")
    log(f"Default vs Best:\n{default_vs_best[['model', 'f1_macro_default', 'f1_macro_best', 'improve_pct']]}")

    log("HOÀN TẤT. Tất cả bảng số liệu đã lưu vào code/results/credit_info_*.csv")
    log("Hình EDA đã lưu vào luanvan_latex/figures/eda_credit_info_*.png")


if __name__ == "__main__":
    main()
