"""Trang 5 — Mô hình dự báo Nhóm Nợ trên tập dữ liệu fct_l.xlsx"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from config.config import (
    MODEL_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS, SCORE_BANDS,
    DATA_FCT_L, DATA_FCT_L_PROCESSED,
    FCT_L_TARGET_COL, FCT_L_NUMERICAL_COLS, FCT_L_CATEGORICAL_COLS,
    FCT_L_SMOTE_ORIGINAL, FCT_L_SMOTE_30K, FCT_L_NHOMNO_LABELS, GROUP_COLORS,
)
from src.fct_l_preprocessing import (
    prepare_fct_l, build_raw_feature_pool, engineer_business_features,
    feature_to_group, FEATURE_GROUPS, ID_LEAKAGE_COLS, DATE_COLS,
)
from src.models import build_pipeline, available_models, IMBALANCE_OPTIONS, compute_sample_weights
from src.evaluation import (
    compute_metrics, plot_confusion_matrix, plot_feature_importance,
    repeated_stratified_recall, predict_with_class_weights, tune_class_boost,
)
from src.scoring import (
    build_score_df, plot_score_distribution,
    plot_score_by_group, proba_to_score, classify_score,
)
from src.imbalance_benchmark import benchmark_strategies, plot_imbalance_comparison, plot_recall_by_group
from src.explainability import compute_shap_values, plot_shap_global, plot_shap_group_importance, plot_shap_local

st.set_page_config(page_title="Mô hình FCT_L", page_icon="📊", layout="wide")
st.title("📊 Mô hình Dự báo Nhóm Nợ — FCT_L")
st.markdown("Xây dựng và huấn luyện mô hình phân loại nhóm nợ **1–5** trên tập dữ liệu `fct_l.xlsx`.")


# ── Cache: đọc DataFrame từ bytes ────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dữ liệu…")
def _load_df(raw_bytes: bytes, is_csv: bool) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    if is_csv:
        return pd.read_csv(buf, low_memory=False)
    return pd.read_excel(buf)


@st.cache_resource(show_spinner="Đang tải mô hình FCT_L…")
def _load_saved_model(key: str):
    p = MODEL_DIR / f"model_fct_l_{key}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _features_path(key: str) -> Path:
    return MODEL_DIR / f"model_fct_l_{key}_features.pkl"


# ── Sidebar ───────────────────────────────────────────────────────────────────
PAGE_IMBALANCE_OPTIONS = {
    "SMOTE tùy chỉnh theo tỷ lệ FCT_L (khuyến nghị)": "fct_l_custom",
    **IMBALANCE_OPTIONS,
}

with st.sidebar:
    st.header("⚙️ Cấu hình")

    want_30k = st.radio(
        "Nguồn dữ liệu",
        ["fct_l_30k.csv (30,000 hồ sơ — khuyến nghị)", "fct_l.xlsx (5,400 hồ sơ gốc)"],
        index=0,
        help="Tập 30k được tạo bằng Stratified Bootstrap, giữ nguyên tỉ lệ nhóm nợ.",
    )
    prefer_30k = "30k" in want_30k

    feature_set_label = st.radio(
        "Bộ đặc trưng",
        ["Rút gọn (16 biến, đã chọn bằng IV)", "Đầy đủ + kỹ thuật đặc trưng (~90 biến)"],
        index=0,
        help="Bộ đầy đủ dùng toàn bộ 178 cột thô sau khi loại ID/khóa thay thế, "
             "leakage (vd. số ngày quá hạn) và cột hằng số, cộng thêm các đặc "
             "trưng kỹ thuật sinh theo nghiệp vụ (tỷ lệ sử dụng hạn mức, tỷ lệ "
             "thu hồi nợ, độ mới của giải ngân…).",
    )
    use_full_set = feature_set_label.startswith("Đầy đủ")

    MODEL_OPTIONS = available_models()
    model_label = st.selectbox(
        "Chọn mô hình",
        options=list(MODEL_OPTIONS.keys()),
        index=list(MODEL_OPTIONS.keys()).index("Random Forest") if "Random Forest" in MODEL_OPTIONS else 0,
    )
    model_key = MODEL_OPTIONS[model_label]

    imbalance_label = st.selectbox(
        "Chiến lược xử lý mất cân bằng",
        options=list(PAGE_IMBALANCE_OPTIONS.keys()),
        index=0,
    )
    imbalance_choice = PAGE_IMBALANCE_OPTIONS[imbalance_label]

    run_repeated_cv = st.checkbox(
        "Repeated Stratified K-Fold cho nhóm hiếm (chậm hơn)",
        value=False,
        help=("Nhóm 3, 4 ở fct_l chỉ có 11-13 (bộ gốc) hoặc ~60-72 (bộ 30k) "
              "hồ sơ — Recall đo trên 1 lần train/test split có phương sai "
              "rất lớn. Lặp lại Stratified K-Fold nhiều lần và lấy trung "
              "bình cho ước lượng đáng tin cậy hơn."),
    )
    fct_l_n_repeats = 3
    if run_repeated_cv:
        fct_l_n_repeats = st.number_input("Số lần lặp (n_repeats)", min_value=1, max_value=20, value=3)

    st.markdown("---")
    st.markdown("**Phân phối nhóm nợ (CLASSIFICATION)**")
    dist_info = {
        "5k":  {"N1": "5,247 (97.2%)", "N2": "87 (1.6%)", "N3": "11 (0.2%)", "N4": "13 (0.2%)", "N5": "23 (0.4%)"},
        "30k": {"N1": "~29,151 (97.2%)", "N2": "~483 (1.6%)", "N3": "~60 (0.2%)", "N4": "~72 (0.2%)", "N5": "~129 (0.4%)"},
    }
    for g, v in dist_info["30k" if prefer_30k else "5k"].items():
        color = list(GROUP_COLORS.values())[int(g[1]) - 1]
        st.markdown(
            f"<span style='background:{color};color:#fff;padding:1px 6px;"
            f"border-radius:3px'>{g}</span> {v}",
            unsafe_allow_html=True,
        )

train_btn = st.button("🚀 Huấn luyện mô hình", type="primary")


# ── Data loading với Cloud fallback ──────────────────────────────────────────
def _resolve_data(prefer_30k: bool):
    """
    Ưu tiên: file local → fallback về xlsx local → st.file_uploader.
    Trả về (df, is_30k_sized).
    """
    # 1. Thử 30k nếu được chọn
    if prefer_30k and DATA_FCT_L_PROCESSED.exists():
        return _load_df(DATA_FCT_L_PROCESSED.read_bytes(), is_csv=True), True

    # 2. Thử fct_l.xlsx gốc
    if DATA_FCT_L.exists():
        if prefer_30k:
            st.info(
                "ℹ️ `fct_l_30k.csv` không tìm thấy cục bộ — "
                "đang dùng `fct_l.xlsx` gốc (5,400 hồ sơ)."
            )
        return _load_df(DATA_FCT_L.read_bytes(), is_csv=False), False

    # 3. Không có file nào → uploader
    st.warning(
        "⚠️ Không tìm thấy file dữ liệu trên server.\n\n"
        "Upload **`fct_l.xlsx`** (hoặc `fct_l_30k.csv`) để tiếp tục:"
    )
    uploaded = st.file_uploader(
        "Chọn file dữ liệu FCT_L",
        type=["xlsx", "csv"],
        key="fctl_upload",
        help="fct_l.xlsx — 5,400 hồ sơ gốc | fct_l_30k.csv — 30,000 hồ sơ (khuyến nghị)",
    )
    if uploaded is None:
        st.info("👆 Upload file để bắt đầu.")
        st.stop()

    raw_bytes = uploaded.read()
    is_csv = uploaded.name.endswith(".csv")
    df = _load_df(raw_bytes, is_csv)
    is_large = len(df) > 10_000
    return df, is_large


df_raw, actual_30k = _resolve_data(prefer_30k)

# SMOTE strategy dựa trên kích thước thực tế của data đã load
smote_strategy = FCT_L_SMOTE_30K if actual_30k else FCT_L_SMOTE_ORIGINAL


def _resolve_feature_set(df: pd.DataFrame):
    """Trả về (df_for_features, numerical_cols, categorical_cols, info: dict) theo lựa chọn sidebar."""
    if not use_full_set:
        num_p = [c for c in FCT_L_NUMERICAL_COLS if c in df.columns]
        cat_p = [c for c in FCT_L_CATEGORICAL_COLS if c in df.columns]
        return df, num_p, cat_p, None

    num_raw, cat_raw = build_raw_feature_pool(df)
    df_eng, eng_cols = engineer_business_features(df)
    num_p = num_raw + eng_cols
    cat_p = cat_raw
    info = {
        "n_total_cols": df.shape[1],
        "n_id_leakage": len([c for c in ID_LEAKAGE_COLS if c in df.columns]),
        "n_date_cols": len([c for c in DATE_COLS if c in df.columns]),
        "n_raw_numerical": len(num_raw),
        "n_raw_categorical": len(cat_raw),
        "n_engineered": len(eng_cols),
        "eng_cols": eng_cols,
    }
    return df_eng, num_p, cat_p, info


df_features, num_p, cat_p, feature_info = _resolve_feature_set(df_raw)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_eda, tab_train, tab_imbalance, tab_shap, tab_score = st.tabs(
    ["📋 Tổng quan dữ liệu", "🤖 Huấn luyện & Đánh giá",
     "⚖️ So sánh xử lý mất cân bằng", "🧭 Giải thích SHAP", "💳 Chấm điểm tín dụng"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    df_valid = df_raw.dropna(subset=[FCT_L_TARGET_COL]).copy()
    df_valid[FCT_L_TARGET_COL] = df_valid[FCT_L_TARGET_COL].astype(int)

    r, c = df_raw.shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số hồ sơ",     f"{r:,}")
    m2.metric("Số cột",       c)
    m3.metric("Hồ sơ hợp lệ", f"{len(df_valid):,}")
    m4.metric("Số nhóm nợ",   df_valid[FCT_L_TARGET_COL].nunique())

    st.markdown("#### Phân phối nhóm nợ (CLASSIFICATION)")
    vc = df_valid[FCT_L_TARGET_COL].value_counts().sort_index()
    imbalance_ratio = int(vc.iloc[0] / vc.iloc[-1]) if vc.iloc[-1] > 0 else 0
    fig_cls = go.Figure(go.Bar(
        x=[FCT_L_NHOMNO_LABELS.get(k, str(k)) for k in vc.index],
        y=vc.values,
        marker_color=[GROUP_COLORS.get(k, "#999") for k in vc.index],
        text=[f"{v:,} ({v/len(df_valid)*100:.1f}%)" for v in vc.values],
        textposition="auto",
    ))
    fig_cls.update_layout(
        xaxis_title="Nhóm nợ", yaxis_title="Số lượng", height=360,
        annotations=[{
            "x": 0.5, "y": 1.08, "xref": "paper", "yref": "paper",
            "text": f"<b>Imbalance ratio: {imbalance_ratio:,}x</b>",
            "showarrow": False, "font": {"size": 13, "color": "red"},
        }],
    )
    st.plotly_chart(fig_cls, use_container_width=True)

    if feature_info is not None:
        st.markdown("#### Bể đặc trưng đầy đủ — quy tắc loại trừ từ 178 cột thô")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Tổng cột thô", feature_info["n_total_cols"])
        e2.metric("Loại: ID/khóa/leakage", feature_info["n_id_leakage"])
        e3.metric("Loại: cột ngày (→ sinh đặc trưng)", feature_info["n_date_cols"])
        e4.metric("Còn lại đưa vào bể đặc trưng", len(num_p) + len(cat_p))
        st.caption(
            f"Trong đó {feature_info['n_raw_numerical'] + feature_info['n_raw_categorical']} "
            f"cột thô (đã loại cột hằng số/gần hằng số) + {feature_info['n_engineered']} "
            f"đặc trưng kỹ thuật mới sinh theo nghiệp vụ: `{', '.join(feature_info['eng_cols'])}`."
        )

        st.markdown("#### Số lượng đặc trưng theo nhóm nghiệp vụ")
        group_counts = []
        used = set(num_p) | set(cat_p)
        for gname, g in FEATURE_GROUPS.items():
            n_in_group = len([c for c in g["numerical"] + g["categorical"] if c in used])
            if n_in_group > 0:
                group_counts.append({"Nhóm nghiệp vụ": gname, "Số đặc trưng": n_in_group})
        st.dataframe(pd.DataFrame(group_counts).sort_values("Số đặc trưng", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.markdown("#### Thống kê các biến đặc trưng được dùng (bộ rút gọn)")
        feat_cols = num_p + cat_p
        feat_df = pd.DataFrame({
            "Cột":      feat_cols,
            "Loại":     ["Số học"] * len(num_p) + ["Phân loại"] * len(cat_p),
            "Null (%)": [f"{df_raw[c].isnull().mean()*100:.1f}%" if c in df_raw.columns else "—" for c in feat_cols],
            "Unique":   [df_raw[c].nunique() if c in df_raw.columns else "—" for c in feat_cols],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("#### Phân phối biến số theo nhóm nợ")
    num_avail = [c for c in num_p if c in df_valid.columns]
    if num_avail:
        sel_feat = st.selectbox("Chọn biến", num_avail, key="eda_feat")
        fig_box = px.box(
            df_valid.assign(Nhóm=df_valid[FCT_L_TARGET_COL].map(lambda x: f"Nhóm {x}")),
            x="Nhóm", y=sel_feat, color="Nhóm",
            title=f"Phân phối {sel_feat} theo nhóm nợ",
        )
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Huấn luyện khi nhấn button
# ══════════════════════════════════════════════════════════════════════════════
if train_btn:
    with st.spinner(f"Đang chuẩn bị dữ liệu và huấn luyện {model_label}…"):
        try:
            X_all, y_all = prepare_fct_l(df_features, FCT_L_TARGET_COL, num_p, cat_p)

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                random_state=RANDOM_STATE,
            )

            if imbalance_choice == "fct_l_custom":
                pipe = build_pipeline(
                    model_key, cat_p, num_p, random_state=RANDOM_STATE,
                    imbalance_strategy="custom", custom_smote_strategy=smote_strategy,
                )
            else:
                pipe = build_pipeline(
                    model_key, cat_p, num_p, random_state=RANDOM_STATE,
                    imbalance_strategy=imbalance_choice,
                )

            # XGBoost không có tham số class_weight built-in (khác
            # LogisticRegression/RandomForest/LightGBM) — phải tự tính
            # sample_weight và truyền qua fit() để cost-sensitive thực sự
            # có hiệu lực khi chọn chiến lược "class_weight".
            use_sample_weight = model_key == "xgboost" and imbalance_choice == "class_weight"
            sample_weight_fn = compute_sample_weights if use_sample_weight else None

            fct_l_rcv_df, fct_l_rcv_summary = None, None
            if run_repeated_cv:
                with st.spinner(f"Repeated Stratified K-Fold ({fct_l_n_repeats} lần)…"):
                    fct_l_rcv_df, fct_l_rcv_summary = repeated_stratified_recall(
                        pipe, X_all, y_all, n_splits=CV_FOLDS,
                        n_repeats=fct_l_n_repeats, random_state=RANDOM_STATE,
                        sample_weight_fn=sample_weight_fn,
                    )

            if use_sample_weight:
                pipe.fit(X_train, y_train, classifier__sample_weight=compute_sample_weights(y_train))
            else:
                pipe.fit(X_train, y_train)

            y_pred  = pipe.predict(X_test) + 1
            y_proba = pipe.predict_proba(X_test)
            y_true  = y_test + 1
            metrics = compute_metrics(y_true, y_pred, y_proba)

            # Lưu model + danh sách cột đã dùng (không dừng nếu lỗi ghi file trên Cloud)
            try:
                model_path = MODEL_DIR / f"model_fct_l_{model_key}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(pipe, f)
                with open(_features_path(model_key), "wb") as f:
                    pickle.dump({"cat_cols": cat_p, "num_cols": num_p, "use_full_set": use_full_set}, f)
                (MODEL_DIR / "best_fct_l_model.txt").write_text(model_key)
                _load_saved_model.clear()
            except Exception:
                pass

            st.session_state.update({
                "fct_l_pipe":       pipe,
                "fct_l_y_true":     y_true,
                "fct_l_y_pred":     y_pred,
                "fct_l_y_proba":    y_proba,
                "fct_l_metrics":    metrics,
                "fct_l_model_lbl":  model_label,
                "fct_l_model_key":  model_key,
                "fct_l_cat_cols":   cat_p,
                "fct_l_num_cols":   num_p,
                "fct_l_X_train":    X_train,
                "fct_l_y_train":    y_train,
                "fct_l_X_test":     X_test,
                "fct_l_y_test":     y_test,
                "fct_l_X_test_raw": X_test,
                "fct_l_X_all":      X_all,
                "fct_l_y_all":      y_all,
                "fct_l_use_full_set": use_full_set,
                "fct_l_rcv_df":       fct_l_rcv_df,
                "fct_l_rcv_summary":  fct_l_rcv_summary,
            })
            st.session_state.pop("fct_l_imbalance_bm", None)

            st.success(
                f"✅ Hoàn tất! **{model_label}** — "
                f"Macro F1: **{metrics['f1_macro']:.4f}** | "
                f"ROC-AUC: **{metrics['roc_auc']:.4f}**"
            )
        except Exception as e:
            st.error(f"Lỗi huấn luyện: {e}")

# Tự động load model đã lưu từ session trước (nếu có)
if "fct_l_pipe" not in st.session_state:
    best_txt = MODEL_DIR / "best_fct_l_model.txt"
    if best_txt.exists():
        saved_key = best_txt.read_text().strip()
        saved_pipe = _load_saved_model(saved_key)
        if saved_pipe is not None:
            try:
                feat_meta_path = _features_path(saved_key)
                if feat_meta_path.exists():
                    with open(feat_meta_path, "rb") as f:
                        meta = pickle.load(f)
                    saved_cat, saved_num = meta["cat_cols"], meta["num_cols"]
                    saved_df_for_features = df_features if meta.get("use_full_set") else df_raw
                else:
                    # Model cũ lưu trước khi có bộ đầy đủ — giả định bộ rút gọn 16 biến
                    saved_cat = [c for c in FCT_L_CATEGORICAL_COLS if c in df_raw.columns]
                    saved_num = [c for c in FCT_L_NUMERICAL_COLS if c in df_raw.columns]
                    saved_df_for_features = df_raw

                X_all, y_all = prepare_fct_l(saved_df_for_features, FCT_L_TARGET_COL, saved_num, saved_cat)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                    random_state=RANDOM_STATE,
                )
                y_pred  = saved_pipe.predict(X_test) + 1
                y_proba = saved_pipe.predict_proba(X_test)
                y_true  = y_test + 1
                metrics = compute_metrics(y_true, y_pred, y_proba)
                saved_lbl = next((l for l, k in available_models().items() if k == saved_key), saved_key)
                st.session_state.update({
                    "fct_l_pipe":       saved_pipe,
                    "fct_l_y_true":     y_true,
                    "fct_l_y_pred":     y_pred,
                    "fct_l_y_proba":    y_proba,
                    "fct_l_metrics":    metrics,
                    "fct_l_model_key":  saved_key,
                    "fct_l_model_lbl":  saved_lbl,
                    "fct_l_cat_cols":   saved_cat,
                    "fct_l_num_cols":   saved_num,
                    "fct_l_X_train":    X_train,
                    "fct_l_y_train":    y_train,
                    "fct_l_X_test":     X_test,
                    "fct_l_y_test":     y_test,
                    "fct_l_X_test_raw": X_test,
                    "fct_l_X_all":      X_all,
                    "fct_l_y_all":      y_all,
                    "fct_l_rcv_df":       None,
                    "fct_l_rcv_summary":  None,
                })
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Huấn luyện & Đánh giá
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if "fct_l_metrics" not in st.session_state:
        st.info("👈 Nhấn **Huấn luyện mô hình** ở thanh bên để bắt đầu.")
    else:
        metrics = st.session_state["fct_l_metrics"]
        y_true  = st.session_state["fct_l_y_true"]
        y_pred  = st.session_state["fct_l_y_pred"]
        y_proba = st.session_state["fct_l_y_proba"]
        lbl     = st.session_state.get("fct_l_model_lbl", "")

        st.success(f"✅ Mô hình: **{lbl}** — {len(st.session_state.get('fct_l_num_cols', []) + st.session_state.get('fct_l_cat_cols', []))} đặc trưng")

        c1, c2, c3 = st.columns(3)
        c1.metric("Macro F1",    f"{metrics['f1_macro']:.4f}")
        c2.metric("Weighted F1", f"{metrics['f1_weighted']:.4f}")
        c3.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")

        rcv_df = st.session_state.get("fct_l_rcv_df")
        if rcv_df is not None:
            rcv_summary = st.session_state.get("fct_l_rcv_summary", {})
            st.markdown(
                f"**📊 Repeated Stratified K-Fold "
                f"({rcv_summary.get('n_repeats')}×{rcv_summary.get('n_splits')}-fold) "
                f"— Recall theo nhóm nợ**"
            )
            st.caption(
                "Đánh giá qua nhiều lần lặp thay vì 1 lần train/test split — với "
                "nhóm 3, 4 chỉ có vài chục hồ sơ, Recall của 1 lần split đơn lẻ có "
                "phương sai rất lớn."
            )
            st.dataframe(rcv_df, use_container_width=True, hide_index=True)
            st.caption(
                f"Macro F1 trung bình qua các fold: "
                f"{rcv_summary.get('macro_f1_mean', float('nan')):.4f} ± "
                f"{rcv_summary.get('macro_f1_std', float('nan')):.4f}"
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Confusion Matrix**")
            fig_cm = plot_confusion_matrix(y_true, y_pred)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_b:
            st.markdown("**Classification Report**")
            report_df = pd.DataFrame(metrics.get("report", {})).T
            num_cols_r = report_df.select_dtypes(include=float).columns
            st.dataframe(
                report_df.style.format({c: "{:.3f}" for c in num_cols_r}),
                use_container_width=True,
            )

        pipe = st.session_state.get("fct_l_pipe")
        if pipe:
            fig_fi = plot_feature_importance(pipe, top_n=16)
            if fig_fi:
                st.markdown("**Feature Importance**")
                st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("#### Xác suất dự báo trung bình theo nhóm thực tế")
        proba_by_group = []
        for g in sorted(np.unique(y_true)):
            mask = y_true == g
            row  = {"Nhóm thực tế": f"Nhóm {g}"}
            for i in range(5):
                row[f"P(Nhóm {i+1})"] = y_proba[mask, i].mean().round(4)
            proba_by_group.append(row)
        st.dataframe(pd.DataFrame(proba_by_group), use_container_width=True, hide_index=True)

        with st.expander("🎯 Hiệu chỉnh ngưỡng cho nhóm hiếm (post-hoc class-boost)"):
            st.caption(
                "Tăng Recall nhóm hiếm bằng argmax có trọng số (dịch ranh giới quyết "
                "định) — không cần huấn luyện lại mô hình. Boost tìm được ở đây tối ưu "
                "ngay trên tập test hiện dùng để báo cáo nên chỉ mang tính minh hoạ "
                "hướng cải thiện; áp dụng thực tế cần tìm boost trên tập validation "
                "tách riêng khỏi tập test cuối cùng."
            )
            y_train_fct = st.session_state.get("fct_l_y_train")
            support_counts = pd.Series(y_train_fct).value_counts() if y_train_fct is not None else pd.Series(dtype=int)
            rarest_default = support_counts.sort_values().index[:2].tolist()
            target_classes = st.multiselect(
                "Chọn nhóm (0-based: 0=Nhóm 1 … 4=Nhóm 5) cần tăng Recall",
                options=sorted(np.unique(y_true - 1)),
                default=rarest_default,
                key="fct_l_boost_target",
            )
            if st.button("Tìm boost tối ưu", key="fct_l_boost_btn"):
                if not target_classes:
                    st.warning("Chọn ít nhất 1 nhóm để boost.")
                else:
                    best_boost, best_score = tune_class_boost(y_true - 1, y_proba, target_classes)
                    y_pred_boosted = predict_with_class_weights(y_proba, best_boost) + 1

                    boost_display = {FCT_L_NHOMNO_LABELS.get(c + 1, str(c + 1)): w
                                     for c, w in best_boost.items()}
                    st.success(
                        f"Boost tối ưu {boost_display} → Macro F1 = {best_score:.4f} "
                        f"(gốc chưa boost: {metrics['f1_macro']:.4f})"
                    )

                    c1b, c2b = st.columns(2)
                    with c1b:
                        st.markdown("**Confusion Matrix — sau boost**")
                        st.plotly_chart(
                            plot_confusion_matrix(y_true, y_pred_boosted),
                            use_container_width=True,
                        )
                    with c2b:
                        st.markdown("**Classification Report — sau boost**")
                        report_boosted = pd.DataFrame(classification_report(
                            y_true, y_pred_boosted, output_dict=True, zero_division=0,
                        )).T
                        num_cols_b = report_boosted.select_dtypes(include=float).columns
                        st.dataframe(
                            report_boosted.style.format({c: "{:.3f}" for c in num_cols_b}),
                            use_container_width=True,
                        )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: So sánh xử lý mất cân bằng
# ══════════════════════════════════════════════════════════════════════════════
with tab_imbalance:
    if "fct_l_X_train" not in st.session_state:
        st.info("👈 Huấn luyện mô hình ít nhất 1 lần trước (cần tập train/test cố định) để so sánh.")
    else:
        st.markdown(
            f"So sánh nhiều chiến lược xử lý mất cân bằng, cùng mô hình "
            f"**{st.session_state.get('fct_l_model_lbl','')}** và cùng tập train/test đã tạo ở tab Huấn luyện."
        )
        chosen_labels = st.multiselect(
            "Chọn các chiến lược để so sánh",
            options=list(IMBALANCE_OPTIONS.keys()),
            default=list(IMBALANCE_OPTIONS.keys())[:4],
        )
        run_bm = st.button("🔄 Chạy so sánh mất cân bằng", type="primary", disabled=len(chosen_labels) == 0)

        if run_bm:
            strategies = {lbl: IMBALANCE_OPTIONS[lbl] for lbl in chosen_labels}
            with st.spinner("Đang huấn luyện lại cho từng chiến lược…"):
                df_bm = benchmark_strategies(
                    st.session_state["fct_l_model_key"],
                    st.session_state["fct_l_cat_cols"], st.session_state["fct_l_num_cols"],
                    st.session_state["fct_l_X_train"], st.session_state["fct_l_y_train"],
                    st.session_state["fct_l_X_test"], st.session_state["fct_l_y_test"],
                    strategies,
                )
            st.session_state["fct_l_imbalance_bm"] = df_bm

        if "fct_l_imbalance_bm" in st.session_state:
            df_bm = st.session_state["fct_l_imbalance_bm"]
            fmt_cols = [c for c in df_bm.columns if c != "Chiến lược"]
            st.dataframe(df_bm.style.format({c: "{:.4f}" for c in fmt_cols}),
                        use_container_width=True, hide_index=True)
            st.plotly_chart(plot_imbalance_comparison(df_bm), use_container_width=True)
            st.plotly_chart(plot_recall_by_group(df_bm), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: Giải thích SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "fct_l_pipe" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để xem giải thích SHAP.")
    else:
        pipe        = st.session_state["fct_l_pipe"]
        shap_model_key = st.session_state["fct_l_model_key"]
        X_test_raw  = st.session_state["fct_l_X_test_raw"]

        if st.button("🧮 Tính SHAP values (tối đa 500 hồ sơ mẫu)"):
            with st.spinner("Đang tính SHAP values…"):
                try:
                    shap_values, feat_names, X_disp = compute_shap_values(pipe, X_test_raw, shap_model_key)
                    st.session_state["fct_l_shap"] = (shap_values, feat_names, X_disp)
                except Exception as e:
                    st.error(f"Không tính được SHAP cho mô hình này: {e}")

        if "fct_l_shap" in st.session_state:
            shap_values, feat_names, X_disp = st.session_state["fct_l_shap"]

            class_sel = st.selectbox(
                "Xem theo nhóm nợ", ["Tất cả (trung bình)"] + [f"Nhóm {i+1}" for i in range(5)],
                key="shap_class_sel",
            )
            class_idx = None if class_sel.startswith("Tất cả") else int(class_sel.split()[-1]) - 1

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.plotly_chart(plot_shap_global(shap_values, feat_names, class_idx),
                                use_container_width=True)
            with col_g2:
                st.plotly_chart(
                    plot_shap_group_importance(shap_values, feat_names, feature_to_group, class_idx),
                    use_container_width=True,
                )

            st.markdown("#### 🔎 Giải thích cho 1 hồ sơ cụ thể (hỗ trợ quyết định)")
            row_idx = st.number_input("Chỉ số hồ sơ (trong mẫu SHAP)", 0, len(X_disp) - 1, 0)
            local_lbl = st.selectbox("Giải thích cho nhóm nợ", [f"Nhóm {i+1}" for i in range(5)], key="shap_local_cls")
            local_idx = int(local_lbl.split()[-1]) - 1
            st.plotly_chart(plot_shap_local(shap_values, feat_names, X_disp, row_idx, local_idx),
                            use_container_width=True)
        else:
            st.caption("Bấm nút phía trên để tính SHAP values cho mô hình hiện tại.")


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Chấm điểm tín dụng
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    if "fct_l_y_proba" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để xem kết quả chấm điểm.")
    else:
        y_true  = st.session_state["fct_l_y_true"]
        y_pred  = st.session_state["fct_l_y_pred"]
        y_proba = st.session_state["fct_l_y_proba"]

        df_scores = build_score_df(y_proba, y_pred, y_true)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Số hồ sơ",        f"{len(df_scores):,}")
        c2.metric("Điểm trung bình", f"{df_scores['diem_tin_dung'].mean():.0f}")
        c3.metric("Điểm thấp nhất",  f"{df_scores['diem_tin_dung'].min()}")
        c4.metric("Điểm cao nhất",   f"{df_scores['diem_tin_dung'].max()}")

        fig_dist = plot_score_distribution(df_scores)
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_box_s = plot_score_by_group(df_scores)
        if fig_box_s:
            st.plotly_chart(fig_box_s, use_container_width=True)

        st.markdown("#### Điểm trung bình theo nhóm nợ thực tế")
        tbl = (df_scores.groupby("nhom_thuc_te")["diem_tin_dung"]
                        .agg(["mean", "median", "std", "min", "max"])
                        .round(1))
        tbl.index = [f"Nhóm {i}" for i in tbl.index]
        st.dataframe(tbl, use_container_width=True)

        st.markdown("#### Phân bổ hạng tín dụng")
        grade_order = [b[2] for b in SCORE_BANDS]
        vc_g = df_scores["hang"].value_counts().reindex(grade_order, fill_value=0)
        grade_tbl = pd.DataFrame({
            "Hạng":      vc_g.index,
            "Mô tả":     [next((b[3] for b in SCORE_BANDS if b[2] == g), "") for g in vc_g.index],
            "Số lượng":  vc_g.values,
            "Tỷ lệ (%)": (vc_g.values / len(df_scores) * 100).round(1),
        })
        st.dataframe(grade_tbl, use_container_width=True, hide_index=True)

        csv = df_scores.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải kết quả CSV",
            data=csv,
            file_name=f"credit_scores_fct_l_{st.session_state.get('fct_l_model_key','model')}.csv",
            mime="text/csv",
        )
