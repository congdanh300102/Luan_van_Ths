"""Trang 5 — Mô hình dự báo Nhóm Nợ trên tập dữ liệu Thông tin tín dụng
(train 20260430 / test độc lập 20260507)."""
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
    DATA_CREDIT_INFO_TRAIN, DATA_CREDIT_INFO_TEST,
    CREDIT_INFO_TARGET_COL, CREDIT_INFO_NUMERICAL_COLS, CREDIT_INFO_CATEGORICAL_COLS,
    CREDIT_INFO_SMOTE_STRATEGY, CREDIT_INFO_NHOMNO_LABELS, GROUP_COLORS,
)
from src.credit_info_preprocessing import (
    load_credit_info, prepare_credit_info, transform_credit_info,
    build_raw_feature_pool, engineer_business_features,
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

st.set_page_config(page_title="Mô hình Tín dụng", page_icon="📊", layout="wide")
st.title("📊 Mô hình Dự báo Nhóm Nợ — Thông tin tín dụng")
st.markdown(
    "Xây dựng và huấn luyện mô hình phân loại nhóm nợ **1–5** trên "
    "`Thông tin tín dụng 20260430.xlsx` (train/validation 80/20), sau đó đánh giá "
    "trên tập test độc lập `Thông tin tín dụng 20260507.xlsx`."
)


# ── Cache: đọc DataFrame từ bytes ────────────────────────────────────────────
@st.cache_data(show_spinner="Đang tải dữ liệu…")
def _load_df(raw_bytes: bytes) -> pd.DataFrame:
    return load_credit_info(raw_bytes)


@st.cache_resource(show_spinner="Đang tải mô hình…")
def _load_saved_model(key: str):
    p = MODEL_DIR / f"model_credit_info_{key}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _features_path(key: str) -> Path:
    return MODEL_DIR / f"model_credit_info_{key}_features.pkl"


# ── Sidebar ───────────────────────────────────────────────────────────────────
PAGE_IMBALANCE_OPTIONS = {
    "SMOTE tùy chỉnh theo tỷ lệ dữ liệu (khuyến nghị)": "credit_info_custom",
    **IMBALANCE_OPTIONS,
}

with st.sidebar:
    st.header("⚙️ Cấu hình")

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
        help=("Nhóm 3, 4 vẫn là nhóm hiếm nhất (~1-1,4% dữ liệu) — lặp lại "
              "Stratified K-Fold nhiều lần và lấy trung bình cho ước lượng "
              "Recall đáng tin cậy hơn so với 1 lần train/test split."),
    )
    credit_info_n_repeats = 3
    if run_repeated_cv:
        credit_info_n_repeats = st.number_input("Số lần lặp (n_repeats)", min_value=1, max_value=20, value=3)

train_btn = st.button("🚀 Huấn luyện mô hình", type="primary")


# ── Data loading với Cloud fallback ──────────────────────────────────────────
def _resolve_data(path: Path, uploader_key: str, uploader_label: str):
    """Ưu tiên: file local → st.file_uploader."""
    if path.exists():
        return _load_df(path.read_bytes())

    st.warning(f"⚠️ Không tìm thấy file dữ liệu trên server.\n\nUpload **`{path.name}`** để tiếp tục:")
    uploaded = st.file_uploader(uploader_label, type=["xlsx"], key=uploader_key)
    if uploaded is None:
        st.info("👆 Upload file để bắt đầu.")
        st.stop()
    return load_credit_info(uploaded.read())


df_raw = _resolve_data(DATA_CREDIT_INFO_TRAIN, "credit_info_train_upload", "Chọn file dữ liệu train (20260430)")


def _resolve_feature_set(df: pd.DataFrame):
    """Trả về (df_for_features, numerical_cols, categorical_cols, info: dict)."""
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
tab_eda, tab_train, tab_test, tab_imbalance, tab_shap, tab_score = st.tabs(
    ["📋 Tổng quan dữ liệu", "🤖 Huấn luyện & Đánh giá", "🧪 Test độc lập (20260507)",
     "⚖️ So sánh xử lý mất cân bằng", "🧭 Giải thích SHAP", "💳 Chấm điểm tín dụng"]
)

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    df_valid = df_raw.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()
    df_valid[CREDIT_INFO_TARGET_COL] = df_valid[CREDIT_INFO_TARGET_COL].astype(int)

    r, c = df_raw.shape
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số hồ sơ",     f"{r:,}")
    m2.metric("Số cột",       c)
    m3.metric("Hồ sơ hợp lệ", f"{len(df_valid):,}")
    m4.metric("Số nhóm nợ",   df_valid[CREDIT_INFO_TARGET_COL].nunique())

    st.markdown("#### Phân phối nhóm nợ (Nhóm nợ tự phân loại)")
    vc = df_valid[CREDIT_INFO_TARGET_COL].value_counts().sort_index()
    imbalance_ratio = int(vc.iloc[0] / vc.min()) if vc.min() > 0 else 0
    fig_cls = go.Figure(go.Bar(
        x=[CREDIT_INFO_NHOMNO_LABELS.get(k, str(k)) for k in vc.index],
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

    st.markdown("#### Bể đặc trưng hợp lệ — quy tắc loại trừ từ 33 cột chung train/test")
    st.caption(
        "Tập đặc trưng chỉ dùng 33 cột có mặt ở cả 2 file (train 20260430 có 41 "
        "cột, test 20260507 chỉ có 33 cột) — để mô hình huấn luyện trên train vẫn "
        "chấm điểm được trên tập test độc lập."
    )
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Cột chung train/test", feature_info["n_total_cols"])
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

    st.markdown("#### Phân phối biến số theo nhóm nợ")
    num_avail = [c for c in num_p if c in df_valid.columns]
    if num_avail:
        sel_feat = st.selectbox("Chọn biến", num_avail, key="eda_feat")
        fig_box = px.box(
            df_valid.assign(Nhóm=df_valid[CREDIT_INFO_TARGET_COL].map(lambda x: f"Nhóm {x}")),
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
            X_all, y_all = prepare_credit_info(df_features, CREDIT_INFO_TARGET_COL, num_p, cat_p)

            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, stratify=y_all,
                random_state=RANDOM_STATE,
            )

            if imbalance_choice == "credit_info_custom":
                pipe = build_pipeline(
                    model_key, cat_p, num_p, random_state=RANDOM_STATE,
                    imbalance_strategy="custom", custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY,
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

            credit_info_rcv_df, credit_info_rcv_summary = None, None
            if run_repeated_cv:
                with st.spinner(f"Repeated Stratified K-Fold ({credit_info_n_repeats} lần)…"):
                    credit_info_rcv_df, credit_info_rcv_summary = repeated_stratified_recall(
                        pipe, X_all, y_all, n_splits=CV_FOLDS,
                        n_repeats=credit_info_n_repeats, random_state=RANDOM_STATE,
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
                model_path = MODEL_DIR / f"model_credit_info_{model_key}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(pipe, f)
                with open(_features_path(model_key), "wb") as f:
                    pickle.dump({"cat_cols": cat_p, "num_cols": num_p}, f)
                (MODEL_DIR / "best_credit_info_model.txt").write_text(model_key)
                _load_saved_model.clear()
            except Exception:
                pass

            st.session_state.update({
                "credit_info_pipe":       pipe,
                "credit_info_y_true":     y_true,
                "credit_info_y_pred":     y_pred,
                "credit_info_y_proba":    y_proba,
                "credit_info_metrics":    metrics,
                "credit_info_model_lbl":  model_label,
                "credit_info_model_key":  model_key,
                "credit_info_cat_cols":   cat_p,
                "credit_info_num_cols":   num_p,
                "credit_info_X_train":    X_train,
                "credit_info_y_train":    y_train,
                "credit_info_X_test":     X_test,
                "credit_info_y_test":     y_test,
                "credit_info_X_test_raw": X_test,
                "credit_info_X_all":      X_all,
                "credit_info_y_all":      y_all,
                "credit_info_rcv_df":       credit_info_rcv_df,
                "credit_info_rcv_summary":  credit_info_rcv_summary,
            })
            st.session_state.pop("credit_info_imbalance_bm", None)
            st.session_state.pop("credit_info_holdout_metrics", None)

            st.success(
                f"✅ Hoàn tất! **{model_label}** — "
                f"Macro F1: **{metrics['f1_macro']:.4f}** | "
                f"ROC-AUC: **{metrics['roc_auc']:.4f}**"
            )
        except Exception as e:
            st.error(f"Lỗi huấn luyện: {e}")

# Tự động load model đã lưu từ session trước (nếu có)
if "credit_info_pipe" not in st.session_state:
    best_txt = MODEL_DIR / "best_credit_info_model.txt"
    if best_txt.exists():
        saved_key = best_txt.read_text().strip()
        saved_pipe = _load_saved_model(saved_key)
        if saved_pipe is not None:
            try:
                feat_meta_path = _features_path(saved_key)
                with open(feat_meta_path, "rb") as f:
                    meta = pickle.load(f)
                saved_cat, saved_num = meta["cat_cols"], meta["num_cols"]

                X_all, y_all = prepare_credit_info(df_features, CREDIT_INFO_TARGET_COL, saved_num, saved_cat)
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
                    "credit_info_pipe":       saved_pipe,
                    "credit_info_y_true":     y_true,
                    "credit_info_y_pred":     y_pred,
                    "credit_info_y_proba":    y_proba,
                    "credit_info_metrics":    metrics,
                    "credit_info_model_key":  saved_key,
                    "credit_info_model_lbl":  saved_lbl,
                    "credit_info_cat_cols":   saved_cat,
                    "credit_info_num_cols":   saved_num,
                    "credit_info_X_train":    X_train,
                    "credit_info_y_train":    y_train,
                    "credit_info_X_test":     X_test,
                    "credit_info_y_test":     y_test,
                    "credit_info_X_test_raw": X_test,
                    "credit_info_X_all":      X_all,
                    "credit_info_y_all":      y_all,
                    "credit_info_rcv_df":       None,
                    "credit_info_rcv_summary":  None,
                })
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Huấn luyện & Đánh giá
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    if "credit_info_metrics" not in st.session_state:
        st.info("👈 Nhấn **Huấn luyện mô hình** ở thanh bên để bắt đầu.")
    else:
        metrics = st.session_state["credit_info_metrics"]
        y_true  = st.session_state["credit_info_y_true"]
        y_pred  = st.session_state["credit_info_y_pred"]
        y_proba = st.session_state["credit_info_y_proba"]
        lbl     = st.session_state.get("credit_info_model_lbl", "")

        st.success(f"✅ Mô hình: **{lbl}** — {len(st.session_state.get('credit_info_num_cols', []) + st.session_state.get('credit_info_cat_cols', []))} đặc trưng")
        st.caption("Đánh giá trên tập validation 20% tách từ 20260430 (không phải tập test 20260507 — xem tab **Test độc lập**).")

        c1, c2, c3 = st.columns(3)
        c1.metric("Macro F1",    f"{metrics['f1_macro']:.4f}")
        c2.metric("Weighted F1", f"{metrics['f1_weighted']:.4f}")
        c3.metric("ROC-AUC",     f"{metrics['roc_auc']:.4f}")

        rcv_df = st.session_state.get("credit_info_rcv_df")
        if rcv_df is not None:
            rcv_summary = st.session_state.get("credit_info_rcv_summary", {})
            st.markdown(
                f"**📊 Repeated Stratified K-Fold "
                f"({rcv_summary.get('n_repeats')}×{rcv_summary.get('n_splits')}-fold) "
                f"— Recall theo nhóm nợ**"
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

        pipe = st.session_state.get("credit_info_pipe")
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
            y_train_ci = st.session_state.get("credit_info_y_train")
            support_counts = pd.Series(y_train_ci).value_counts() if y_train_ci is not None else pd.Series(dtype=int)
            rarest_default = support_counts.sort_values().index[:2].tolist()
            target_classes = st.multiselect(
                "Chọn nhóm (0-based: 0=Nhóm 1 … 4=Nhóm 5) cần tăng Recall",
                options=sorted(np.unique(y_true - 1)),
                default=rarest_default,
                key="credit_info_boost_target",
            )
            if st.button("Tìm boost tối ưu", key="credit_info_boost_btn"):
                if not target_classes:
                    st.warning("Chọn ít nhất 1 nhóm để boost.")
                else:
                    best_boost, best_score = tune_class_boost(y_true - 1, y_proba, target_classes)
                    y_pred_boosted = predict_with_class_weights(y_proba, best_boost) + 1

                    boost_display = {CREDIT_INFO_NHOMNO_LABELS.get(c + 1, str(c + 1)): w
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
# Tab 3: Test độc lập trên 20260507
# ══════════════════════════════════════════════════════════════════════════════
with tab_test:
    if "credit_info_pipe" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để đánh giá trên tập test độc lập.")
    else:
        st.markdown(
            "Đánh giá pipeline đã fit (tiền xử lý + mô hình, huấn luyện trên "
            "20260430) trên `Thông tin tín dụng 20260507.xlsx` — dữ liệu hoàn "
            "toàn tách biệt, không tham gia vào bước train/validation 80/20 ở "
            "trên. Đây là con số phản ánh đúng khả năng tổng quát hoá của mô hình."
        )
        eval_test_btn = st.button("📥 Nạp 20260507 và đánh giá", type="primary")

        if eval_test_btn:
            with st.spinner("Đang tải và đánh giá trên tập test…"):
                try:
                    df_test_raw = _resolve_data(
                        DATA_CREDIT_INFO_TEST, "credit_info_test_upload",
                        "Chọn file dữ liệu test (20260507)",
                    )
                    df_test_eng, _ = engineer_business_features(df_test_raw)
                    df_test_valid = df_test_eng.dropna(subset=[CREDIT_INFO_TARGET_COL]).copy()

                    cat_cols = st.session_state["credit_info_cat_cols"]
                    num_cols = st.session_state["credit_info_num_cols"]
                    X_holdout = transform_credit_info(df_test_valid, num_cols, cat_cols)
                    y_holdout_true = df_test_valid[CREDIT_INFO_TARGET_COL].astype(int).values

                    pipe = st.session_state["credit_info_pipe"]
                    y_holdout_pred  = pipe.predict(X_holdout) + 1
                    y_holdout_proba = pipe.predict_proba(X_holdout)
                    holdout_metrics = compute_metrics(y_holdout_true, y_holdout_pred, y_holdout_proba)

                    st.session_state["credit_info_holdout_metrics"] = holdout_metrics
                    st.session_state["credit_info_holdout_y_true"]  = y_holdout_true
                    st.session_state["credit_info_holdout_y_pred"]  = y_holdout_pred
                except Exception as e:
                    st.error(f"Lỗi đánh giá trên tập test: {e}")

        if "credit_info_holdout_metrics" in st.session_state:
            hm = st.session_state["credit_info_holdout_metrics"]
            hy_true = st.session_state["credit_info_holdout_y_true"]
            hy_pred = st.session_state["credit_info_holdout_y_pred"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số hồ sơ test", f"{len(hy_true):,}")
            c2.metric("Macro F1",      f"{hm['f1_macro']:.4f}")
            c3.metric("Weighted F1",   f"{hm['f1_weighted']:.4f}")
            c4.metric("ROC-AUC",       f"{hm['roc_auc']:.4f}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Confusion Matrix — tập test 20260507**")
                st.plotly_chart(plot_confusion_matrix(hy_true, hy_pred), use_container_width=True)
            with col_b:
                st.markdown("**Classification Report — tập test 20260507**")
                report_df = pd.DataFrame(hm.get("report", {})).T
                num_cols_r = report_df.select_dtypes(include=float).columns
                st.dataframe(
                    report_df.style.format({c: "{:.3f}" for c in num_cols_r}),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: So sánh xử lý mất cân bằng
# ══════════════════════════════════════════════════════════════════════════════
with tab_imbalance:
    if "credit_info_X_train" not in st.session_state:
        st.info("👈 Huấn luyện mô hình ít nhất 1 lần trước (cần tập train/test cố định) để so sánh.")
    else:
        st.markdown(
            f"So sánh nhiều chiến lược xử lý mất cân bằng, cùng mô hình "
            f"**{st.session_state.get('credit_info_model_lbl','')}** và cùng tập train/test đã tạo ở tab Huấn luyện."
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
                    st.session_state["credit_info_model_key"],
                    st.session_state["credit_info_cat_cols"], st.session_state["credit_info_num_cols"],
                    st.session_state["credit_info_X_train"], st.session_state["credit_info_y_train"],
                    st.session_state["credit_info_X_test"], st.session_state["credit_info_y_test"],
                    strategies,
                )
            st.session_state["credit_info_imbalance_bm"] = df_bm

        if "credit_info_imbalance_bm" in st.session_state:
            df_bm = st.session_state["credit_info_imbalance_bm"]
            fmt_cols = [c for c in df_bm.columns if c != "Chiến lược"]
            st.dataframe(df_bm.style.format({c: "{:.4f}" for c in fmt_cols}),
                        use_container_width=True, hide_index=True)
            st.plotly_chart(plot_imbalance_comparison(df_bm), use_container_width=True)
            st.plotly_chart(plot_recall_by_group(df_bm), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Tab 5: Giải thích SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_shap:
    if "credit_info_pipe" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để xem giải thích SHAP.")
    else:
        pipe        = st.session_state["credit_info_pipe"]
        shap_model_key = st.session_state["credit_info_model_key"]
        X_test_raw  = st.session_state["credit_info_X_test_raw"]

        if st.button("🧮 Tính SHAP values (tối đa 500 hồ sơ mẫu)"):
            with st.spinner("Đang tính SHAP values…"):
                try:
                    shap_values, feat_names, X_disp = compute_shap_values(pipe, X_test_raw, shap_model_key)
                    st.session_state["credit_info_shap"] = (shap_values, feat_names, X_disp)
                except Exception as e:
                    st.error(f"Không tính được SHAP cho mô hình này: {e}")

        if "credit_info_shap" in st.session_state:
            shap_values, feat_names, X_disp = st.session_state["credit_info_shap"]

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
# Tab 6: Chấm điểm tín dụng
# ══════════════════════════════════════════════════════════════════════════════
with tab_score:
    if "credit_info_y_proba" not in st.session_state:
        st.info("👈 Huấn luyện mô hình trước để xem kết quả chấm điểm.")
    else:
        y_true  = st.session_state["credit_info_y_true"]
        y_pred  = st.session_state["credit_info_y_pred"]
        y_proba = st.session_state["credit_info_y_proba"]

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
            file_name=f"credit_scores_{st.session_state.get('credit_info_model_key','model')}.csv",
            mime="text/csv",
        )
