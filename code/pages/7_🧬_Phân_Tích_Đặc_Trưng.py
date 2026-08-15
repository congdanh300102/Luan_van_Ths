"""Trang 7 — Phân tích & Lựa chọn Đặc trưng — so sánh song song 2 bộ dữ liệu

Trả lời 4 câu hỏi nghiên cứu, chạy độc lập trên cả 2 bộ để đối chiếu kết luận:
  1. Nhiều đặc trưng hơn có luôn giúp mô hình tốt hơn không?
  2. Nhóm đặc trưng nghiệp vụ nào đóng góp lớn nhất cho việc phân loại nhóm nợ?
  3. Có thể giảm từ nhiều xuống bao nhiêu đặc trưng mà vẫn giữ gần hết hiệu năng?
  4. Ngân hàng nên ưu tiên thu thập những nhóm thông tin nào để đạt hiệu quả cao nhất?

Dataset 1 (Data_credit_rating_VN.xlsx) chỉ có 21 cột thô — quy mô rất khác
Dataset 2 (Thông tin tín dụng, 33 cột chung giữa 2 file train/test). Pool
"đầy đủ" của Dataset 1 gồm 11 đặc trưng đã chọn qua IV (dùng ở trang 2) + 2
đặc trưng từng bị loại vì IV quá thấp (SEX, LOAIKH) — đây là toàn bộ
candidate hợp lệ còn lại sau khi loại leakage/trùng lặp/hằng số, không phải
một pool tương đương quy mô với Dataset 2.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import io
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from config.config import (
    RANDOM_STATE, DATA_RAW, TARGET_COL, DROP_COLS,
    CATEGORICAL_COLS, NUMERICAL_COLS, TEST_SIZE,
    CREDIT_INFO_SMOTE_STRATEGY,
)
from src.preprocessing import prepare, DATASET1_FEATURE_GROUPS, dataset1_feature_to_group
from src.credit_info_preprocessing import FEATURE_GROUPS, feature_to_group
from src.models import build_pipeline, available_models
from src.evaluation import compute_metrics
from src.data_loader import get_raw_bytes
from src.feature_selection import (
    importance_from_pipeline, group_importance_table, plot_group_importance,
    cumulative_importance_curve, recommend_k_for_target, plot_cumulative_importance,
    evaluate_performance_vs_k, plot_performance_vs_k, data_investment_priority,
)

st.set_page_config(page_title="Phân tích Đặc trưng", page_icon="🧬", layout="wide")
st.title("🧬 Phân tích & Lựa chọn Đặc trưng")
st.markdown(
    "Trả lời 4 câu hỏi nghiên cứu về số lượng và nhóm đặc trưng — chạy song song "
    "trên **cả 2 bộ dữ liệu** để đối chiếu xem kết luận có nhất quán không, dù "
    "quy mô đặc trưng của 2 bộ chênh lệch rất lớn (13 vs 178)."
)

tab_ds1, tab_ds2 = st.tabs([
    "📁 Dataset 1 — Data_credit_rating_VN (tối đa 13 đặc trưng)",
    "📁 Dataset 2 — Thông tin tín dụng (33 cột chung train/test)",
])


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — Data_credit_rating_VN.xlsx
# ══════════════════════════════════════════════════════════════════════════════
with tab_ds1:
    st.markdown(
        "Dataset 1 chỉ có 21 cột thô. Sau khi loại leakage (`NHOMNO`, `NHOMNO_TCBS`), "
        "cột trùng lặp (`DUNO_QD`, `MIACCTTYPDESC`, `MJACCTTYPDESC`) và cột hằng số/vô "
        "nghĩa (`CURRENCYCD`, `ID_TIME`, `DESC_TIME`), pool đầy đủ còn lại chỉ có "
        "**13 đặc trưng khả dụng**: 11 đặc trưng đã chọn qua IV (dùng ở Trang 2) + "
        "2 đặc trưng từng bị loại vì IV quá thấp (`SEX`, `LOAIKH`) — đưa lại vào đây "
        "để kiểm chứng việc thêm đặc trưng yếu có giúp ích không."
    )

    @st.cache_data(show_spinner="Đang tải Dataset 1…")
    def _load_ds1(raw_bytes: bytes) -> pd.DataFrame:
        return pd.read_excel(io.BytesIO(raw_bytes))

    ds1_raw_bytes = get_raw_bytes(DATA_RAW)
    df_ds1_raw = _load_ds1(ds1_raw_bytes)

    # Pool đầy đủ = DROP_COLS gốc TRỪ 2 cột IV thấp (SEX, LOAIKH) — giữ chúng lại
    # làm candidate, không đụng tới DROP_COLS dùng ở Trang 2.
    full_drop_cols = [c for c in DROP_COLS if c not in ("SEX", "LOAIKH")]
    X_ds1_full, y_ds1_1based = prepare(df_ds1_raw, full_drop_cols, TARGET_COL)
    y_ds1 = y_ds1_1based - 1

    cat_full_ds1 = [c for c in CATEGORICAL_COLS + ["SEX", "LOAIKH"] if c in X_ds1_full.columns]
    num_full_ds1 = [c for c in NUMERICAL_COLS if c in X_ds1_full.columns]
    curated_11 = [c for c in CATEGORICAL_COLS + NUMERICAL_COLS if c in X_ds1_full.columns]

    st.caption(
        f"Pool đầy đủ: {len(num_full_ds1) + len(cat_full_ds1)} đặc trưng "
        f"({len(num_full_ds1)} numerical + {len(cat_full_ds1)} categorical) | "
        f"Bộ đã chọn qua IV (Trang 2): {len(curated_11)} đặc trưng."
    )

    ds1_model_options = available_models()
    c_sel1, c_sel2 = st.columns([2, 1])
    with c_sel1:
        ds1_model_label = st.selectbox(
            "Mô hình dùng để phân tích",
            options=list(ds1_model_options.keys()),
            index=(list(ds1_model_options.keys()).index("XGBoost")
                   if "XGBoost" in ds1_model_options else 0),
            key="ds1_model_sel",
        )
    ds1_model_key = ds1_model_options[ds1_model_label]

    if st.button("🚀 Huấn luyện trên pool đầy đủ (13 đặc trưng)", key="ds1_train_btn"):
        with st.spinner("Đang huấn luyện Dataset 1…"):
            X_train1, X_test1, y_train1, y_test1 = train_test_split(
                X_ds1_full, y_ds1, test_size=TEST_SIZE, stratify=y_ds1,
                random_state=RANDOM_STATE,
            )
            pipe1 = build_pipeline(ds1_model_key, cat_full_ds1, num_full_ds1,
                                   random_state=RANDOM_STATE,
                                   imbalance_strategy="smote_moderate")
            pipe1.fit(X_train1, y_train1)

            st.session_state.update({
                "ds1_pipe": pipe1, "ds1_model_key": ds1_model_key,
                "ds1_X_train": X_train1, "ds1_y_train": y_train1,
                "ds1_X_test": X_test1, "ds1_y_test": y_test1,
                "ds1_cat": cat_full_ds1, "ds1_num": num_full_ds1,
                "ds1_X_all": X_ds1_full,
            })
            st.session_state.pop("ds1_perf_vs_k", None)
            st.session_state.pop("ds1_compare", None)

    if "ds1_pipe" not in st.session_state:
        st.info("👆 Bấm **Huấn luyện** để bắt đầu phân tích Dataset 1.")
    else:
        pipe1 = st.session_state["ds1_pipe"]
        importance1 = importance_from_pipeline(pipe1)
        ordered1 = importance1.index.tolist()

        def _build_and_eval_ds1(subset: list) -> dict:
            cat_sub = [c for c in st.session_state["ds1_cat"] if c in subset]
            num_sub = [c for c in st.session_state["ds1_num"] if c in subset]
            p = build_pipeline(st.session_state["ds1_model_key"], cat_sub, num_sub,
                               random_state=RANDOM_STATE, imbalance_strategy="smote_moderate")
            X_tr, y_tr = st.session_state["ds1_X_train"], st.session_state["ds1_y_train"]
            X_te, y_te = st.session_state["ds1_X_test"], st.session_state["ds1_y_test"]
            p.fit(X_tr[subset], y_tr)
            y_pred  = p.predict(X_te[subset]) + 1
            y_proba = p.predict_proba(X_te[subset])
            y_true  = y_te + 1
            m = compute_metrics(y_true, y_pred, y_proba)
            return {"f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}

        d1q1, d1q2, d1q3, d1q4 = st.tabs([
            "1️⃣ Nhiều đặc trưng hơn có tốt hơn?",
            "2️⃣ Nhóm nào đóng góp lớn nhất?",
            "3️⃣ Giảm còn bao nhiêu đặc trưng?",
            "4️⃣ Nên ưu tiên thu thập gì?",
        ])

        with d1q1:
            st.markdown("#### Đường cong hiệu năng theo số lượng đặc trưng")
            st.caption(
                "Chỉ có tối đa 13 đặc trưng khả dụng nên đường cong ngắn hơn nhiều so "
                "với Dataset 2 — nhưng vẫn đủ để kiểm tra liệu thêm 2 đặc trưng IV thấp "
                "(SEX, LOAIKH) vào 11 đặc trưng đã chọn có cải thiện hiệu năng không."
            )
            default_ks1 = [k for k in [3, 5, 7, 9, 11, len(ordered1)] if k <= len(ordered1)]
            ks1 = st.multiselect("Các mốc số đặc trưng (k) cần thử", options=sorted(set(default_ks1)),
                                 default=sorted(set(default_ks1)), key="ds1_ks")

            if st.button("▶️ Chạy đánh giá theo k", type="primary",
                        disabled=len(ks1) == 0, key="ds1_run_k"):
                with st.spinner("Đang huấn luyện lại cho từng mốc k…"):
                    df_k1 = evaluate_performance_vs_k(_build_and_eval_ds1, ordered1, sorted(ks1))
                st.session_state["ds1_perf_vs_k"] = df_k1

            if "ds1_perf_vs_k" in st.session_state:
                df_k1 = st.session_state["ds1_perf_vs_k"]
                st.dataframe(df_k1.style.format({c: "{:.4f}" for c in df_k1.columns if c != "k"}),
                            use_container_width=True, hide_index=True)
                st.plotly_chart(plot_performance_vs_k(df_k1), use_container_width=True)

                best_k1 = int(df_k1.loc[df_k1["f1_macro"].idxmax(), "k"])
                peak_f1_1 = df_k1["f1_macro"].max()
                full_f1_1 = df_k1.loc[df_k1["k"] == df_k1["k"].max(), "f1_macro"].values
                verdict1 = (
                    f"Macro F1 cao nhất đạt ở **k = {best_k1}** ({peak_f1_1:.4f}). "
                    + ("Dùng toàn bộ 13 đặc trưng (kể cả 2 đặc trưng IV thấp) KHÔNG "
                       "cải thiện thêm — nhất quán với kết luận ở Dataset 2: nhiều "
                       "đặc trưng hơn không tự động tốt hơn."
                       if len(full_f1_1) and full_f1_1[0] <= peak_f1_1 + 1e-9 and best_k1 < len(ordered1)
                       else "Xem bảng/biểu đồ để đánh giá xu hướng cụ thể.")
                )
                st.markdown(f"**Nhận định:** {verdict1}")

        with d1q2:
            st.markdown("#### Đóng góp theo nhóm đặc trưng nghiệp vụ")
            grp_df1 = group_importance_table(importance1, dataset1_feature_to_group)
            st.dataframe(grp_df1, use_container_width=True, hide_index=True)
            st.plotly_chart(plot_group_importance(grp_df1), use_container_width=True)
            top_group1 = grp_df1.iloc[0]
            st.markdown(
                f"**Nhận định:** nhóm **{top_group1['group']}** đóng góp lớn nhất "
                f"({top_group1['pct']:.1f}% tổng importance)."
            )

        with d1q3:
            st.markdown("#### Đường cong % importance tích lũy")
            curve_df1 = cumulative_importance_curve(importance1)
            target_pct1 = st.slider("Mục tiêu giữ lại % importance", 80, 99, 95, key="ds1_target_pct")
            k_reco1 = recommend_k_for_target(curve_df1, target_pct1)
            st.plotly_chart(plot_cumulative_importance(curve_df1, target_pct1), use_container_width=True)
            st.success(
                f"Chỉ cần **{k_reco1}/{len(ordered1)}** đặc trưng "
                f"(giảm {100*(1 - k_reco1/len(ordered1)):.0f}%) để giữ lại "
                f"~{target_pct1}% tổng importance."
            )

            st.markdown("#### So sánh Pool đầy đủ (13) vs Top-k (đề xuất) vs Bộ đã chọn qua IV (11)")
            compare_ks1 = sorted(set([k_reco1, len(ordered1), len(curated_11)]))
            if st.button("▶️ Chạy so sánh 3 cấu hình", key="ds1_compare_btn"):
                with st.spinner("Đang huấn luyện lại 3 cấu hình…"):
                    rows1 = []
                    for k in compare_ks1:
                        subset = ordered1[:k] if k <= len(ordered1) else ordered1
                        m = _build_and_eval_ds1(subset)
                        tag = ("Bộ đã chọn qua IV (11, hiện tại)" if k == len(curated_11) and k != len(ordered1)
                               else f"Pool đầy đủ ({k} biến)" if k == len(ordered1)
                               else f"Top-{k} (đề xuất)")
                        rows1.append({"Cấu hình": tag, "k": k, **m})
                    st.session_state["ds1_compare"] = pd.DataFrame(rows1)

            if "ds1_compare" in st.session_state:
                df_c1 = st.session_state["ds1_compare"]
                st.dataframe(df_c1.style.format({c: "{:.4f}" for c in df_c1.columns if c not in ("Cấu hình", "k")}),
                            use_container_width=True, hide_index=True)

        with d1q4:
            st.markdown("#### Khuyến nghị ưu tiên đầu tư thu thập dữ liệu")
            st.caption(
                "Kết hợp mức đóng góp hiện tại (importance) với % giá trị thiếu theo "
                "từng nhóm nghiệp vụ trong Dataset 1."
            )
            used_cols1 = set(st.session_state["ds1_num"]) | set(st.session_state["ds1_cat"])
            group_missing1 = {}
            for gname, cols in DATASET1_FEATURE_GROUPS.items():
                present = [c for c in cols if c in used_cols1 and c in X_ds1_full.columns]
                if present:
                    group_missing1[gname] = float(X_ds1_full[present].isnull().mean().mean() * 100)

            grp_df1b = group_importance_table(importance1, dataset1_feature_to_group)
            priority_df1 = data_investment_priority(grp_df1b, group_missing1)
            st.dataframe(priority_df1, use_container_width=True, hide_index=True)

            top1 = priority_df1.iloc[0]
            st.markdown(
                f"**Khuyến nghị:** ưu tiên đầu tư thu thập/hoàn thiện dữ liệu nhóm "
                f"**{top1['Nhóm nghiệp vụ']}** trước — đóng góp {top1['% đóng góp (importance)']:.1f}% "
                f"importance nhưng hiện có {top1['% dữ liệu thiếu hiện tại']:.1f}% giá trị thiếu."
            )


# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — Thông tin tín dụng (33 cột chung train/test)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ds2:
    REQUIRED_KEYS = ["credit_info_pipe", "credit_info_X_train", "credit_info_X_test",
                     "credit_info_cat_cols", "credit_info_num_cols"]
    if not all(k in st.session_state for k in REQUIRED_KEYS):
        st.warning(
            "⚠️ Chưa có mô hình đã huấn luyện. Vào **Trang 5 — Mô hình Tín dụng** "
            "rồi bấm **Huấn luyện mô hình** trước khi quay lại đây."
        )
        st.stop()

    pipe        = st.session_state["credit_info_pipe"]
    model_key   = st.session_state["credit_info_model_key"]
    model_lbl   = st.session_state.get("credit_info_model_lbl", model_key)
    cat_cols    = st.session_state["credit_info_cat_cols"]
    num_cols    = st.session_state["credit_info_num_cols"]
    X_train     = st.session_state["credit_info_X_train"]
    y_train     = st.session_state["credit_info_y_train"]
    X_test      = st.session_state["credit_info_X_test"]
    y_test      = st.session_state["credit_info_y_test"]
    X_all       = st.session_state.get("credit_info_X_all", pd.concat([X_train, X_test]))

    all_features = num_cols + cat_cols
    importance = importance_from_pipeline(pipe)

    st.info(f"Đang phân tích mô hình **{model_lbl}**, huấn luyện trên **{len(all_features)}** đặc trưng.")

    q1, q2, q3, q4 = st.tabs([
        "1️⃣ Nhiều đặc trưng hơn có tốt hơn?",
        "2️⃣ Nhóm nào đóng góp lớn nhất?",
        "3️⃣ Giảm còn bao nhiêu đặc trưng?",
        "4️⃣ Nên ưu tiên thu thập gì?",
    ])

    ordered_features = importance.index.tolist()

    def _build_and_eval(subset: list) -> dict:
        cat_sub = [c for c in cat_cols if c in subset]
        num_sub = [c for c in num_cols if c in subset]
        # smote_moderate mặc định được hiệu chỉnh cho phân phối lớp của Dataset
        # 1 — không tương thích với cỡ mẫu lớn hơn nhiều của Dataset 2 (SMOTE
        # yêu cầu target >= số mẫu gốc). Dùng đúng chiến lược custom đã hiệu
        # chỉnh cho dữ liệu Thông tin tín dụng, nhất quán với Trang 5.
        p = build_pipeline(model_key, cat_sub, num_sub, random_state=RANDOM_STATE,
                           imbalance_strategy="custom",
                           custom_smote_strategy=CREDIT_INFO_SMOTE_STRATEGY)
        p.fit(X_train[subset], y_train)
        y_pred  = p.predict(X_test[subset]) + 1
        y_proba = p.predict_proba(X_test[subset])
        y_true  = y_test + 1
        m = compute_metrics(y_true, y_pred, y_proba)
        return {"f1_macro": m["f1_macro"], "f1_weighted": m["f1_weighted"], "roc_auc": m["roc_auc"]}

    with q1:
        st.markdown("#### Đường cong hiệu năng theo số lượng đặc trưng")
        st.caption(
            "Huấn luyện lại mô hình nhiều lần, mỗi lần chỉ dùng top-k đặc trưng quan "
            "trọng nhất (xếp theo importance của mô hình đầy đủ). Nếu đường cong đi "
            "ngang hoặc giảm sau một điểm nào đó, việc thêm đặc trưng từ đó trở đi "
            "không còn giúp ích — thậm chí có thể gây nhiễu."
        )
        default_ks = sorted(set([5, 10, 15, 20, 30, 45, 60, len(ordered_features)]))
        default_ks = [k for k in default_ks if k <= len(ordered_features)]
        ks = st.multiselect("Các mốc số đặc trưng (k) cần thử", options=default_ks, default=default_ks)

        if st.button("▶️ Chạy đánh giá theo k", type="primary", disabled=len(ks) == 0):
            with st.spinner("Đang huấn luyện lại cho từng mốc k…"):
                df_k = evaluate_performance_vs_k(_build_and_eval, ordered_features, sorted(ks))
            st.session_state["credit_info_perf_vs_k"] = df_k

        if "credit_info_perf_vs_k" in st.session_state:
            df_k = st.session_state["credit_info_perf_vs_k"]
            st.dataframe(df_k.style.format({c: "{:.4f}" for c in df_k.columns if c != "k"}),
                        use_container_width=True, hide_index=True)
            st.plotly_chart(plot_performance_vs_k(df_k), use_container_width=True)

            best_k = int(df_k.loc[df_k["f1_macro"].idxmax(), "k"])
            peak_f1 = df_k["f1_macro"].max()
            full_f1 = df_k.loc[df_k["k"] == df_k["k"].max(), "f1_macro"].values
            verdict = (
                f"Macro F1 cao nhất đạt ở **k = {best_k}** ({peak_f1:.4f}). "
                + ("Việc dùng toàn bộ đặc trưng KHÔNG cải thiện thêm so với mốc này — "
                   "trả lời câu hỏi 1: nhiều đặc trưng hơn không tự động tốt hơn."
                   if len(full_f1) and full_f1[0] <= peak_f1 + 1e-9 and best_k < ordered_features.__len__()
                   else "Xem bảng/biểu đồ để đánh giá xu hướng cụ thể.")
            )
            st.markdown(f"**Nhận định:** {verdict}")

    with q2:
        st.markdown("#### Đóng góp theo nhóm đặc trưng nghiệp vụ")
        grp_df = group_importance_table(importance, feature_to_group)
        st.dataframe(grp_df, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_group_importance(grp_df), use_container_width=True)
        top_group = grp_df.iloc[0]
        st.markdown(
            f"**Nhận định:** nhóm **{top_group['group']}** đóng góp lớn nhất "
            f"({top_group['pct']:.1f}% tổng importance)."
        )

    with q3:
        st.markdown("#### Đường cong % importance tích lũy")
        curve_df = cumulative_importance_curve(importance)
        target_pct = st.slider("Mục tiêu giữ lại % importance", 80, 99, 95)
        k_reco = recommend_k_for_target(curve_df, target_pct)
        st.plotly_chart(plot_cumulative_importance(curve_df, target_pct), use_container_width=True)
        st.success(
            f"Chỉ cần **{k_reco}/{len(ordered_features)}** đặc trưng "
            f"(giảm {100*(1 - k_reco/len(ordered_features)):.0f}%) để giữ lại "
            f"~{target_pct}% tổng importance."
        )

        st.markdown("#### So sánh Đầy đủ vs Top-k (đề xuất)")
        compare_ks = sorted(set([k_reco, len(ordered_features)]))
        if st.button("▶️ Chạy so sánh cấu hình"):
            with st.spinner("Đang huấn luyện lại các cấu hình…"):
                rows = []
                for k in compare_ks:
                    subset = ordered_features[:k] if k <= len(ordered_features) else ordered_features
                    m = _build_and_eval(subset)
                    tag = (f"Đầy đủ ({k} biến)" if k == len(ordered_features)
                           else f"Top-{k} (đề xuất)")
                    rows.append({"Cấu hình": tag, "k": k, **m})
                st.session_state["credit_info_compare"] = pd.DataFrame(rows)

        if "credit_info_compare" in st.session_state:
            df_c3 = st.session_state["credit_info_compare"]
            st.dataframe(df_c3.style.format({c: "{:.4f}" for c in df_c3.columns if c not in ("Cấu hình", "k")}),
                        use_container_width=True, hide_index=True)

    with q4:
        st.markdown("#### Khuyến nghị ưu tiên đầu tư thu thập dữ liệu")
        st.caption(
            "Kết hợp mức đóng góp hiện tại (importance) với chất lượng dữ liệu hiện "
            "có (% giá trị thiếu) theo từng nhóm nghiệp vụ. Nhóm vừa quan trọng vừa "
            "đang thiếu dữ liệu nhiều là nơi đầu tư mang lại lợi ích biên lớn nhất."
        )
        used_cols = set(num_cols) | set(cat_cols)
        group_missing = {}
        for gname, g in FEATURE_GROUPS.items():
            cols = [c for c in (g["numerical"] + g["categorical"]) if c in used_cols and c in X_all.columns]
            if cols:
                group_missing[gname] = float(X_all[cols].isnull().mean().mean() * 100)

        grp_df2 = group_importance_table(importance, feature_to_group)
        priority_df = data_investment_priority(grp_df2, group_missing)
        st.dataframe(priority_df, use_container_width=True, hide_index=True)

        top = priority_df.iloc[0]
        st.markdown(
            f"**Khuyến nghị:** ưu tiên đầu tư thu thập/hoàn thiện dữ liệu nhóm "
            f"**{top['Nhóm nghiệp vụ']}** trước — đóng góp {top['% đóng góp (importance)']:.1f}% "
            f"importance nhưng hiện có {top['% dữ liệu thiếu hiện tại']:.1f}% giá trị thiếu."
        )
