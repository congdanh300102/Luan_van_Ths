"""
Lựa chọn/giảm đặc trưng + ưu tiên đầu tư dữ liệu.

Trả lời 4 câu hỏi nghiên cứu:
  Q1. Nhiều đặc trưng hơn có luôn giúp mô hình tốt hơn không?
      -> evaluate_performance_vs_k(): đường cong hiệu năng theo số biến.
  Q2. Nhóm đặc trưng nghiệp vụ nào đóng góp lớn nhất?
      -> group_importance_table().
  Q3. Có thể giảm từ 178 xuống bao nhiêu đặc trưng mà vẫn giữ gần hết hiệu năng?
      -> cumulative_importance_curve() + recommend_k_for_target().
  Q4. Ngân hàng nên ưu tiên thu thập nhóm thông tin nào?
      -> data_investment_priority().
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def importance_from_pipeline(pipe) -> pd.Series:
    """Lấy feature_importances_/|coef_| từ pipeline đã fit, index = tên đặc trưng."""
    clf   = pipe.named_steps["classifier"]
    prep  = pipe.named_steps["preprocessor"]
    names = prep.get_feature_names()

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        raise ValueError("Model không hỗ trợ feature importance trực tiếp.")
    return pd.Series(imp, index=names).sort_values(ascending=False)


def group_importance_table(importance: pd.Series, feature_to_group_fn) -> pd.DataFrame:
    """Gộp importance theo nhóm nghiệp vụ — trả lời câu hỏi 2."""
    df = importance.rename("importance").reset_index().rename(columns={"index": "feature"})
    df["group"] = df["feature"].map(feature_to_group_fn)
    grp = (df.groupby("group")["importance"].sum()
             .sort_values(ascending=False).reset_index())
    grp["pct"] = (grp["importance"] / grp["importance"].sum() * 100).round(1)
    return grp


def plot_group_importance(grp: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        grp, x="importance", y="group", orientation="h",
        text=grp["pct"].map(lambda v: f"{v:.1f}%"),
        labels={"importance": "Tổng importance", "group": "Nhóm nghiệp vụ"},
        title="Đóng góp theo nhóm đặc trưng nghiệp vụ",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(360, len(grp) * 45), showlegend=False,
                      yaxis=dict(categoryorder="total ascending"))
    return fig


def cumulative_importance_curve(importance: pd.Series) -> pd.DataFrame:
    """Số đặc trưng (theo thứ hạng importance giảm dần) vs % importance tích lũy."""
    imp_sorted = importance.sort_values(ascending=False)
    cum = imp_sorted.cumsum() / imp_sorted.sum()
    return pd.DataFrame({
        "rank": range(1, len(imp_sorted) + 1),
        "feature": imp_sorted.index,
        "importance": imp_sorted.values,
        "cum_pct": (cum.values * 100),
    })


def recommend_k_for_target(curve_df: pd.DataFrame, target_pct: float = 95.0) -> int:
    """Số đặc trưng tối thiểu để đạt target_pct% tổng importance."""
    hit = curve_df[curve_df["cum_pct"] >= target_pct]
    return int(hit["rank"].iloc[0]) if len(hit) else int(curve_df["rank"].iloc[-1])


def plot_cumulative_importance(curve_df: pd.DataFrame, target_pct: float = 95.0) -> go.Figure:
    k = recommend_k_for_target(curve_df, target_pct)
    fig = go.Figure(go.Scatter(
        x=curve_df["rank"], y=curve_df["cum_pct"],
        mode="lines+markers", line=dict(color="#2980b9"),
    ))
    fig.add_hline(y=target_pct, line_dash="dash", line_color="red",
                  annotation_text=f"{target_pct:.0f}% importance")
    fig.add_vline(x=k, line_dash="dash", line_color="green",
                  annotation_text=f"k = {k}")
    fig.update_layout(
        title="Đường cong % importance tích lũy theo số đặc trưng",
        xaxis_title="Số đặc trưng (xếp theo importance giảm dần)",
        yaxis_title="% importance tích lũy", height=420,
    )
    return fig


def evaluate_performance_vs_k(build_and_eval_fn, ordered_features: list, ks: list) -> pd.DataFrame:
    """
    Với mỗi k trong ks: train lại trên top-k đặc trưng đầu tiên trong
    ordered_features (đã xếp theo importance giảm dần), đánh giá và trả về
    bảng hiệu năng theo k — dữ liệu cho câu hỏi 1 và câu hỏi 3.

    build_and_eval_fn(feature_subset: list[str]) -> dict với ít nhất
    {"f1_macro": ..., "f1_weighted": ..., "roc_auc": ...}
    """
    rows = []
    for k in ks:
        subset = ordered_features[:k]
        metrics = build_and_eval_fn(subset)
        rows.append({"k": k, **metrics})
    return pd.DataFrame(rows)


def plot_performance_vs_k(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, name in [("f1_macro", "Macro F1"), ("f1_weighted", "Weighted F1"), ("roc_auc", "ROC-AUC")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["k"], y=df[col], mode="lines+markers", name=name))
    fig.update_layout(
        title="Hiệu năng mô hình theo số lượng đặc trưng sử dụng",
        xaxis_title="Số đặc trưng (k)", yaxis_title="Điểm số",
        yaxis=dict(range=[0, 1]), height=420,
    )
    return fig


def data_investment_priority(group_importance_df: pd.DataFrame,
                             group_missing_pct: dict) -> pd.DataFrame:
    """
    Kết hợp mức đóng góp (importance) với chất lượng dữ liệu hiện tại (null %)
    theo nhóm nghiệp vụ -> bảng xếp hạng ưu tiên đầu tư thu thập — câu hỏi 4.

    Logic: nhóm có importance cao NHƯNG dữ liệu hiện đang thiếu nhiều (null% cao)
    là nơi đầu tư mang lại lợi ích biên lớn nhất (importance cao, dư địa cải
    thiện dữ liệu cũng cao).
    """
    df = group_importance_df.copy()
    df["null_pct"] = df["group"].map(lambda g: group_missing_pct.get(g, np.nan))
    df["priority_score"] = df["pct"] * (df["null_pct"].fillna(0) / 100 + 0.1)
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["ưu_tiên"] = range(1, len(df) + 1)
    return df[["ưu_tiên", "group", "pct", "null_pct", "priority_score"]].rename(columns={
        "group": "Nhóm nghiệp vụ",
        "pct": "% đóng góp (importance)",
        "null_pct": "% dữ liệu thiếu hiện tại",
        "priority_score": "Điểm ưu tiên",
    })
