"""Giải thích mô hình bằng SHAP — hỗ trợ multiclass (5 nhóm nợ)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap

NHOMNO_LABELS = {0: "Nhóm 1", 1: "Nhóm 2", 2: "Nhóm 3", 3: "Nhóm 4", 4: "Nhóm 5"}

_TREE_MODELS = {"random_forest", "xgboost", "lightgbm", "decision_tree"}


def _normalize_shap(raw) -> np.ndarray:
    """Chuẩn hoá output SHAP (list theo class hoặc ndarray) về (n_samples, n_features, n_classes)."""
    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    raw = np.asarray(raw)
    if raw.ndim == 2:
        return raw[:, :, None]
    return raw


def compute_shap_values(pipe, X: pd.DataFrame, model_key: str, max_samples: int = 500):
    """
    Tính SHAP values cho pipeline đã fit.

    Returns
    -------
    shap_values : np.ndarray, shape (n_samples, n_features, n_classes)
    feature_names : list[str]
    X_display : pd.DataFrame — dữ liệu (đã transform) tương ứng với shap_values, dùng để hiển thị giá trị gốc
    """
    prep = pipe.named_steps["preprocessor"]
    clf  = pipe.named_steps["classifier"]
    feature_names = prep.get_feature_names()

    X_t = prep.transform(X)
    X_t_df = X_t if isinstance(X_t, pd.DataFrame) else pd.DataFrame(X_t, columns=feature_names)

    if len(X_t_df) > max_samples:
        X_t_df = X_t_df.sample(max_samples, random_state=42)

    if model_key == "catboost":
        from catboost import Pool
        cat_idx = prep.cat_feature_indices()
        pool = Pool(X_t_df, cat_features=cat_idx)
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(pool)
    elif model_key in _TREE_MODELS:
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_t_df)
    else:  # logistic — mô hình tuyến tính
        background = shap.sample(X_t_df, min(100, len(X_t_df)), random_state=42)
        explainer = shap.LinearExplainer(clf, background)
        raw = explainer.shap_values(X_t_df)

    shap_values = _normalize_shap(raw)
    return shap_values, feature_names, X_t_df.reset_index(drop=True)


def _mean_abs_importance(shap_values: np.ndarray, class_idx: int | None) -> np.ndarray:
    if class_idx is None:
        return np.abs(shap_values).mean(axis=(0, 2))
    return np.abs(shap_values[:, :, class_idx]).mean(axis=0)


def plot_shap_global(shap_values: np.ndarray, feature_names: list,
                     class_idx: int | None = None, top_n: int = 20) -> go.Figure:
    """Bar |SHAP| trung bình — toàn cục (class_idx=None) hoặc theo 1 nhóm nợ cụ thể."""
    imp = _mean_abs_importance(shap_values, class_idx)
    order = np.argsort(imp)[::-1][:top_n]

    title = "Tầm quan trọng đặc trưng (SHAP, trung bình toàn bộ nhóm nợ)"
    if class_idx is not None:
        title = f"Tầm quan trọng đặc trưng (SHAP) — {NHOMNO_LABELS.get(class_idx, class_idx)}"

    fig = px.bar(
        x=imp[order][::-1], y=[feature_names[i] for i in order][::-1],
        orientation="h", labels={"x": "Mean |SHAP value|", "y": "Đặc trưng"},
        title=title, color=imp[order][::-1], color_continuous_scale="Reds",
    )
    fig.update_layout(height=max(400, top_n * 22), showlegend=False)
    return fig


def plot_shap_group_importance(shap_values: np.ndarray, feature_names: list,
                               feature_to_group_fn, class_idx: int | None = None) -> go.Figure:
    """Gộp |SHAP| theo nhóm nghiệp vụ — trả lời 'nhóm đặc trưng nào đóng góp lớn nhất?'."""
    imp = _mean_abs_importance(shap_values, class_idx)
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df["group"] = df["feature"].map(feature_to_group_fn)
    grp = (df.groupby("group")["importance"].sum()
             .sort_values(ascending=False).reset_index())
    grp["pct"] = (grp["importance"] / grp["importance"].sum() * 100).round(1)

    fig = px.bar(
        grp, x="importance", y="group", orientation="h",
        text=grp["pct"].map(lambda v: f"{v:.1f}%"),
        labels={"importance": "Tổng |SHAP value|", "group": "Nhóm nghiệp vụ"},
        title="Đóng góp theo nhóm đặc trưng nghiệp vụ (SHAP)",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(360, len(grp) * 45), showlegend=False,
                      yaxis=dict(categoryorder="total ascending"))
    return fig


def plot_shap_local(shap_values: np.ndarray, feature_names: list, X_display: pd.DataFrame,
                    row_idx: int, class_idx: int, top_n: int = 10) -> go.Figure:
    """Giải thích cục bộ cho 1 hồ sơ — top yếu tố đẩy dự báo về phía 1 nhóm nợ cụ thể."""
    values = shap_values[row_idx, :, class_idx]
    order  = np.argsort(np.abs(values))[::-1][:top_n]

    feats  = [feature_names[i] for i in order][::-1]
    vals   = values[order][::-1]
    raw_vals = [X_display.iloc[row_idx][feature_names[i]] for i in order][::-1]
    colors = ["#e74c3c" if v > 0 else "#2980b9" for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:.3f} (giá trị: {rv})" for v, rv in zip(vals, raw_vals)],
        textposition="auto",
    ))
    fig.update_layout(
        title=f"Vì sao dự báo nghiêng về {NHOMNO_LABELS.get(class_idx, class_idx)}? "
              f"(đỏ = tăng rủi ro nhóm này, xanh = giảm)",
        xaxis_title="SHAP value", height=max(360, top_n * 34),
    )
    return fig
