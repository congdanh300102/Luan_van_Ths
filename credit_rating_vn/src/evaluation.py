"""Evaluation utilities — trả về Plotly figures (không save file)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

NHOMNO_LABELS = {
    1: "Nhóm 1",
    2: "Nhóm 2",
    3: "Nhóm 3",
    4: "Nhóm 4",
    5: "Nhóm 5",
}
GROUP_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    f1_macro    = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    report = classification_report(
        y_true, y_pred,
        target_names=[NHOMNO_LABELS.get(c, str(c)) for c in sorted(np.unique(y_true))],
        output_dict=True,
    )
    return {"f1_macro": f1_macro, "f1_weighted": f1_weighted,
            "roc_auc": auc, "report": report}


def plot_confusion_matrix(y_true, y_pred) -> go.Figure:
    classes = sorted(np.unique(y_true))
    labels  = [NHOMNO_LABELS.get(c, str(c)) for c in classes]
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    text = [[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)"
             for j in range(len(classes))] for i in range(len(classes))]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels, text=text,
        texttemplate="%{text}", colorscale="Blues",
        showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Dự báo", yaxis_title="Thực tế",
        height=420,
    )
    return fig


def plot_feature_importance(pipeline, top_n: int = 20) -> go.Figure | None:
    try:
        clf   = pipeline.named_steps["classifier"]
        prep  = pipeline.named_steps["preprocessor"]
        names = prep.get_feature_names()

        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_).mean(axis=0)
        else:
            return None

        idx = np.argsort(imp)[::-1][:top_n]
        fig = px.bar(
            x=imp[idx[::-1]], y=[names[i] for i in idx[::-1]],
            orientation="h",
            labels={"x": "Importance", "y": "Feature"},
            title=f"Top {top_n} Feature Importance",
            color=imp[idx[::-1]],
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=max(400, top_n * 22), showlegend=False)
        return fig
    except Exception:
        return None


def plot_model_comparison(results: list) -> go.Figure:
    df  = pd.DataFrame(results)
    metrics = ["f1_macro", "f1_weighted", "roc_auc"]
    labels  = {"f1_macro": "Macro F1", "f1_weighted": "Weighted F1", "roc_auc": "ROC-AUC"}

    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(
            name=labels[m],
            x=df["model"],
            y=df[m],
            text=df[m].map(lambda v: f"{v:.4f}"),
            textposition="auto",
        ))
    fig.update_layout(
        barmode="group", title="So sánh hiệu năng các mô hình",
        yaxis=dict(range=[0, 1]), height=400,
    )
    return fig


def cross_val_scores(pipeline, X, y,
                     cv_folds: int = 5, random_state: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=["f1_macro", "f1_weighted"],
        n_jobs=-1,
    )
    return {
        "f1_macro_mean":    results["test_f1_macro"].mean(),
        "f1_macro_std":     results["test_f1_macro"].std(),
        "f1_weighted_mean": results["test_f1_weighted"].mean(),
        "f1_weighted_std":  results["test_f1_weighted"].std(),
    }
